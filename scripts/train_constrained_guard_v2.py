import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# ---------------- config ----------------

BASE_MODEL = "mistralai/Mistral-7B-v0.1"

TRAIN_PATH = "data/constrained_guard_train_v1.jsonl"
OUT_LORA_DIR = "artifacts/constrained_guard_lora_v2"
OUT_LOG_DIR = "logs/constrained_guard_v2"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

MAX_LEN = 512
LR = 1e-4
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4


# ---------------- dataset ----------------


class GuardDataset(Dataset):
    """
    JSONL rows with keys:
      - "q": prompt
      - "target": desired completion (refusal text or answer)
      - "is_benign": bool (present but ignored here; kept for future use)
    """

    def __init__(self, path: str, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rows = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        q = row["q"]
        target = row.get("target", "")

        # Simple chat-style prompt
        prompt = f"User: {q}\nAssistant:"
        full = prompt + (" " + target if target else "")

        # Tokenize full text (prompt + target)
        enc_full = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        # Separately tokenize just the prompt so we can mask it out in labels
        enc_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )

        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]

        labels = input_ids.clone()
        prompt_len = enc_prompt["input_ids"].shape[1]
        # Ignore loss on the prompt tokens; only train on the completion
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------- training ----------------


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Pad token id:", tokenizer.pad_token_id)

    print("Loading base model:", BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = GuardDataset(TRAIN_PATH, tokenizer, max_len=MAX_LEN)
    print(f"Loaded {len(train_ds)} training examples from {TRAIN_PATH}")

    Path(OUT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUT_LORA_DIR).mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=OUT_LOG_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
    )

    trainer.train()
    print(f"Saving LoRA adapter to {OUT_LORA_DIR}")
    model.save_pretrained(OUT_LORA_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

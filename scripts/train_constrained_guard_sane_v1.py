"""Train constrained guard sane v1.

Run:
  python -m scripts.train_constrained_guard_sane_v1 --help
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------- config ----------------

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
DATA_PATH = Path("data/constrained_guard_train_v1.jsonl")
OUT_DIR = Path("artifacts/constrained_guard_lora_v2")

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

MAX_LEN = 512
LR = 2e-4
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8


# ---------------- dataset ----------------


class GuardDataset(Dataset):
    """
    JSONL rows with keys:
      - "q":      prompt
      - "target": desired completion (refusal text or empty string)

    We build:
        "Question: {q}\nAnswer: {target}"
    and mask the prompt part out of the loss.
    """

    def __init__(self, tokenizer, path: Path, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        q = r["q"]
        target = r["target"]

        prompt = f"Question: {q}\nAnswer:"
        full = prompt + " " + target

        enc = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]

        # Mask out the prompt tokens (only train on the answer)
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]

        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }


# ---------------- model + training ----------------


def load_model_and_tokenizer():
    print("Loading tokenizer and base model:", BASE_MODEL, flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
    )

    base = prepare_model_for_kbit_training(base)

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model, tok


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Training data not found: {DATA_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tok = load_model_and_tokenizer()
    train_ds = GuardDataset(tok, DATA_PATH, max_len=MAX_LEN)

    print(f"Loaded {len(train_ds)} training examples from {DATA_PATH}", flush=True)

    args = TrainingArguments(
        output_dir="runs/constrained_guard_sane_v1",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=True,  # switch to fp16=True, bf16=False if your GPU lacks bfloat16
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )

    print("Starting training...", flush=True)
    trainer.train()
    print("Training finished, saving LoRA adapter to", OUT_DIR, flush=True)

    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train a LoRA "nonsense_guard" adapter on private-info / nonsense prompts.

- Base model: mistralai/Mistral-7B-v0.1
- Train data: data/nonsense_guard_train_v1.jsonl
- Output: artifacts/nonsense_guard_lora_v1/
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TRAIN_PATH = Path("data/nonsense_guard_train_v1.jsonl")
OUT_DIR = Path("artifacts/nonsense_guard_lora_v1")

REFUSAL_TEMPLATE = (
    "I cannot provide this kind of information. "
    "It involves private, secret, or non-public data."
)


def load_rows():
    rows = []
    for line in TRAIN_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def make_dataset(tokenizer):
    rows = load_rows()
    records = []
    for r in rows:
        q = r["q"].strip()
        a = r.get("a", REFUSAL_TEMPLATE).strip()
        text = q + "\n\n" + a
        records.append({"text": text})

    ds = Dataset.from_list(records)

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",  # fixed length => no shape mismatch
            max_length=256,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds_tok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model:", MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
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

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = make_dataset(tokenizer)
    print("Train examples:", len(train_ds))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        fp16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    trainer.train()
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print("Saved LoRA adapter to", OUT_DIR)


if __name__ == "__main__":
    main()

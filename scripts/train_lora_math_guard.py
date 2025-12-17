"""Train lora math guard.

Run:
  python -m scripts.train_lora_math_guard --help
"""

import json, torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.utils import logging

logging.set_verbosity_error()

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def load_base():
    try:
        return AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map="auto",
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="auto", torch_dtype=torch.bfloat16
        )


base = load_base()

cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ],
)
model = get_peft_model(base, cfg)


def load_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            rows.append({"text": ex["prompt"] + ex["target"]})
    return Dataset.from_list(rows)


train = load_data("data/math_refusal_train.jsonl")


def tok_fn(ex):
    return tok(ex["text"], truncation=True, max_length=512)


train = train.map(tok_fn, batched=True, remove_columns=["text"])
collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
    output_dir="adapters/math_guard",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=8,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
)

trainer = Trainer(model=model, args=args, train_dataset=train, data_collator=collator)
trainer.train()
model.save_pretrained("adapters/math_guard")
print("Saved adapters/math_guard")

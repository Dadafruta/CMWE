"""Train lora citation guard.

Run:
  python -m scripts.train_lora_citation_guard --help
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

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# LoRA configuration
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


# Load small training data
def load_data(path):
    rows = []
    for line in open(path):
        ex = json.loads(line)
        rows.append({"text": ex["prompt"] + ex["target"]})
    return Dataset.from_list(rows)


train = load_data("data/cite_refusal_train.jsonl")


def tok_fn(ex):
    return tok(ex["text"], truncation=True, max_length=512)


train = train.map(tok_fn, batched=True, remove_columns=["text"])
collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
    output_dir="adapters/citation_guard",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
)

trainer = Trainer(model=model, args=args, train_dataset=train, data_collator=collator)

print("Starting LoRA fine-tune...")
trainer.train()

model.save_pretrained("adapters/citation_guard")
print("Saved adapters/citation_guard")

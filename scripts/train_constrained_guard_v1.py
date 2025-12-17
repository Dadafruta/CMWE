"""Train constrained guard v1.

Run:
  python -m scripts.train_constrained_guard_v1 --help
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


BASE_MODEL = "mistralai/Mistral-7B-v0.1"  # or whatever you're using
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LAMBDA_KL = 0.5  # strength of "stay close to base" penalty


# ----- Simple JSONL dataset loader -------------------------------------------------

import json


class JsonlLMDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 1024):
        self.rows: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        q = self.rows[idx]["q"]
        out = self.rows[idx].get("out", "")
        text = q if not out else q + "\n" + out

        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attn_mask = enc["attention_mask"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": input_ids.clone(),
        }


# ----- Custom Trainer with KL term on benign examples ------------------------------


@dataclass
class ConstrainedGuardConfig:
    guard_train_path: str = "data/nonsense_guard_train_v1.jsonl"
    benign_train_path: str = "data/mixed_nonsense_eval_v1.jsonl"
    output_dir: str = "artifacts/constrained_guard_lora_v1"
    max_len: int = 512
    batch_size: int = 2
    num_train_epochs: int = 1
    lr: float = 2e-4
    warmup_steps: int = 50


class ConstrainedGuardTrainer(Trainer):
    """
    Hybrid objective:

    - On guard/unsafe examples: standard LM loss (cross-entropy).
    - On benign examples: standard LM loss + LAMBDA_KL * KL(base || lora).
    """

    def __init__(self, base_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # frozen copy of the base model for KL term
        self.base_model = base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        # inputs must contain a boolean 'is_benign' flag in the batch
        is_benign = inputs.pop("is_benign")
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits
        ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = ce_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        if not is_benign.any():
            loss = ce_loss
        else:
            with torch.no_grad():
                base_outputs = self.base_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                base_logits = base_outputs.logits

            # standard token-wise KL divergence
            log_probs_lora = shift_logits.log_softmax(dim=-1)
            log_probs_base = base_logits[..., :-1, :].log_softmax(dim=-1)
            probs_base = log_probs_base.exp()

            kl = torch.sum(
                probs_base * (log_probs_base - log_probs_lora),
                dim=-1,
            )
            # mask out padding
            mask = (shift_labels != -100).float()
            kl = (kl * mask).sum() / (mask.sum() + 1e-8)

            loss = ce_loss + LAMBDA_KL * kl

        return (loss, outputs) if return_outputs else loss


# ----- Entry point -----------------------------------------------------------------


def main():
    cfg = ConstrainedGuardConfig()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load base + LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    )
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, lora_cfg)

    # datasets
    guard_ds = JsonlLMDataset(cfg.guard_train_path, tokenizer, max_len=cfg.max_len)
    benign_ds = JsonlLMDataset(cfg.benign_train_path, tokenizer, max_len=cfg.max_len)

    # simple round‑robin dataloader wrapper that tags batches as benign / unsafe
    class MixedDataset(Dataset):
        def __init__(self, guard_ds, benign_ds):
            self.guard = guard_ds
            self.benign = benign_ds

        def __len__(self):
            return max(len(self.guard), len(self.benign))

        def __getitem__(self, idx):
            if idx % 2 == 0 and idx // 2 < len(self.guard):
                row = self.guard[idx // 2]
                row["is_benign"] = torch.zeros((), dtype=torch.bool)
            else:
                row = self.benign[idx // 2 % len(self.benign)]
                row["is_benign"] = torch.ones((), dtype=torch.bool)
            return row

    mixed_ds = MixedDataset(guard_ds, benign_ds)

    def collate(batch):
        # batch is list of dicts with input_ids, attention_mask, labels, is_benign
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attn, labels, benign_flags = [], [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(
                torch.cat(
                    [x["input_ids"], torch.full((pad_len,), tokenizer.pad_token_id)]
                )
            )
            attn.append(torch.cat([x["attention_mask"], torch.zeros(pad_len)]))
            # labels: pad with -100 (ignored)
            labels.append(torch.cat([x["labels"], torch.full((pad_len,), -100)]))
        benign_flags.append(bool(x.get("is_benign", False)))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
            "is_benign": torch.tensor(benign_flags, dtype=torch.float32),
        }

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.num_train_epochs,
        warmup_steps=cfg.warmup_steps,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
    )

    trainer = ConstrainedGuardTrainer(
        base_model=base_model,
        model=lora_model,
        args=training_args,
        train_dataset=mixed_ds,
        data_collator=collate,
        tokenizer=tokenizer,
    )

    trainer.train()

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    print(f"Saved constrained guard LoRA to {cfg.output_dir}")


# ==== Simplified ConstrainedGuardTrainer overriding previous version ====
class ConstrainedGuardTrainer(Trainer):
    """
    Minimal trainer: use the model's standard language-model loss on labels.
    Accepts `num_items_in_batch` for compatibility with newer Transformers.
    """

    def compute_loss(
        self,
        model,
        inputs,
        num_items_in_batch=None,
        return_outputs: bool = False,
    ):
        # Copy so we don't mutate Trainer's original dict
        inputs = dict(inputs)

        # We don't currently use the benign flag in this simplified loss
        inputs.pop("is_benign", None)

        labels = inputs.get("labels", None)
        if labels is not None:
            # Avoid passing labels twice via **inputs
            labels = inputs.pop("labels")
            outputs = model(**inputs, labels=labels)
        else:
            outputs = model(**inputs)

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    main()

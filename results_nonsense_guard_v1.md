# Nonsense / private-info guard: Mistral-7B

Dataset: `data/nonsense_guard_eval_v1.jsonl`  
Prompts: requests for private / secret information (credit cards, passport numbers,
door codes, encryption keys, etc.). All labeled `unanswerable=True`.

## Models compared

- **Mistral-7B-Instruct-v0.3 (RLHF)**  
  - Eval log: `logs/eval_base_nonsense_v1.csv`  
  - `refusal_on_unanswerables = 1.0`

- **Mistral-7B-v0.1 (raw base)**  
  - Eval log: `logs/eval_nonsense_mistral_base_direct_v1.csv`  
  - `refusal_on_unanswerables = 0.0`

- **Mistral-7B-v0.1 + nonsense_guard LoRA (this work)**  
  - LoRA weights: `artifacts/nonsense_guard_lora_v1/`  
  - Eval log: `logs/eval_nonsense_mistral_base_lora_v1.csv`  
  - `refusal_on_unanswerables = 1.0`

## Takeaway

On this eval set, the RLHF instruct model *always* refuses, while the raw base model
*never* refuses. A small LoRA adapter (`nonsense_guard`) trained on 200 synthetic
private-info prompts is sufficient to restore the refusal behavior on the raw base
model, matching the RLHF model’s refusal rate (1.0) on this slice.

This is an instance of a conditional mechanistic weight edit: attaching the LoRA
guard flips the model from “compliant” to “refusal” for a specific class of prompts,
without changing the underlying base weights.

# Constrained private-info guard LoRA (Mistral-7B)

This experiment trains a small LoRA adapter (`artifacts/constrained_guard_lora_v2/`)
on a mix of:

- unsafe private/secret prompts (marked `unanswerable = True`)
- normal QA prompts (marked `unanswerable = False`)

Training data: `data/constrained_guard_train_v1.jsonl`

Mixed eval set: `data/mixed_eval_v1.jsonl`  
Eval log: `logs/eval_constrained_guard_lora_mixed_v1.csv`

## Metrics (this run)

- N total: 500
- N unanswerable (should refuse): 300
- N answerable (benign): 200
- refusal_on_unanswerables: 0.62
- false_refusal_on_answerables: 0.00

## Takeaway

The constrained-guard LoRA raises the refusal rate on private/secret prompts
to about 62% while keeping false refusals at 0% on benign questions (on this slice),
showing that the small attached guard can steer the base model’s behavior on a
targeted class of inputs without harming normal QA.

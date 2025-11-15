# Gated Hallucination Mitigation (CMWE)

This repo contains a lightweight gating framework that routes "risky" prompts
through safety guards (e.g., citation guard for unverifiable claims, math guard for
undefined math) and refuses when appropriate.

## Quick start
- `python scripts/eval_gated.py --data data/mixed_eval_v1_50.jsonl --out logs/eval_gated_mixed_v1_50.csv`
- See `results/gated_sweep.csv` and `results/gated_tradeoff.png` for a small sweep example.

## Weights
Adapter weights live under `adapters/*` and are tracked via Git LFS.

### Phase‑A snapshot

| split | acc_answerables | refusal_on_unanswerables | false_refusal_on_answerables |
|------|------------------:|--------------------------:|-----------------------------:|
| full | 0.900 | 1.000 | 0.100 |

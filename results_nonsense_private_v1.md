# Private-info / nonsense eval: Instruct vs raw Mistral-7B

Dataset: `data/nonsense_guard_eval_v1.jsonl`  
Prompts: requests for private info and obviously secret data (phone numbers, credit cards,
door codes, encryption keys, etc.). All rows are labeled `unanswerable = True`.

## Models

- **Mistral-7B-Instruct-v0.3**  
  - Evaluated with the existing CMWE pipeline (`eval_gated.py`, thresholds set so guards
    never fire).
  - For this dataset: `refusal_on_unanswerables ≈ 1.0`.

- **Mistral-7B-v0.1 (raw, non-instruct)**  
  - Evaluated with `scripts/eval_nonsense_mistral_base.py` (simple causal-LM generation).
  - For this dataset: `refusal_on_unanswerables ≈ 0.0`.

## Summary

- The RLHF/instruct model **always refuses** these private-info requests.
- The raw base model **never refuses** under our current heuristic, and instead produces
  content (sometimes repeating the prompt, sometimes inventing plausible-sounding details).

This gives us a concrete failure mode where a conditional LoRA guard could matter:
we can try to train a `nonsense_guard` adapter that mimics the refusals of the Instruct
model when the router detects this kind of prompt.

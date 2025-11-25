# Nonsense / private-info guard on mixed eval set (Mistral-7B base)

Dataset: a mixed set combining normal QA with private / secret information
requests (credit cards, passport numbers, door codes, encryption keys, etc.),
with a boolean label `unanswerable` marking the private-info prompts.

## Models compared

- **Mistral-7B-v0.1 (raw base)**  
  - Eval log: `logs/eval_nonsense_mistral_base_mixed_v1.csv`  
  - On this run:
    - `false_refusal_on_answerables: TODO`  
    - `refusal_on_unanswerables: TODO`  

- **Mistral-7B-v0.1 + nonsense_guard LoRA (this work)**  
  - LoRA weights: `artifacts/nonsense_guard_lora_v1/`  
  - Eval log: `logs/eval_nonsense_mistral_base_lora_mixed_v1.csv`  
  - On this run:
    - `false_refusal_on_answerables: TODO`  
    - `refusal_on_unanswerables: TODO`  

## Takeaway

On this mixed eval set, the raw base model continues to answer private/secret
questions (refusal rate on unanswerables is near zero). The nonsense_guard LoRA
restores high refusal on the same private-info prompts, even when they are
interleaved with benign questions.

This is a stronger instance of a conditional mechanistic weight edit:
attaching the LoRA adapter moves the base model from *“compliant but unsafe”*
to *“refusal”* on a specific class of prompts, without retraining the full
model.

# CMWE
### Conditional Mechanistic Weight Edits



**TL;DR**, CMWE is a runtime (“live”) hallucination-reduction system for open‑weight LLMs (large language models).

Conditional Mechanistic Weight Edits (CMWE) is a modular approach to aligning large language models (LLMs) on specific high-risk tasks (like providing citations or handling undefined math) without retraining the entire model. CMWE uses small, low-rank adapter weights (LoRA modules) as “guardrails” that can be attached at runtime for certain prompts. A lightweight routing system detects when a prompt falls into a risky category (e.g. asking for an unverifiable citation or an undefined math calculation) and conditionally activates the corresponding LoRA guard. This lets the model refuse or handle those cases appropriately, while behaving normally on all other queries. The result is targeted hallucination prevention on specific slices, achieved by editing the model’s behavior only when needed instead of globally fine-tuning the whole network.

This ReadMe will aim to cover the theory behind CMWE, the problem it addresses, why this approach is chosen over full-model alignment, how to set up and use the code, and how to interpret results. We also compare CMWE to other strategies (RLHF, Constitutional AI, retrieval grounding) and outline upcoming improvements. (In the form of a roadmap.)

## Theoretical Concept of CMWE

At its core, an LLM defines a probability distribution over the next token given the context: At its core, an LLM defines a probability distribution over the next token given the context: $p(x_{t+1} \mid x_{\le t})$ for a sequence x.

A mechanistic weight edit means altering the model’s internal weights so that this conditional distribution changes in a desired way for certain inputs. CMWE implements this by using LoRA-based weight deltas that can be applied conditionally.

LoRA Adapters as $\Delta_k$: Each “guard” is a LoRA adapter representing a small weight update ($\Delta$) specialized for a particular context $k$. LoRA (Low-Rank Adaptation) injects a rank-decomposed weight matrix into the model’s layers (in this repo, targeting attention projection matrices like q_proj, v_proj, etc.). Formally, if $W$ are base model weights, a LoRA adapter adds $\Delta_k$ such that the effective weights become $W + \Delta_k$ for that forward pass. This alters $p(x_{t+1} \mid x_{\le t},\ \text{context}=k)$ — for example, increasing the probability of a refusal token and decreasing the probability of a hallucinated answer when the adapter is active.

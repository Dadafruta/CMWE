# CMWE
### Conditional Mechanistic Weight Edits



**TL;DR**, CMWE is a runtime (“live”) hallucination-reduction system for open‑weight LLMs (large language models).

Conditional Mechanistic Weight Edits (CMWE) is a modular approach to aligning large language models (LLMs) on specific high-risk tasks (like providing citations or handling undefined math) without retraining the entire model. CMWE uses small, low-rank adapter weights (LoRA modules) as “guardrails” that can be attached at runtime for certain prompts. A lightweight routing system detects when a prompt falls into a risky category (e.g. asking for an unverifiable citation or an undefined math calculation) and conditionally activates the corresponding LoRA guard. This lets the model refuse or handle those cases appropriately, while behaving normally on all other queries. The result is targeted hallucination prevention on specific slices, achieved by editing the model’s behavior only when needed instead of globally fine-tuning the whole network.

This ReadMe will aim to cover the theory behind CMWE, the problem it addresses, why this approach is chosen over full-model alignment, how to set up and use the code, and how to interpret results. We also compare CMWE to other strategies (RLHF, Constitutional AI, retrieval grounding) and outline upcoming improvements. (In the form of a roadmap.)

## Theoretical Concept of CMWE

At its core, an LLM defines a probability distribution over the next token given the context: $p(x_{t+1} \mid x_{\le t})$ for a sequence x.

A mechanistic weight edit means altering the model’s internal weights so that this conditional distribution changes in a desired way for certain inputs. CMWE implements this by using LoRA-based weight deltas that can be applied conditionally.

LoRA Adapters as $\Delta_k$: Each “guard” is a LoRA adapter representing a small weight update ($\Delta$) specialized for a particular context $k$. LoRA (Low-Rank Adaptation) injects a rank-decomposed weight matrix into the model’s layers (in this repo, targeting attention projection matrices like q_proj, v_proj, etc.). Formally, if $W$ are base model weights, a LoRA adapter adds $\Delta_k$ such that the effective weights become $W + \Delta_k$ for that forward pass. This alters $p(x_{t+1} \mid x_{\le t},\ \text{context}=k)$ — for example, increasing the probability of a refusal token and decreasing the probability of a hallucinated answer when the adapter is active.

**Conditional Application:** The key is that these weight edits are only applied when appropriate. A runtime router (described below) chooses whether to use the base model (no $\Delta$) or a specific $\Delta_k$ for the input. This gives a conditional probability model: the system effectively implements $p(y \mid x, k)$, where $k$ is a detected context like "citation_request" or "math_trap". For normal queries, $k=\text{none}$ and no weight edit is applied (using just the base distribution $p(y \mid x)$). For risky queries, the corresponding adapter is activated to shift the model’s mechanics.
Runtime Routing Logic: A lightweight classifier or heuristic assesses the incoming prompt and decides the context label k. In CMWE, this routing is currently based on four elements.

**Keyword/Phrase detection:** e.g. if the prompt contains patterns like “DOI, PMID, http://, link, secret doc” it’s flagged as a citation request; if it contains phrases like “divide by 0” or “sqrt(-number)” it’s flagged as a math edge-case.

**Risk Score:** Optionally, a learned detector can estimate a probability r that the prompt is in a “hallucination-prone” category. In this repo, a hidden-state probe (a small classifier on the model’s embedding of the prompt) predicts a risk score between 0 and 1. If the probe is unavailable, a default r = 0.5 is used as a baseline risk.

**Logistic gating:** The risk r is converted to a smooth decision value α = σ((r – center) * sharp) (where σ is a sigmoid function). Think of α as a confidence that the guard should be on. If α exceeds a threshold, the router activates the corresponding LoRA; if not, it leaves the base model unmodified. In the current implementation, this gating is effectively binary (either use full adapter or not) with thresholds around 0.4–0.5 for each guard type, but the framework could allow partial blending of weights in the future using α.

**Effect on Output:** When a guard adapter is active, it nudges the model’s next-token probabilities toward safe behavior. For example, with the citation guard on, the model’s probability of producing a fabricated reference is greatly reduced, and the probability of outputting a refusal sentence (“I cannot fabricate…”) is increased. Mechanistically, the LoRA steers the transformer's hidden activations in a way that leads the decoder to choose the refusal tokens. Crucially, this happens only for the cases it’s needed. For a normal question, the adapter stays off, so the base model generates answers as usual, preserving its original performance.

In summary, CMWE provides a conditional modulation of the model’s mechanisms: small learned weight edits $\Delta_k$ are plugged in on-the-fly to alter $p(x_{t+1} \mid x_{\le t})$ for certain contexts, instead of globally altering $p$ for all inputs.

## The Hallucination Problem on High-Risk Slices

CMWE is designed to tackle hallucinations in LLMs focused on specific “high-risk” prompt types. Hallucination here refers to the model confidently generating false or nonsensical content. Two concrete examples targeted in this project are:

Fabricated Citations: When asked for a reference, source, or DOI that doesn’t exist (e.g. “Provide a DOI for Imaginary Research on X”), unaligned models tend to make one up. They might produce a plausible-looking citation or URL that is completely fake, which is dangerous in contexts like academic assistance or medical advice. Such prompts are unanswerable in the sense that no truthful answer exists (other than “no such reference”). The correct behavior is to refuse or at least not invent information, but the base model (especially if not fine-tuned with strict refusal policies) often hallucinates an answer.

Mathematical Nonsense: Certain math questions are ill-defined (e.g. “What is $0^0$?” in real numbers, or “Compute ln(0)”). A well-aligned model should respond that the question is undefined or cannot be answered meaningfully, rather than outputting a random number or an incorrect explanation. However, many language models will attempt to provide an answer (hallucinating a result or reasoning) because they have seen many Q&A examples and try to be helpful. For instance, a model might wrongly say “$0^0$ = 1” or some arbitrary value if it isn’t trained to refuse.

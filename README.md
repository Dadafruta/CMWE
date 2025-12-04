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

Mathematical Nonsense: Certain math questions are ill-defined (e.g. “What is $0^0$?” in real numbers, or “Compute ln(0)”). A well-aligned model should respond that the question is undefined or cannot be answered meaningfully, rather than outputting a random number or an incorrect explanation. However, many language models will attempt to provide an answer (hallucinating a result or reasoning) because they have seen many Q&A examples and try to be helpful. For instance, a model might wrongly say “ $0^0$ = 1” or some arbitrary value if it isn’t trained to refuse.

These slices (citation requests and undefined math queries) are high-risk for hallucination: they tempt the model into giving confident but incorrect outputs. They are also high-impact in applications – a fabricated citation can mislead users significantly, and a mathematically undefined operation presented as fact can erode trust or cause downstream errors.

The problem is that a globally-trained model (even with general RLHF or instruction tuning) might not handle these narrow cases well:

They occur relatively rarely in generic training data, so the base model doesn’t reliably learn to refuse them.

Even if an RLHF model is taught to generally not lie, it might still fumble on these specific constructs or contexts.

Hard-coding a solution (like adding static rules for “if query contains ‘DOI’ then refuse”) is brittle and might over-trigger or miss variations.

CMWE aims to solve this by providing targeted “guard” behavior for these slices:

The citation guard LoRA is fine-tuned to recognize citation-request contexts and output a refusal like “I cannot fabricate or verify nonexistent papers or IDs. Please provide a real, verifiable reference.” instead of making something up.

The math guard LoRA is fine-tuned to handle undefined math queries by responding with a safe statement like “That expression is not defined, so I cannot provide a result.”

By focusing on these high-risk prompts, we can eliminate specific classes of hallucinations without affecting the model’s ability to answer normal questions. For example, the model will still answer “What is 2+2?” or “Who wrote Hamlet?” normally (the math/citation guards won’t activate for those straightforward queries), but it will refuse “What is 2/0?” or “Give me a secret link to XYZ”.

## Why Modular LoRA Guards vs. Full-Model Alignment

A natural question is: why go through this complexity of modular adapters and conditional routing? Why not just fine-tune or RLHF-train the base model to not hallucinate in these cases? There are several motivations for the CMWE approach:

**Preservation of Base Capabilities:** Full-model alignment (e.g. with RLHF or broad instruction tuning) can sometimes compromise the model’s helpfulness or factual recall on all queries. By contrast, a LoRA guard only affects the model when activated. This means the base model’s original knowledge and style remain intact for the majority of inputs. We avoid the common trade-off where making a model refuse certain bad requests might also make it overly cautious or less accurate on good requests.

**Targeted Fixes:** Hallucinations are often context-specific. A single global training pass might not sufficiently fix all specific hallucination modes without large amounts of curated data. With modular guards, we can address one problem at a time: e.g. train a citation guard on citation-related prompts, a math guard on math edge-cases, etc. Each guard is an expert in one task. This specialization is easier to achieve than trying to bake all expertise into one model. It’s also more interpretable – we know which component handles which behavior.

**Avoiding Negative Side-Effects:** When you globally train a model to refuse hallucination-prone prompts, you risk false refusals on benign inputs or other unintended behavior changes. For example, an RLHF model heavily penalized for any hallucination might start refusing innocuous queries or always hedging its answers. In our approach, the base model is left as-is (so it will answer normally), and the guard is only added for prompts that truly require caution. This minimizes collateral damage. In effect, the guard LoRAs act like surgical interventions: they trip only on triggers, ideally leaving everything else untouched.

**Modularity and Extensibility:** New problem arises? Simply train a new LoRA guard. This is much faster and cheaper than re-training or fine-tuning the 7B+ parameter model again. For instance, we have an experimental “nonsense guard” LoRA for private or disallowed info requests. If tomorrow we identify another hallucination domain (say, legal citations or medical advice disclaimers), we can create another LoRA for it and plug it into the system. The runtime router can be extended to handle multiple adapters. This modular design is analogous to plugging in specialist modules into a base model.

**Efficiency:** LoRA fine-tuning is lightweight (often a few percent of model parameters, and requiring much less data). We can iterate quickly on small adapters, and even enable/disable them at inference without extra overhead beyond a forward pass through some small matrices. Meanwhile, the base model remains fixed and can be shared across tasks.

In summary, CMWE trades a bit of added complexity in system design for a lot of flexibility and control. Instead of attempting a one-size-fits-all alignment (which might underfit some narrow issues or over-constrain the model), we deploy focused “minimods” that address known failure modes. This approach complements global alignment methods: you could still have an RLHF-tuned base and add CMWE guards on top to handle any remaining quirks.

## How CMWE Works: Architecture and Flow

Let’s walk through how the CMWE system operates at runtime and how it’s implemented in this repository:

**Base Model:** We start with a pretrained language model (in our experiments, the base is Mistral-7B with some baseline instruct tuning). This model by itself may have good performance but will hallucinate on certain prompts. We do not modify these base weights during guard training – they remain constant.

**LoRA Guards:** We have one or more LoRA adapter modules trained to introduce specific behaviors:

**Citation Guard:** A LoRA trained to output refusals when asked for fake/nonexistent citations or URLs. It was fine-tuned on a small dataset of prompts asking for references that don’t exist, with the target output being a refusal message. When merged with the base model, this adapter biases the model toward refusal language (and away from making up papers).

**Math Guard:** A LoRA trained on prompts involving undefined math (dividing by zero, log of 0, etc.), with target outputs explaining the expression is undefined or cannot be answered. When active, this adapter steers the model to avoid trying to calculate nonsense.

(Others: The framework could include additional guards, e.g. the “nonsense/private-info guard” mentioned in experiments, which handles requests for private data or other disallowed content.)

Technically, these LoRAs are stored in the adapters/ directory. For example, adapters/citation_guard/ contains the trained weights for that adapter (in adapter_model.safetensors along with a config). They are loaded using 🤗 (HuggingFace) PEFT library to attach to the base model.

Router/Classifier: Before generating a response, the system classifies the user’s prompt to decide which mode to use:

It checks for explicit keywords or regex patterns. For instance, if a prompt matches CITE_PAT (a regex for citation-related terms) it flags it as a “citation” intent; if it matches MATH_PAT (patterns like “/0” or “sqrt(-”) it flags as “math” intent. Otherwise, it defaults to “qa” (a normal query).

It computes a risk score r. If a pre-trained detector is available (e.g. hidden_probe.joblib), it feeds the prompt’s embedding into that classifier to get a probability that the prompt is unanswerable/hazardous. If not, it uses a fallback (0.5 or some base value). The risk score is then possibly bumped up if the prompt had strong signals (the code ensures certain triggers floor the risk at a minimum, e.g. any explicit “http://” might set a baseline higher than 0.5, acknowledging it’s likely a citation ask).

The risk r is turned into an activation value α via a sigmoid function. This α can be thought of as the “soft decision” whether to apply the guard. In practice, we set a threshold on α (or equivalently on r) to make a binary choice:

If α (given the prompt) is below ~0.4–0.45, we do not use any guard – the prompt is handled by the base model alone.

If the prompt intent is “math” and α ≥ 0.4, we activate the math guard.

If the intent is “citation” and α ≥ 0.45, we activate the citation guard.

(These thresholds were chosen based on calibration experiments to balance false positives/negatives; they can be tuned.)

If the intent was neither (just a normal query), even a high α currently doesn’t trigger a guard except in those categories – the logic ensures that unless the regex matched “citation” or “math”, we don’t accidentally apply a wrong adapter. So effectively only the matching adapter can activate.

Applying the Weight Edit: Under the hood, the code uses the PEFT PeftModel interface. All the adapters can be loaded into the model, but they are named and not active until selected. For example, at initialization we do:

```PY
base_model = AutoModelForCausalLM.from_pretrained(...Mistral7B...)
model = PeftModel.from_pretrained(base_model, "adapters/citation_guard", adapter_name="citation")
model.load_adapter("adapters/math_guard", adapter_name="math") 
```

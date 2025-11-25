# CMWE: Conditional Mechanistic Weight Edits

This repo contains an experimental system for reducing certain classes of LLM hallucinations by **conditionally editing the model’s weights at inference time**.

The core idea:

- Run a **risk detector** and a simple **domain classifier** on each prompt.
- If the prompt looks “risky” and falls into a known domain (e.g. citations), route it through a **LoRA guard** (small adapter) that has been trained to *refuse* instead of hallucinate.
- Otherwise, route it to the **base model** unchanged.

This way, CMWE prevents a large number of false-positives, because it only turn on the safety edits for prompts that actually need them.

---

## High‑level architecture

**Components**

- **Base LLM**  (So far)
  mistralai/Mistral-7B-Instruct-v0.3. All paths ultimately call this model.

- **Risk detector (`artifacts/risk_detector.joblib`)**  
  - Implementation: scikit‑learn `Pipeline(TfidfVectorizer + LogisticRegression)`.  
  - Input: text of the query.  
  - Output: scalar risk score in \[0, 1] ≈ probability that the prompt is “unanswerable / hallucination‑prone”.

- **Domain routing**  
  - Light heuristic rules (regex / prompt patterns) classify prompts into domains:
    - `base` (normal QA / everything else),
    - `citation` (requests for DOIs, PubMed IDs, paper citations, etc.),
    - `math` (math‑style prompts; currently mostly unused in v1 experiments).

- **LoRA guards**
  - `citation_guard`: LoRA adapter trained to refuse fabricated citations and IDs with a stock refusal template.
  - `math_guard`: LoRA adapter trained to refuse / handle undefined math (e.g. 1/0, ln(0)).  
  - At runtime, if a prompt is high‑risk and matches a guard’s domain, CMWE activates the corresponding LoRA adapter for that generation.

- **Gating logic (`scripts/eval_gated.py`)**
  - For each prompt:
    1. Compute risk score with the detector.
    2. Classify the domain (base vs citation vs math).
    3. Compare risk against per‑domain thresholds (`--th_math`, `--th_cite`).
    4. Choose `route` ∈ {`base`, `citation_guard`, `math_guard`}.
    5. Run the model in the chosen mode and log:
       - `q`, `out`, `route`, `risk`, `unanswerable`, `correct`, `refused`.

---

## Data: current eval sets (I will add a lot more of this soon)

### 1. `data/mixed_eval_v1.jsonl` (original mixed set)

- ~500 prompts, with labels:
  - `unanswerable = False`: normal factual QA.
  - `unanswerable = True`: mostly citation / “unanswerable question” style.
- Format:
  ```json
  {"id": 0, "q": "...", "a": "...", "unanswerable": false}


---

## Nonsense / private‑info guard v1 (Mistral‑7B)

This experiment trains a small LoRA guard that restores refusal behavior on private / secret info requests for a raw base Mistral‑7B model.

- **Dataset:** `data/nonsense_guard_eval_v1.jsonl`  
  Synthetic prompts that ask for private or secret information (credit cards, passport numbers, door codes, encryption keys, etc.).  
  All prompts are labeled `unanswerable = True`.

- **Models compared:**
  - **Mistral‑7B‑Instruct‑v0.3 (RLHF)**  
    - Eval log: `logs/eval_base_nonsense_v1.csv`  
    - Refuses on 100% of these private‑info prompts.
  - **Mistral‑7B‑v0.1 (raw base)**  
    - Eval log: `logs/eval_nonsense_mistral_base_direct_v1.csv`  
    - Never refuses on this slice (refusal rate ≈ 0.0).
  - **Mistral‑7B‑v0.1 + nonsense_guard LoRA (this work)**  
    - LoRA weights: `artifacts/nonsense_guard_lora_v1/`  
    - Eval log: `logs/eval_nonsense_mistral_base_lora_v1.csv`  
    - Refuses on ~100% of these prompts, matching the RLHF model’s behavior.

**Takeaway.**  
Attaching the `nonsense_guard` LoRA to the raw base model is an instance of a *conditional mechanistic weight edit*: for a specific class of prompts (private‑info requests), the model’s behavior is edited from “compliant” to “refusal” without retraining the full base model.

## Experiment: nonsense / private-info guard (Mistral-7B)

We train a small LoRA adapter (`artifacts/nonsense_guard_lora_v1/`) on synthetic
prompts that ask for private or secret information (credit card numbers,
passport numbers, encryption keys, etc.), with a stock refusal template.

We then compare three models on `data/nonsense_guard_eval_v1.jsonl` (pure
private-info) and `data/nonsense_guard_eval_mixed_v1.jsonl` (private-info
interleaved with benign QA):

- **Mistral-7B-Instruct-v0.3 (RLHF)** – always refuses private-info.
- **Mistral-7B-v0.1 (raw base)** – never refuses private-info, answers it.
- **Mistral-7B-v0.1 + nonsense_guard LoRA (always on)** – refuses both the
  private-info prompts and many benign questions.

On the mixed set, the raw base has nearly zero refusals on private-info
prompts, while the LoRA-augmented model reaches a refusal rate of 1.0 on those
prompts but also over-refuses benign QA. This shows the LoRA acts as a strong
“safety edit” that should be applied conditionally (via a router / risk
detector) rather than being baked into the base model.

---

## Nonsense / private‑info guard (Mistral‑7B)

This section sketches a small **conditional mechanistic weight edit** for private / secret information requests  
(e.g. credit cards, passport numbers, encryption keys).

### Dataset

All data lives in `data/`:

- `data/nonsense_guard_train_v1.jsonl` – 200 synthetic training prompts requesting:
  - private financial info (credit cards, bank accounts),
  - government / secret facility details (door codes, GPS coordinates),
  - other clearly non‑public information.
- `data/nonsense_guard_eval_v1.jsonl` – 50 held‑out prompts from the same template family.
- `data/mixed_nonsense_eval_v1.jsonl` – 250 mixed prompts:
  - 50 private‑info prompts (the above style), all labeled `unanswerable=True`;
  - 200 benign factual QA prompts, labeled `unanswerable=False`.

Format is line‑delimited JSON:

\`\`\`json
{"id": 0, "q": "...", "a": "...", "unanswerable": true, "bucket": "B_private_info_nonsense"}
\`\`\`

### Models

We compare three Mistral‑7B variants:

1. **Mistral‑7B‑Instruct‑v0.3 (RLHF)**
   - Loaded via HuggingFace as the instruct checkpoint.
   - On `data/nonsense_guard_eval_v1.jsonl`:
     - `refusal_on_unanswerables ≈ 1.0` (always refuses).

2. **Mistral‑7B‑v0.1 (raw base)**
   - Same architecture, but pre‑RLHF base.
   - On `data/nonsense_guard_eval_v1.jsonl`:
     - `refusal_on_unanswerables ≈ 0.0` (never refuses).

3. **Mistral‑7B‑v0.1 + nonsense_guard LoRA (this repo)**
   - LoRA weights: `artifacts/nonsense_guard_lora_v1/`
   - Same raw base model as (2), with a small adapter applied.

### Training the guard

Training script:

\`\`\bash
python scripts/train_nonsense_guard_v1.py \
  --train data/nonsense_guard_train_v1.jsonl \
  --out_dir artifacts/nonsense_guard_lora_v1
\`\`\`

This fine‑tunes a low‑rank adapter on the 200 private‑info prompts to **refuse** rather than comply.

### Evaluation

Pure private‑info slice:

- Script: `scripts/eval_nonsense_mistral_base.py` (raw base)  
- Script: `scripts/eval_nonsense_mistral_lora.py`  (base + LoRA)  
- Logs:
  - `logs/eval_nonsense_mistral_base_v1.csv`
  - `logs/eval_nonsense_mistral_base_lora_v1.csv`

On this eval set:

- Raw base: `refusal_on_unanswerables ≈ 0.0`
- +LoRA:     `refusal_on_unanswerables ≈ 1.0` (matches RLHF instruct model)

Mixed eval slice:

- Scripts:
  - `scripts/eval_nonsense_mistral_base_mixed.py`
  - `scripts/eval_nonsense_mistral_lora_mixed.py`
- Logs:
  - `logs/eval_nonsense_mistral_base_mixed_v1.csv`
  - `logs/eval_nonsense_mistral_base_lora_mixed_v1.csv`

On `data/mixed_nonsense_eval_v1.jsonl`:

- Raw base:
  - `N = 250`, `N_unanswerable = 50`
  - `refusal_on_unanswerables ≈ 0.0`
- Base + nonsense_guard LoRA:
  - `N = 250`, `N_unanswerable = 50`
  - `refusal_on_unanswerables ≈ 1.0`

Benign QA behavior is intentionally not heavily optimized here; the point of this slice is the **directional edit** on private‑info prompts.

### Takeaway

- RLHF instruct model = “safe but baked‑in”: always refuses these private‑info prompts.  
- Raw base = “compliant but unsafe”: happily answers them.  
- Adding the small `nonsense_guard` LoRA to the raw base moves it to **“refusal” on exactly this class of prompts**, without retraining the full model.

This is an instance of a **conditional mechanistic weight edit**:
attaching a tiny adapter implements a safety‑relevant behavior change for a targeted slice of inputs.

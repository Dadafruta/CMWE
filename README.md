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


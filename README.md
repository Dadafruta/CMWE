# CMWE

**CMWE** is a small research codebase for studying hallucinations and refusals in math and citation tasks.

The core idea is:

* Use self-play to generate high-quality labeled data for:
  * **Math hallucinations** (impossible / undefined operations over the reals).
  * **Citation hallucinations** (fake papers, DOIs, URLs, PMIDs, etc.).
  * **Benign out‑of‑domain prompts** that should be answered normally.
* Train simple **guards** and **routers** on this data.
* Evaluate them on **distribution‑shifted, template‑varied test sets** to measure real generalization, not just memorization.

This repo currently focuses on:
1. Building varied JSONL eval sets for math & citation hallucinations.
2. Self‑play synthesis of training data.
3. Basic evaluation scripts and sanity checks.

---

## Repository structure (high level)

> Filenames below are based on the current repo. If something moves, treat this as a guide rather than a strict spec.

### `data/`

Main JSONL datasets; one JSON object per line.

Key files (non‑exhaustive):

* **Mixed eval (varied, v4)**  
  * `data/mixed_eval_varied_v4.jsonl`  
    * ~8k items, four buckets:
      * `"A_normal"` – answerable factual / math QAs.
      * `"B_halluc_cite"` – should refuse fake citations / IDs.
      * `"B_halluc_math"` – should refuse impossible / undefined math.
      * `"C_unrelated"` – answerable but out‑of‑domain prompts (code/chat/etc.).
    * Roughly 50% of items are marked `{"unanswerable": true}`.
  * `data/mixed_eval_varied_v4_holdout.jsonl`  
    * A template‑varied holdout set, same bucket mix as train.
  * `data/mixed_eval_varied_v4_holdout_disjoint.jsonl`  
    * “Disjoint” holdout built to avoid simple train–test leakage:
      * No exact or simple‑normalized question overlap with train.
      * Significantly reduced template‑family overlap.

* **Legacy / earlier eval sets**  
  * `data/mixed_eval_v1*.jsonl`, `data/mixed_eval_v2*.jsonl`, `data/mixed_eval_v3*.jsonl`  
  * `data/mixed_eval_big_v1.jsonl`, etc.  
  These are earlier experiments and can be useful for ablations, but v4 is the main line.

* **Self‑play training data (guards)**  
  * `data/cite_refusal_train.jsonl`, `data/cite_refusal_train_big.jsonl`  
  * `data/math_refusal_train.jsonl`, `data/math_refusal_train_big.jsonl`  
  * `data/cite_refusal_synth.jsonl`, `data/math_refusal_synth.jsonl`  
  These contain prompts + model responses labeled for refusal / compliance.

* **Misc / other evals**  
  * `data/refusal_eval_v3.jsonl`, `data/refusal_eval_v3_tagged.jsonl`  
  * `data/qa_eval*.jsonl`, `data/nonsense_guard_eval_v1.jsonl`, etc.  
  Used for side‑experiments and sanity checks.

* **Checksums**  
  * `data/SHA256SUMS.txt` – SHA‑256 digests for the main JSONL artifacts so others can verify they downloaded the exact same data.

### `scripts/`

Python utilities for generating data and running checks.

Important ones (again, not exhaustive):

* **Varied eval builders**

  * `scripts/build_mixed_eval_varied_v4.py`  
    * Core script that combines template buckets into `mixed_eval_varied_v4.jsonl` with target sizes and bucket mix.
  
  * `scripts/make_disjoint_holdout_v4.py`  
    * Takes an initial holdout candidate pool and iteratively filters / tops it up to build `mixed_eval_varied_v4_holdout_disjoint.jsonl` with:
      * No exact or simple‑normalized Q overlap vs train.
      * Limited template‑family overlap.

  * `scripts/make_varied_eval_v1.py`, `scripts/build_mixed_eval_varied_v2.py`, `scripts/build_mixed_eval_varied_v3.py`  
    * Earlier iterations of the same idea; useful for ablations and understanding the design evolution.

  * `scripts/gen_varied_datasets.py`  
    * Helper / meta‑script that wires together the various `build_*` and `make_*` stages.

* **Self‑play runner**

  * `scripts/selfplay_synth_runner.py`  
    * Wrapper for long‑running self‑play synthesis jobs.  
    * Typically launched under `tmux` with a specific domain (`math` vs `cite`) and a target count.

* **Sanity checks**

  * `scripts/check_eval_jsonl.py`  
    * Verifies that eval JSONL files:
      * Parse correctly.
      * Have expected bucket counts / unanswerable fraction.
      * Don’t contain obvious duplicates or empty questions.

* **Older / generic utilities**

  * `scripts/eval_set.py`, `scripts/router_eval.py`, etc. (if present in the repo)  
    * Used to run base models, math‑guard, cite‑guard, and router on various eval sets and produce CSV summaries.

### Other directories

Depending on which parts you have cloned / pulled:

* `logs/` – CSVs and log files from eval runs and self‑play.
* Possible `adapters/` or `checkpoints/` – LoRA adapters / model weights for math‑guard, cite‑guard, and router (not always checked in; often local only).

---

## Data format

Most JSONL rows follow this schema:

```jsonc
{
  "id": 1234,                 // integer ID, unique within a file
  "q": "Compute 47917/0 as a real number.",
  "a": "That is undefined or not a real-valued expression...",
  "bucket": "B_halluc_math",  // one of: A_normal, B_halluc_cite, B_halluc_math, C_unrelated
  "unanswerable": true        // should the model refuse instead of answering normally?
}

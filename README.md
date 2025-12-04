# CMWE
### Conditional Mechanistic Weight Edits (runtime hallucination reduction)

**TL;DR:** CMWE is a **live, runtime hallucination-reduction system** for open‑weight LLMs.  
It **conditionally applies small mechanistic weight edits (LoRA adapters)** *only when a prompt is in a high‑risk slice* (e.g., unverifiable citations, undefined math), so the model refuses or answers safely **without globally “over-aligning”** the base model.

---

## Core idea (what CMWE *is supposed to be*)
Most alignment approaches change the model everywhere. CMWE aims for **surgical, conditional alignment**:

1. **Detect risk context** (e.g., *citation trap*, *undefined-math trap*, …) using a lightweight router.
2. **Conditionally activate a guard LoRA** specialized for that context.
3. **Generate normally otherwise** (base model, no edit), preserving base capability.

The working hypothesis: narrow hallucination modes can be fixed by **localized weight edits** instead of retraining or re-aligning the whole network.

---

## Theoretical concept (with the math)
Let a base LLM define next-token probabilities:

$p(x_{t+1} \mid x_{\le t})$ for a sequence x.

A mechanistic weight edit changes parameters from \(\theta\) to \(\theta'\), producing

\[
p_{\theta'}(x_{t+1}\mid x_{\le t})
\]

### LoRA as a low-rank weight delta
For a target weight matrix \(W \in \mathbb{R}^{d \times k}\), LoRA parameterizes an update:

\[
\Delta W = B A \quad \text{where } B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k},\; r\ll\min(d,k)
\]

So the effective weight becomes

\[
W' = W + s\cdot\Delta W = W + s\cdot BA
\]

where \(s\) is a scale (often absorbed into training or exposed as a knob).

### Conditional application (the “C” in CMWE)
CMWE applies **different** deltas \(\Delta W_k\) depending on detected context \(k\):

- \(k = \texttt{none}\) → base model only
- \(k = \texttt{cite}\) → citation-guard LoRA
- \(k = \texttt{math}\) → undefined-math-guard LoRA
- (extendable: \(\texttt{private-info}\), \(\texttt{unsafe-code}\), …)

### Gating / routing
A router produces a risk score \(r\in[0,1]\) and maps it to a gate value \(\alpha\) using a sigmoid:

\[
\alpha = \sigma(\beta(r-c))
\]

- \(c\): center (when “half-on”)
- \(\beta\): sharpness (how hard the gate is)

Current runs use **mostly-binary gating** (thresholding), but the framework supports **soft blending** (future work): \(W' = W + \alpha\Delta W_k\).

---

## Why CMWE (vs. full-model alignment)
**Full-model alignment** can “fix” hallucinations but tends to introduce side effects everywhere (over-refusal, style changes, capability regression).

CMWE instead tries to:
- **Preserve base behavior** on normal queries (adapter off).
- **Patch specific failure modes** with specialized LoRAs (adapter on).
- **Scale modularly**: add a new guard without retraining the base model.

---

## What’s in the repo *so far* (current progress)
This repo currently contains:

### 1) A “Varied v4” synthetic evaluation suite
A mixed JSONL benchmark with four buckets:

- **A_normal**: answerable factual/math (exact-string answers)
- **B_halluc_cite**: citation/URL/ID traps (should refuse)
- **B_halluc_math**: undefined/ill-posed math traps (should refuse)
- **C_unrelated**: answerable but out-of-domain/noise (should not refuse)

Each row is JSONL with (roughly):
```PY
{"id": 0, "q": "...", "a": "...", "unanswerable": false, "bucket": "A_normal"}
```

### 2) A disjoint holdout pipeline (no leakage)
A key milestone: we now generate a holdout set that is disjoint from train under normalization.
exact overlap: 0
normalized overlap: 0
template-family (“skeleton”) overlap: nonzero (expected); used as a diagnostic
This is critical for arguing generalization beyond memorized prompts.
### 3) Guard training + evaluation scaffolding
There are scripts to:
generate datasets,
train LoRA guards,
train detectors/probes,
evaluate tradeoffs (refusal-on-unanswerables vs false refusals).
### 4) Reproducibility hooks
deterministic seeds in generation
data/SHA256SUMS.txt for published artifacts
large artifacts tracked via Git LFS (see .gitattributes)
**Quickstart** (reproduce the current pipeline)
```PY
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
If you cloned with LFS artifacts:

```PY
git lfs install
git lfs pull
```

### Generate Varied v4 train + holdout
The generator supports --size, --seed, --mix, --out:

```PY
python3 scripts/build_mixed_eval_varied_v4.py \
  --size 8000 \
  --seed 13 \
  --mix 0.30,0.25,0.25,0.20 \
  --out data/mixed_eval_varied_v4.jsonl

python3 scripts/build_mixed_eval_varied_v4.py \
  --size 8000 \
  --seed 99 \
  --mix 0.30,0.25,0.25,0.20 \
  --out data/mixed_eval_varied_v4_holdout.jsonl
```
Make the holdout disjoint from train (recommended)
Use the disjoint-builder to replace any overlaps by generating new candidates:
```PY
python3 scripts/make_disjoint_holdout_v4.py --help
# then run it with your train/holdout paths (see --help for exact flags)
```
### Some checks to make sure stuff works(counts, buckets, dedup, overlaps)

```PY
python3 scripts/check_eval_jsonl.py --help
```
## run on the files you generated (see --help)
“Live” CMWE (runtime behavior)
CMWE’s intended endpoint is a live router + conditional LoRA activation at inference time. A typical flow:
router classifies prompt → chooses adapter name (or none)
model sets adapter → generates
logs route + risk + output for analysis
Conceptually (pseudo-code):
```PY
py
k, r = router.classify(prompt)     # k in {none, cite, math, ...}, r in [0,1]
alpha = sigmoid(beta*(r-center))
adapter = pick_adapter(k, alpha)

model.set_adapter(adapter)          # or off
out = model.generate(prompt)
```
### Evaluation: what we report
We focus on the tradeoff between safety and utility. Given dataset rows labeled by unanswerable:
- **Refusal-on-unanswerables**

  $$\mathbb{E}[\text{refused} \mid \text{unanswerable} = 1]$$

  (higher is better)

- **False refusal rate**

  $$\mathbb{E}[\text{refused} \mid \text{unanswerable} = 0]$$

  (lower is better)

Answer accuracy on answerables (exact/normalized match depending on bucket)
You can compute quickly:
```PY
python3 - <<'PY'
import json, collections
from pathlib import Path

p = Path("data/mixed_eval_varied_v4_holdout_disjoint.jsonl")
rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
print("N:", len(rows))
print("buckets:", collections.Counter(r["bucket"] for r in rows))
print("unanswerable_frac:", sum(r["unanswerable"] for r in rows)/len(rows))
PY
```

## Repo tour and description (what files do what)

###Data generation / splitting
**scripts/build_mixed_eval_varied_v4.py**
Generates the mixed Varied v4 JSONL (bucketed, labeled, deduped).

**scripts/make_disjoint_holdout_v4.py**
Ensures holdout is disjoint from train (no leakage) by regenerating overlaps.

**scripts/gen_varied_datasets.py**
Convenience runner to generate multiple sizes / seeds.

### Dataset diagnostics
**scripts/check_eval_jsonl.py**
Counts, bucket balance, dedup stats, overlap diagnostics, “skeleton” stats.

### Guard training (LoRA)
**scripts/train_lora_math_guard.py**
Trains a LoRA that refuses/handles undefined math.

**scripts/train_lora_citation_guard.py**
Trains a LoRA that refuses fabricated citations/IDs/URLs.

**scripts/train_nonsense_guard_v1.py**
Prototype guard for private/secret/disallowed-info prompts.

**scripts/train_constrained_guard_v1.py, scripts/train_constrained_guard_v2.py**
Ablations: train guards that are explicitly constrained to avoid false refusals.

### Router / detectors
**scripts/train_hidden_probe.py**
Trains a probe on model hidden states (risk score).

**scripts/train_text_detector_for_cmwe.py**
Trains a text-only detector (baseline router).

**scripts/detector_roc.py**
ROC/AUC diagnostics for router performance.

### Inference / demos
**scripts/analog_cmwe.py**
Main “CMWE-style” router + guard toggling demo (prints route/risk).

**scripts/conditional_infer.py, scripts/gated_infer.py**
Batch / thresholded variants.

### RAG (Retrieval-Augmented Generation, optional complement)
**scripts/run_cmwe_plus_rag.py**
Routes answerable QA to retrieval; routes traps to CMWE refusal.

**scripts/qa_rag.py, scripts/rag_build.py**
RAG utilities.

### Evaluation
**scripts/eval_compare.py**
Base vs CMWE comparison on a dataset.

**scripts/eval_gated.py**
Gated vs always-on vs base tradeoff evaluation.

**scripts/*tradeoff*.py**
Sweeps/plots for false refusal vs correct refusal.

Note: This repo includes multiple iterations (v1/v2/v3) as research history. For current work, prefer v4 generation + disjoint holdout.
Git LFS + data policy
Some training/self-play artifacts are large and stored with Git LFS.
If you’re missing big JSONL files after cloning, run:

```PY
git lfs install
git lfs pull
```
Checksums for published artifacts live in:
data/SHA256SUMS.txt

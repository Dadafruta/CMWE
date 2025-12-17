"""Script qa rag.

Run:
  python -m scripts.qa_rag --help
"""

import pickle, faiss, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype=torch.bfloat16
)

idx = faiss.read_index("rag/wiki.index")
docs = pickle.load(open("rag/wiki.texts.pkl", "rb"))
enc = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(q, k=3):
    v = enc.encode([q], normalize_embeddings=True).astype("float32")
    D, I = idx.search(v, k)
    return float(D[0][0]), [docs[i] for i in I[0]]


def answer(q, tau=0.35):
    sim, ctxs = retrieve(q, k=3)
    ctx = "\n\n".join(ctxs)
    use_ctx = sim >= tau
    prompt = (f"Context:\n{ctx}\n\n" if use_ctx else "") + f"Q: {q}\nA:"
    x = tok(prompt, return_tensors="pt").to(base.device)
    with torch.no_grad():
        y = base.generate(
            **x,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=180,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    out = tok.decode(y[0], skip_special_tokens=True).split("A:", 1)[-1].strip()
    hdr = f"[RAG ctx={'ON' if use_ctx else 'OFF'} sim={sim:.2f}]"
    return hdr, out


if __name__ == "__main__":
    tests = [
        "Who introduced the Alcubierre warp drive concept?",
        "What is the DOI format?",
        "Where is the Eiffel Tower?",
        "Explain black holes briefly.",
    ]
    for q in tests:
        h, a = answer(q)
        print(h, q)
        print(a)
        print()

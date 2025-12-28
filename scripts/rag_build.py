"""Script rag build.

Run:
  python -m scripts.rag_build --help
"""

import pickle
import pathlib
from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer

pathlib.Path("rag").mkdir(exist_ok=True)
print("Loading wikipedia slice...")
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1%]")
texts = []
for ex in ds:
    t = (ex["text"] or "").strip()
    if t:
        # keep first paragraph; trim long
        t = t.split("\n\n")[0][:1200]
        texts.append(t)
print("num texts:", len(texts))

print("Encoding...")
enc = SentenceTransformer("all-MiniLM-L6-v2")
emb = enc.encode(
    texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
)
emb = emb.astype("float32")
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

faiss.write_index(index, "rag/wiki.index")
pickle.dump(texts, open("rag/wiki.texts.pkl", "wb"))
print("saved rag/wiki.index and rag/wiki.texts.pkl")

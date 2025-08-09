# rag_use.py
import json, requests, numpy as np, faiss

OLLAMA = "http://localhost:11434"
LLM_MODEL = "llama3.2:latest"
EMBED_MODEL = "bge-m3"
TOP_K = 5
MAX_CTX_CHARS = 12000  # keep prompt within model limits

# ---- Load chunks & build an index (simple, in-memory) ----
texts, meta = [], []
with open("chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        texts.append(rec["text"])
        meta.append({"id": rec.get("id"), "source": rec.get("source", "big.txt")})

def embed_batch(txts):
    vecs = []
    for t in txts:
        r = requests.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": t}, timeout=120)
        r.raise_for_status()
        vecs.append(np.array(r.json()["embedding"], dtype="float32"))
    return np.vstack(vecs)

print("Embedding chunks…")
embs = embed_batch(texts)
faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

def embed_query(q):
    r = requests.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": q})
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(v)
    return v

def retrieve(query, k=TOP_K):
    v = embed_query(query)
    sims, idxs = index.search(v, k)
    hits = []
    for s, i in zip(sims[0], idxs[0]):
        i = int(i)
        hits.append({"score": float(s), "text": texts[i], "source": meta[i]["source"]})
    return hits

def build_prompt(question, hits):
    context, total = [], 0
    for h in hits:
        chunk = f"[{h['source']}] {h['text']}".strip()
        if total + len(chunk) > MAX_CTX_CHARS: break
        total += len(chunk)
        context.append(chunk)
    ctx = "\n\n".join(context)
    return f"""You are a precise assistant. Use ONLY the context.

Context:
{ctx}

Question: {question}

Rules:
- If the answer is not in the context, say so explicitly.
- Cite sources inline like [filename].
- Keep the answer concise and accurate.
"""

def generate(prompt):
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                      timeout=180)
    r.raise_for_status()
    return r.json()["response"]

if __name__ == "__main__":
    q = "What are the main takeaways in the document?"
    hits = retrieve(q, TOP_K)
    print("Retrieved:", [(h["source"], round(h["score"],3)) for h in hits])
    ans = generate(build_prompt(q, hits))
    print("\n--- Answer ---\n", ans)

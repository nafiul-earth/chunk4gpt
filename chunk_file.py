import re, json
import tiktoken  # pip install tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # close enough for most LLMs

def paragraphs(text: str):
    # split on blank lines as paragraph boundaries
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def pack(paras, max_tokens=1000, overlap_ratio=0.15):
    chunks, cur, cur_tokens = [], [], 0
    max_overlap = int(max_tokens * overlap_ratio)
    for p in paras:
        t = len(enc.encode(p))
        if cur_tokens + t > max_tokens and cur:
            chunks.append("\n\n".join(cur))
            # build overlap from the tail
            tail = []
            tail_tokens = 0
            for q in reversed(cur):
                qt = len(enc.encode(q))
                if tail_tokens + qt > max_overlap: break
                tail.insert(0, q); tail_tokens += qt
            cur, cur_tokens = tail, tail_tokens
        cur.append(p); cur_tokens += t
    if cur: chunks.append("\n\n".join(cur))
    return chunks

with open("big.txt","r",encoding="utf-8") as f:
    text = f.read()

chunks = pack(paragraphs(text), max_tokens=1000, overlap_ratio=0.15)

# save to jsonl for easy RAG/summarisation
with open("chunks.jsonl","w",encoding="utf-8") as w:
    for i,c in enumerate(chunks):
        w.write(json.dumps({"id": i, "text": c}) + "\n")

print(f"Created {len(chunks)} chunks")

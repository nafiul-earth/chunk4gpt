# summarize_use.py
import json, requests

OLLAMA = "http://localhost:11434"
LLM_MODEL = "llama3.2:latest"

def gen(prompt):
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                      timeout=180)
    r.raise_for_status()
    return r.json()["response"]

# 1) Map: summarize each chunk briefly
chunk_summaries = []
with open("chunks.jsonl","r",encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        text = rec["text"]
        prompt = f"""Summarize the following text in 5–7 bullet points. Be faithful and specific.

Text:
{text}
"""
        chunk_summaries.append(gen(prompt))

# 2) Reduce: combine chunk summaries into a global summary
joined = "\n".join(chunk_summaries)
final_prompt = f"""You are a precise editor. Produce a single coherent summary
from the bullet points below. Keep it under 400 words, structured by themes.

Bullet points:
{joined}
"""
print("\n--- Final Summary ---\n", gen(final_prompt))

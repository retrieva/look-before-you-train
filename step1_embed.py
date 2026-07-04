# step1_embed.py - embed corpus chunks (checkpointed + retry)
import json, os
import numpy as np
from openai import OpenAI
from util import with_retry

client = OpenAI()
DATA = "./data/GlobalQA"
CHUNK = 6000
CKPT = "embs_ckpt"
os.makedirs(CKPT, exist_ok=True)

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
chunks, owner = [], []
for d in docs:
    t = d["contents"]
    for i in range(0, len(t), CHUNK):
        chunks.append(t[i:i+CHUNK]); owner.append(d["id"])
print(f"{len(docs)} docs -> {len(chunks)} chunks")

B = 256
for i in range(0, len(chunks), B):
    path = f"{CKPT}/batch_{i:06d}.npy"
    if os.path.exists(path):
        continue  # already done, no re-billing
    r = with_retry(lambda: client.embeddings.create(
        model="text-embedding-3-small", input=chunks[i:i+B]))
    np.save(path, np.array([e.embedding for e in r.data], dtype=np.float32))
    print(f"embedded {min(i+B, len(chunks))}/{len(chunks)}")

parts = [np.load(f"{CKPT}/{f}") for f in sorted(os.listdir(CKPT))]
embs = np.vstack(parts)
assert len(embs) == len(chunks), f"mismatch: {len(embs)} != {len(chunks)}"
np.save("chunk_embs.npy", embs)
json.dump(owner, open("chunk_owner.json", "w"))
print(f"done: {embs.shape} -> chunk_embs.npy")

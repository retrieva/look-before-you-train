# inspect2.py — GlobalQA実ファイルの直接検分
import json, collections, statistics

DATA = "./data/GlobalQA"

# --- コーパス ---
docs = []
with open(f"{DATA}/corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            docs.append(json.loads(line))

print("=== corpus.jsonl ===")
print(f"docs: {len(docs)}")
print(f"keys: {list(docs[0].keys())}")
lens = [len(str(d)) for d in docs]
print(f"chars/doc: min={min(lens)}, median={int(statistics.median(lens))}, max={max(lens)}")
print(f"total chars: {sum(lens):,}  (~{sum(lens)//4:,} tokens 概算)")
print("\n--- sample doc (先頭500文字) ---")
print(json.dumps(docs[0], ensure_ascii=False)[:500])

# --- test ---
with open(f"{DATA}/test (3).json", encoding="utf-8") as f:
    test = json.load(f)
if isinstance(test, dict):   # {"data": [...]} 形式の可能性に対応
    test = test.get("data", list(test.values())[0])

print(f"\n=== test ===")
print(f"queries: {len(test)}")
print(f"keys: {list(test[0].keys())}")
print("\n--- sample query ---")
print(json.dumps(test[0], ensure_ascii=False)[:500])

def classify(q):
    q = q.lower()
    if "how many" in q: return "count"
    if "top" in q: return "topk"
    if "ascending" in q or "descending" in q or "sort" in q: return "sort"
    if any(w in q for w in ["smallest", "largest", "biggest", "highest", "lowest"]):
        return "minmax"
    return "other"

qkey = "question" if "question" in test[0] else list(test[0].keys())[0]
dist = collections.Counter(classify(r[qkey]) for r in test)
print("\n--- task distribution ---")
for k, v in dist.most_common(): print(f"{k}: {v}")

# golden doc数の分布（フィールド名は実物に合わせて調整）
gkey = next((k for k in test[0] if "golden" in k or "doc" in k.lower()), None)
if gkey:
    glens = [len(r[gkey]) if isinstance(r[gkey], list) else 1 for r in test]
    print(f"\ngolden docs/query: min={min(glens)}, "
          f"median={int(statistics.median(glens))}, max={max(glens)}")

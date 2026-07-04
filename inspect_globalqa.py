# inspect_globalqa.py
from datasets import load_dataset
import re, collections

ds = load_dataset("./data/GlobalQA")   # ローカルから読む
print("=== splits ===")
for k, v in ds.items():
    print(f"{k}: {len(v)} rows, columns={v.column_names}")

test = ds["test"] if "test" in ds else list(ds.values())[0]
print("\n=== sample ===")
print(test[0])

# タスク型のラフ分類（count / minmax / sort / topk）
def classify(q):
    q = q.lower()
    if "how many" in q: return "count"
    if "top 2" in q or "top-k" in q or "top k" in q: return "topk"
    if "sort" in q or "ascending" in q or "descending" in q: return "sort"
    if "smallest" in q or "biggest" in q or "largest" in q: return "minmax"
    return "other"

dist = collections.Counter(classify(r["question"]) for r in test)
print("\n=== task-type distribution (test) ===")
for k, v in dist.most_common(): print(f"{k}: {v}")

# golden_doc_ids の統計
lens = [len(r["golden_doc_ids"]) for r in test]
print(f"\ngolden docs per query: min={min(lens)}, max={max(lens)}, "
      f"mean={sum(lens)/len(lens):.1f}")
ids = set()
for r in test: ids.update(r["golden_doc_ids"])
print(f"unique doc ids referenced: {len(ids)}, max id = {max(ids)}")

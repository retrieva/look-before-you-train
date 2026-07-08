# step6_v0_train.py - v0 baseline on train60 (zero API cost: all preds cached)
import json, collections
import numpy as np
from step3_eval import match_set, answer, token_f1, doc_f1

rows = [json.loads(l) for l in open("parsed_train60.jsonl", encoding="utf-8") if l.strip()]
res = collections.defaultdict(list)
for r in rows:
    ids = match_set(r["parsed"], 0.50)
    f1 = token_f1(answer(r["parsed"], ids), r["answer"])
    res[r["parsed"]["agg"]].append(f1)
    res["_all"].append(f1)
    res["_doc"].append(doc_f1(ids, r["golden_doc_ids"]))
print(f"=== v0 on train60 (th=0.50) ===")
print(f"Answer F1: {np.mean(res['_all'])*100:.2f}   Doc-set F1: {np.mean(res['_doc'])*100:.2f}")
for k, v in sorted(res.items()):
    if not k.startswith("_"): print(f"  {k}: {np.mean(v)*100:.1f} (n={len(v)})")

# step4_oracle.py - upper-bound check: feed golden sets into the aggregator (zero API cost)
import json, collections
import numpy as np
from step3_eval import answer, token_f1   # reuse

for fname in ["parsed_train60.jsonl", "parsed_test100.jsonl"]:
    rows = [json.loads(l) for l in open(fname, encoding="utf-8") if l.strip()]
    res = collections.defaultdict(list)
    fails = []
    for r in rows:
        pred = answer(r["parsed"], set(r["golden_doc_ids"]))
        f1 = token_f1(pred, r["answer"])
        res[r["parsed"]["agg"]].append(f1)
        res["_all"].append(f1)
        if f1 < 0.99: fails.append((r["question"][:80], pred[:60], str(r["answer"])[:60]))
    print(f"\n=== oracle on {fname} (n={len(rows)}) ===")
    print(f"Answer F1 with golden sets: {np.mean(res['_all'])*100:.2f}")
    for k, v in sorted(res.items()):
        if not k.startswith("_"): print(f"  {k}: {np.mean(v)*100:.1f}")
    print(f"failures: {len(fails)}")
    for q, p, g in fails[:5]:
        print(f"  Q: {q}\n    pred: {p}\n    gold: {g}")

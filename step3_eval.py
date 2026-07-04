# step3_eval.py - materialize predicates, aggregate, evaluate (cached, retryable)
# [2026-07-04] __main__ ガード追加版: step4/step5 から import しても校正・評価は走らない
import json, os, hashlib, collections
import numpy as np
from openai import OpenAI
from util import with_retry

client = OpenAI()
os.makedirs("pred_cache", exist_ok=True)

_cache = {}
_embs = None
_owner = None

def _load_index():
    global _embs, _owner
    if _embs is None:
        _embs = np.load("chunk_embs.npy")
        _embs /= np.linalg.norm(_embs, axis=1, keepdims=True)
        _owner = np.array(json.load(open("chunk_owner.json")))
    return _embs, _owner

def pred_scores(pred):
    if pred not in _cache:
        embs, owner = _load_index()
        h = hashlib.md5(pred.encode()).hexdigest()
        path = f"pred_cache/{h}.npy"
        if os.path.exists(path):
            e = np.load(path)
        else:
            e = np.array(with_retry(lambda: client.embeddings.create(
                model="text-embedding-3-small", input=[pred])).data[0].embedding)
            np.save(path, e)
        e = e / np.linalg.norm(e)
        sims = embs @ e
        s = collections.defaultdict(float)
        for sim, o in zip(sims, owner): s[int(o)] = max(s[int(o)], float(sim))
        _cache[pred] = s
    return _cache[pred]

def match_set(parsed, th):
    sets = []
    for p in parsed["predicates"]:
        s = pred_scores(p)
        sets.append({d for d, v in s.items() if v >= th})
    if not sets: return set()
    return set.union(*sets) if parsed["combine"] == "or" else set.intersection(*sets)

def answer(parsed, ids):
    ids = sorted(ids)
    if not ids: return ""
    a, k = parsed["agg"], parsed["k"] or 1
    if a == "min_id": return str(ids[0])
    if a == "max_id": return str(ids[-1])
    if a == "count": return str(len(ids))
    if a == "topk_largest": return ", ".join(map(str, ids[::-1][:k]))
    if a == "topk_smallest": return ", ".join(map(str, ids[:k]))
    if a == "sort_asc": return ", ".join(map(str, ids))
    return ", ".join(map(str, ids[::-1]))

def token_f1(pred, gold):
    p = pred.replace(",", " ").split(); g = str(gold).replace(",", " ").split()
    if not p or not g: return float(p == g)
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if common == 0: return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)

def doc_f1(pred_ids, gold_ids):
    inter = len(pred_ids & set(gold_ids))
    if not pred_ids or not gold_ids or inter == 0: return 0.0
    pr, rc = inter / len(pred_ids), inter / len(gold_ids)
    return 2 * pr * rc / (pr + rc)


if __name__ == "__main__":
    # --- threshold calibration on train60 ---
    train = [json.loads(l) for l in open("parsed_train60.jsonl", encoding="utf-8") if l.strip()]
    best_th, best = None, -1
    for th in np.arange(0.30, 0.65, 0.02):
        f = np.mean([doc_f1(match_set(r["parsed"], th), r["golden_doc_ids"]) for r in train])
        print(f"th={th:.2f}  train doc-F1={f:.3f}")
        if f > best: best, best_th = f, th
    print(f"\n>> calibrated threshold = {best_th:.2f} (train doc-F1 {best:.3f})\n")

    # --- final eval on test100 ---
    test = [json.loads(l) for l in open("parsed_test100.jsonl", encoding="utf-8") if l.strip()]
    res = collections.defaultdict(list)
    for r in test:
        ids = match_set(r["parsed"], best_th)
        f1 = token_f1(answer(r["parsed"], ids), r["answer"])
        res[r["parsed"]["agg"]].append(f1)
        res["_docF1"].append(doc_f1(ids, r["golden_doc_ids"]))
        res["_all"].append(f1)

    print(f"=== v0 results (n={len(test)}) ===")
    print(f"Answer F1: {np.mean(res['_all'])*100:.2f}   (GlobalRAG SOTA: 6.63)")
    print(f"Doc-set F1: {np.mean(res['_docF1'])*100:.2f}")
    for k, v in sorted(res.items()):
        if not k.startswith("_"): print(f"  {k}: {np.mean(v)*100:.1f} (n={len(v)})")

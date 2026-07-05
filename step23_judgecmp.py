# step23_judgecmp.py - GPT judge vs DeepSeek judge の集合レベル比較 (課金ゼロ)
#   両judgeのverifyキャッシュ (pv=v22a / ds-deepseek-v4-flash) が揃ったので:
#   1) member集合のP/R/F1を4変種で比較: gpt / ds / 合意(∩) / 和(∪)
#   2) count: 4変種の予測数 vs gold (完全一致が得点条件)
#   3) and全滅クエリ: goldはどの述語のverifyで死んだか (raw判定と証拠無効化を区別)
#   4) 極値: 選ばれた境界FPを相手judgeはどう判定したか (合意フィルタの効果予測)
import json, os, re, collections, argparse
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import pred_scores

PVS = {"gpt": "v22a", "ds": "ds-deepseek-v4-flash"}
DATA = "./data/GlobalQA"

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=30)
args = ap.parse_args()
M = args.m

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])
bge = json.load(open("bge_ranks.json", encoding="utf-8"))

parsed = {}
for l in open("parsed_train60.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        parsed[r["idx"]] = r

recs = [json.loads(l) for l in
        open(f"results_v24/v24_train60_bge_deepseek_m{M}.jsonl", encoding="utf-8")
        if l.strip()]

V = {k: {} for k in PVS}
for l in open("verify_cache.jsonl", encoding="utf-8"):
    if l.strip():
        try:
            r = json.loads(l)
            for k, pv in PVS.items():
                if r.get("pv") == pv:
                    V[k][(r["p"], r["d"])] = r
        except (json.JSONDecodeError, KeyError):
            pass
print({k: len(v) for k, v in V.items()})

_pools = {}
def pool(p):
    if p not in _pools:
        br = [doc_ids[i] for i in np.argsort(bm25.get_scores(tok(p)))[::-1][:M]]
        _pools[p] = set(br) | set(bge[p][:M])
    return _pools[p]

def verdict(k, r, d):
    """judge k での membership。None=未検証あり"""
    preds = r["parsed"]["predicates"]
    comb = r["parsed"]["combine"]
    if comb == "or":
        need = [p for p in preds if d in pool(p)]
        if not need:
            return False
        vs = [V[k].get((p, d)) for p in need]
        if any(v is not None and v["v"] for v in vs):
            return True
        return None if any(v is None for v in vs) else False
    if not any(d in pool(p) for p in preds):
        return False
    vs = [V[k].get((p, d)) for p in preds]
    if any(v is not None and not v["v"] for v in vs):
        return False
    return None if any(v is None for v in vs) else True

def members(r, mode):
    out = set()
    for d in doc_ids:
        g, s = verdict("gpt", r, d), verdict("ds", r, d)
        if mode == "gpt" and g is True: out.add(d)
        elif mode == "ds" and s is True: out.add(d)
        elif mode == "inter" and g is True and s is True: out.add(d)
        elif mode == "union" and (g is True or s is True): out.add(d)
    return out

def prf(mem, gold):
    tp = len(mem & gold)
    P = tp / len(mem) if mem else 0.0
    R = tp / len(gold) if gold else 1.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F

# ---- 1+2) 4変種の集合比較 ----
print(f"\n=== 1) member集合F1 (m={M}): gpt / ds / 合意∩ / 和∪ ===")
agg = collections.defaultdict(list)
for rec in recs:
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    row = []
    for mode in ["gpt", "ds", "inter", "union"]:
        mem = members(r, mode)
        P, R, F = prf(mem, gold)
        agg[mode].append(F)
        row.append(f"{mode}:F={F:.2f}(|{len(mem)}|)")
    print(f"  qid{rec['qid']:>5} {rec['op']:13s} |gold|={len(gold):>3}  " + "  ".join(row))
    if rec["op"] == "count":
        cs = {mode: len(members(r, mode)) for mode in ["gpt", "ds", "inter", "union"]}
        print(f"      count予測: gold={rec['gold']}  " +
              "  ".join(f"{k}={v}" for k, v in cs.items()))
print("  -- set F1平均: " + "  ".join(f"{k}={np.mean(v):.2f}" for k, v in agg.items()))

# ---- 3) and全滅クエリの解剖 ----
print("\n=== 3) and結合クエリ: goldの述語別verify状況 ===")
for rec in recs:
    r = parsed[rec["qid"]]
    if r["parsed"]["combine"] != "and":
        continue
    preds = r["parsed"]["predicates"]
    print(f"\n  qid{rec['qid']} {rec['op']} 述語数={len(preds)} |gold|={len(r['golden_doc_ids'])}")
    for i, p in enumerate(preds):
        print(f"    述語[{i}]: {p[:90]}")
    shown = 0
    for g in r["golden_doc_ids"]:
        if shown >= 5:
            break
        rows = []
        for i, p in enumerate(preds):
            cell = []
            for k in ["gpt", "ds"]:
                v = V[k].get((p, g))
                if v is None:
                    cell.append(f"{k}:-")
                elif v["v"]:
                    cell.append(f"{k}:T")
                elif v.get("raw"):
                    cell.append(f"{k}:T→証拠NG")
                else:
                    cell.append(f"{k}:F")
            rows.append(f"[{i}]" + "/".join(cell))
        print(f"    gold doc{g:>5}: " + "  ".join(rows))
        shown += 1

# ---- 4) 極値: 採用境界docへの相手judgeの意見 ----
print("\n=== 4) 極値: 採用docに対する両judgeの判定 (合意フィルタの効果予測) ===")
ids_re = re.compile(r"\d+")
for rec in recs:
    if rec["op"] not in ("min_id", "max_id") and not rec["op"].startswith("topk"):
        continue
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    for d in map(int, ids_re.findall(rec["pred"])):
        tag = "GOLD" if d in gold else "FP"
        g, s = verdict("gpt", r, d), verdict("ds", r, d)
        note = "合意フィルタで排除可" if tag == "FP" and (g is False or s is False) else ""
        print(f"  qid{rec['qid']:>5} {rec['op']:13s} 採用doc{d:>5} [{tag}]"
              f" gpt={g} ds={s}  {note}")

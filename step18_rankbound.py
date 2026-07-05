# step18_rankbound.py - 順位束縛メンバーシップのオフライン較正 (課金ゼロ)
#   H2確定 (gold = 検索露出 ∩ DeepSeek判定) を受け、既存verifyキャッシュで
#   member_m = (いずれかの述語で順位<m) ∩ verify合格 を m スイープし、
#   集合P/R/F1 と Answer F1 を測る。対象はverifyがほぼ全候補に済んでいる
#   qid 173 (topk_smallest) と qid 659 (sort_asc)。
#   参考として「順位束縛のみ (verifyなし)」も併記し、verifyの寄与を示す。
import json, os, re, collections
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import pred_scores, token_f1

PV = "v22a"
DATA = "./data/GlobalQA"
FULL = [173, 659]      # verify網羅済みのクエリ (パイロットログ参照)
MS = [10, 20, 30, 50, 80, 100, 150, 200]

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

parsed = {}
for l in open("parsed_train60.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        parsed[r["idx"]] = r

verify = {}
for l in open("verify_cache.jsonl", encoding="utf-8"):
    if l.strip():
        try:
            v = json.loads(l)
            if v.get("pv") == PV:
                verify[(v["p"], v["d"])] = v["v"]
        except (json.JSONDecodeError, KeyError):
            pass

def ranks_for(pred):
    order = list(np.argsort(bm25.get_scores(tok(pred)))[::-1])
    br = {doc_ids[i]: rk for rk, i in enumerate(order)}
    s = pred_scores(pred)
    dr = {d: rk for rk, d in enumerate(sorted(s, key=s.get, reverse=True))}
    return br, dr

def fmt(op, ids, k):
    if op == "count":
        return str(len(ids))
    if op == "min_id":
        return str(ids[0]) if ids else ""
    if op == "max_id":
        return str(ids[-1]) if ids else ""
    if op == "topk_smallest":
        return ", ".join(map(str, ids[:k]))
    if op == "topk_largest":
        return ", ".join(map(str, ids[::-1][:k]))
    if op == "sort_asc":
        return ", ".join(map(str, ids))
    return ", ".join(map(str, ids[::-1]))

for qid in FULL:
    r = parsed[qid]
    preds = r["parsed"]["predicates"]
    comb = r["parsed"]["combine"]
    op = r["parsed"]["agg"]
    k = r["parsed"].get("k") or 1
    gold = set(r["golden_doc_ids"])
    pr = {p: ranks_for(p) for p in preds}

    def pred_rank(p, d):
        br, dr = pr[p]
        return min(br.get(d, 10**9), dr.get(d, 10**9))

    print(f"\n=== qid {qid} {op} ({comb}) |gold|={len(gold)} 正解=\"{r['answer']}\" ===")
    print(f"{'m':>4} | {'順位のみ P/R/F1':>22} | {'verify併用 P/R/F1':>22}"
          f" {'|mem|':>5} {'ansF1':>6}")
    for m in MS:
        rank_only, both = set(), set()
        for d in doc_ids:
            if comb == "or":
                elig = [p for p in preds if pred_rank(p, d) < m]
                if not elig:
                    continue
                rank_only.add(d)
                if any(verify.get((p, d)) is True for p in elig):
                    both.add(d)
            else:
                if not any(pred_rank(p, d) < m for p in preds):
                    continue
                rank_only.add(d)
                if all(verify.get((p, d)) is True for p in preds):
                    both.add(d)

        def prf(s):
            tp = len(s & gold)
            P = tp / len(s) if s else 0.0
            R = tp / len(gold)
            F = 2 * P * R / (P + R) if P + R else 0.0
            return P, R, F
        p1, r1, f1a = prf(rank_only)
        p2, r2, f2 = prf(both)
        ans = token_f1(fmt(op, sorted(both), k), r["answer"])
        print(f"{m:>4} | {p1:5.2f} {r1:5.2f} {f1a:5.2f}        "
              f"| {p2:5.2f} {r2:5.2f} {f2:5.2f}        "
              f"{len(both):>5} {ans*100:>6.1f}")

print("\n読み方:")
print("  - verify併用のF1が順位のみを明確に上回っていれば、二要素設計 (順位束縛+検証)")
print("    が正当化される。最良mが両クエリで近ければそのmをv2.3の既定値に採用")
print("  - m選択則: setR>=0.7を保つ最小のm付近で、Answer F1最大の点を取る")

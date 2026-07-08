# step29_clausebound.py - H3の裁定: 「goldは句内の各述語の検索でそれぞれ浅い」(課金ゼロ)
#   構築はGlobalQA論文の記述どおり検索ステップの集合演算 (intersection/union)。
#   ならばAND句由来のgoldは各conjunctのtop-r交差に載るはず。
#   step28の minrank (全述語のmin) はAND意味論に緩すぎた -> 句単位の maxrank で再裁定。
#   定義: clause_rank(c,d) = max_{p∈c} rank_p(d),  best_rank(d) = min_{c} clause_rank(c,d)
#         (rank_p = BM25順位とBGE順位の小さい方)
#   1) 全gold の best_rank 分布 (verify不要 -> rankout/unknown含む全goldを測定) = H3のリコール面
#   2) 現member FP の best_rank 分布 = 分離面
#   3) count/sort: member ∧ best_rank<r で再回答 (キャッシュのみ)
#   4) 極値: 採用FP と gold答えdoc の best_rank 比較
# 使い方: python step29_clausebound.py --m 30
import json, os, re, argparse, collections
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import token_f1

DATA = "./data/GlobalQA"

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train60")
ap.add_argument("--judge", default="gpt", choices=["gpt", "deepseek"])
ap.add_argument("--ds-model", default="deepseek-v4-flash", dest="ds_model")
ap.add_argument("--m", type=int, default=30)
args = ap.parse_args()
M = args.m
PV = "v22a" if args.judge == "gpt" else "ds-" + args.ds_model

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])
bge = json.load(open("bge_ranks.json", encoding="utf-8"))

parsed = {}
for l in open(f"parsed2_{args.split}.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        parsed[r["idx"]] = r

recs = [json.loads(l) for l in
        open(f"results_v25/v25_{args.split}_{args.judge}_m{M}.jsonl", encoding="utf-8")
        if l.strip()]

verify = {}
for l in open("verify_cache.jsonl", encoding="utf-8"):
    if l.strip():
        try:
            v = json.loads(l)
            if v.get("pv") == PV:
                verify[(v["p"], v["d"])] = v
        except (json.JSONDecodeError, KeyError):
            pass

INF = 10**9
_bmr, _bgr = {}, {}
def bm_rank(p):
    if p not in _bmr:
        order = np.argsort(bm25.get_scores(tok(p)))[::-1]
        _bmr[p] = {doc_ids[i]: rk for rk, i in enumerate(order)}
    return _bmr[p]
def bg_rank(p):
    if p not in _bgr:
        _bgr[p] = {d: rk for rk, d in enumerate(bge.get(p, []))}
    return _bgr[p]
def prank(p, d):
    return min(bm_rank(p).get(d, INF), bg_rank(p).get(d, INF))
def clause_rank(c, d):
    return max(prank(p, d) for p in c)
def best_rank(r, d):
    return min(clause_rank(c, d) for c in r["parsed"]["clauses"])

_pl = {}
def poolc(p):
    if p not in _pl:
        _pl[p] = ({d for d, rk in bm_rank(p).items() if rk < M} |
                  {d for d, rk in bg_rank(p).items() if rk < M})
    return _pl[p]

def clause_status(c, d):
    if not any(d in poolc(p) for p in c):
        return "unexposed"
    vs = [verify.get((p, d)) for p in c]
    if any(v is not None and not v["v"] for v in vs):
        return "false"
    if any(v is None for v in vs):
        return "unknown"
    return "true"

def doc_status(r, d):
    sts = [clause_status(c, d) for c in r["parsed"]["clauses"]]
    if "true" in sts:
        return "member"
    if all(s == "unexposed" for s in sts):
        return "rankout"
    if "unknown" in sts:
        return "unknown"
    return "vfalse"

RS = [10, 15, 20, 30, 50, 80]

# ---- 1) gold の best_rank 分布 (H3リコール面, verify不要) ----
print("\n=== 1) gold の best_rank カバレッジ (句単位: 全conjunctが<r) ===")
allg = []
for rec in recs:
    r = parsed[rec["qid"]]
    brs = [best_rank(r, g) for g in r["golden_doc_ids"]]
    allg += brs
    cov = "  ".join(f"r{r_}:{np.mean(np.array(brs) < r_)*100:3.0f}%" for r_ in RS)
    print(f"  qid{rec['qid']:>5} {rec['op']:13s} |gold|={len(brs):>3}"
          f" 中央値={int(np.median(brs)) if np.median(brs) < INF else '∞':>4}  {cov}")
allg = np.array(allg)
print("  -- 全gold: " + "  ".join(
    f"r{r_}:{np.mean(allg < r_)*100:.0f}%" for r_ in RS)
    + f"  中央値={int(np.median(allg)) if np.median(allg) < INF else '∞'}")

# ---- 2) member FP の best_rank 分布 (分離面) ----
print("\n=== 2) 現member FP の best_rank (goldとの分離) ===")
allf = []
for rec in recs:
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    fps = [d for d in doc_ids if doc_status(r, d) == "member" and d not in gold]
    if not fps:
        continue
    brs = np.array([best_rank(r, d) for d in fps])
    allf += list(brs)
    print(f"  qid{rec['qid']:>5} {rec['op']:13s} FP={len(fps):>3}"
          f" 中央値={int(np.median(brs)):>4}  " + "  ".join(
              f"r{r_}:{np.mean(brs < r_)*100:3.0f}%" for r_ in RS))
allf = np.array(allf)
if len(allf):
    print("  -- 全FP:   " + "  ".join(
        f"r{r_}:{np.mean(allf < r_)*100:.0f}%" for r_ in RS)
        + f"  中央値={int(np.median(allf))}")
print("  (H3が正しければ: goldのカバレッジ >> FPのカバレッジ となるrが存在する)")

# ---- 3) count/sort: 句単位束縛での再回答 ----
print(f"\n=== 3) count/sort: member ∧ best_rank<r (句単位) ===")
agg = {r_: [] for r_ in RS + [INF]}
for rec in recs:
    if rec["op"] not in ("count", "sort_asc", "sort_desc"):
        continue
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    mem = [d for d in doc_ids if doc_status(r, d) == "member"]
    line = f"  qid{rec['qid']:>5} {rec['op']:9s} gold={str(rec['gold'])[:10]:>10s}:"
    for r_ in RS + [INF]:
        ids = sorted(d for d in mem if best_rank(r, d) < r_)
        if rec["op"] == "count":
            pred = str(len(ids))
        elif rec["op"] == "sort_asc":
            pred = ", ".join(map(str, ids))
        else:
            pred = ", ".join(map(str, ids[::-1]))
        f1 = token_f1(pred, rec["gold"])
        agg[r_].append(f1)
        tag = f"n={len(ids)}" if rec["op"] == "count" else f"|{len(ids)}|"
        line += f"  r{'∞' if r_ == INF else r_}:{f1:.2f}({tag})"
    print(line)
print("  -- Answer F1平均: " + "  ".join(
    f"r{'∞' if r_ == INF else r_}={np.mean(v)*100:.1f}" for r_, v in agg.items()))

# ---- 4) 極値: 採用FP vs gold答えdoc の best_rank ----
print("\n=== 4) 極値: 採用doc と gold答えdoc の best_rank ===")
ids_re = re.compile(r"\d+")
for rec in recs:
    if rec["op"] not in ("min_id", "max_id") and not rec["op"].startswith("topk"):
        continue
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    for d in [int(x) for x in ids_re.findall(rec["pred"] or "")]:
        tag = "GOLD" if d in gold else "FP"
        print(f"  qid{rec['qid']:>5} {rec['op']:13s} 採用doc{d:>5} [{tag}]"
              f" best_rank={best_rank(r, d)}")
    for g in [int(x) for x in ids_re.findall(str(rec["gold"]))]:
        print(f"      gold答えdoc{g:>5}: best_rank={best_rank(r, g)}"
              f"  状態={doc_status(r, g)}")

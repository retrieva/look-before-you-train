# step28_rankbound2.py - 順位束縛 (v2.3) を v2.5 の上に事後フィルタとして復活させる無料裁定
#   仮説: gold=構築露出∩judge (H2) なので、「正当充足型FP」(2418の118件等) は
#   verify精度では切れず、露出順位で切るしかない。member ∧ (minrank < r) をスイープ。
#   1) count/sort: r ∈ {5,10,15,20,25,30,∞} で set-F1 / count予測 / Answer F1 を再計算
#   2) 極値: 採用FPと gold答えdoc の minrank を比較 (間に入るrで答えが翻るか)
#   3) gold vfalse の内訳: raw=True だが証拠無効化で死んだもの (証拠検証のFNコスト測定)
#   4) qid1078 を OR (述語不変) で再評価したときの member/答え (パース修正の上限見積り)
# 使い方: python step28_rankbound2.py --m 30   (課金ゼロ)
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
def minrank(preds, d):
    return min(min(bm_rank(p).get(d, INF), bg_rank(p).get(d, INF)) for p in preds)

def pool(p):
    br = {d for d, rk in bm_rank(p).items() if rk < M}
    gr = {d for d, rk in bg_rank(p).items() if rk < M}
    return br | gr
_pl = {}
def poolc(p):
    if p not in _pl:
        _pl[p] = pool(p)
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

# ---- 1) count/sort: 順位束縛スイープ ----
print(f"\n=== 1) count/sort: member ∧ minrank<r のスイープ (m={M}) ===")
RS = [5, 10, 15, 20, 25, 30, INF]
sweep_qids = [rec for rec in recs
              if rec["op"] in ("count", "sort_asc", "sort_desc")]
agg_f1 = {r_: [] for r_ in RS}
agg_set = {r_: [] for r_ in RS}
for rec in sweep_qids:
    r = parsed[rec["qid"]]
    preds = sorted({p for c in r["parsed"]["clauses"] for p in c})
    gold = set(r["golden_doc_ids"])
    mem = [d for d in doc_ids if doc_status(r, d) == "member"]
    line = f"  qid{rec['qid']:>5} {rec['op']:9s} gold={rec['gold'][:12]:>12s} |gold|={len(gold):>3}:"
    for r_ in RS:
        ids = sorted(d for d in mem if minrank(preds, d) < r_)
        if rec["op"] == "count":
            pred = str(len(ids))
        elif rec["op"] == "sort_asc":
            pred = ", ".join(map(str, ids))
        else:
            pred = ", ".join(map(str, ids[::-1]))
        f1 = token_f1(pred, rec["gold"])
        tp = len(set(ids) & gold)
        P = tp / len(ids) if ids else 0.0
        R = tp / len(gold)
        sF = 2 * P * R / (P + R) if P + R else 0.0
        agg_f1[r_].append(f1); agg_set[r_].append(sF)
        tag = f"|{len(ids)}|" if rec["op"] != "count" else f"n={len(ids)}"
        line += f"  r{'∞' if r_ == INF else r_}:{f1:.2f}({tag})"
    print(line)
print("  -- Answer F1平均: " + "  ".join(
    f"r{'∞' if r_ == INF else r_}={np.mean(agg_f1[r_])*100:.1f}" for r_ in RS))
print("  -- set F1平均:    " + "  ".join(
    f"r{'∞' if r_ == INF else r_}={np.mean(agg_set[r_]):.2f}" for r_ in RS))

# ---- 2) 極値: 採用FPとgold答えdocの順位分離 ----
print("\n=== 2) 極値: 採用doc と gold答えdoc の minrank 比較 ===")
ids_re = re.compile(r"\d+")
for rec in recs:
    if rec["op"] not in ("min_id", "max_id") and not rec["op"].startswith("topk"):
        continue
    r = parsed[rec["qid"]]
    preds = sorted({p for c in r["parsed"]["clauses"] for p in c})
    gold = set(r["golden_doc_ids"])
    picked = [int(x) for x in ids_re.findall(rec["pred"] or "")]
    gans = [int(x) for x in ids_re.findall(str(rec["gold"]))]
    for d in picked:
        tag = "GOLD" if d in gold else "FP"
        print(f"  qid{rec['qid']:>5} {rec['op']:13s} 採用doc{d:>5} [{tag}] minrank={minrank(preds, d)}")
    for g in gans:
        print(f"      gold答えdoc{g:>5}: minrank={minrank(preds, g)}"
              f"  状態={doc_status(r, g)}")

# ---- 3) gold vfalse の内訳: 証拠無効化によるFN ----
print("\n=== 3) gold vfalse の死因: verify自体か、証拠検証の無効化か ===")
n_vf = n_evkill = 0
for rec in recs:
    r = parsed[rec["qid"]]
    for g in r["golden_doc_ids"]:
        if doc_status(r, g) != "vfalse":
            continue
        n_vf += 1
        # raw判定を使えばtrueになる句があるか
        for c in r["parsed"]["clauses"]:
            if clause_status(c, g) != "false":
                continue
            vs = [verify.get((p, g)) for p in c]
            if all(v is not None and (v["v"] or v.get("raw")) for v in vs):
                n_evkill += 1
                print(f"  qid{rec['qid']:>5} gold doc{g:>5}: raw判定なら通過 (証拠無効化で死亡)")
                for p, v in zip(c, vs):
                    if v.get("raw") and not v["v"]:
                        print(f"      {p[:60]}  無効化された証拠:\"{v.get('ev','')[:80]}\"")
                break
print(f"  -- gold vfalse {n_vf}件中、証拠無効化が決定打: {n_evkill}件")

# ---- 4) qid1078 を OR に組み替えた場合 (述語不変・キャッシュのみ) ----
print("\n=== 4) qid1078 OR再評価 (パース修正の上限見積り) ===")
if 1078 in parsed:
    r = parsed[1078]
    preds = sorted({p for c in r["parsed"]["clauses"] for p in c})
    gold = set(r["golden_doc_ids"])
    mem, unk = set(), 0
    for d in set().union(*[poolc(p) for p in preds]):
        vs = [verify.get((p, d)) for p in preds if d in poolc(p)]
        if any(v is not None and v["v"] for v in vs):
            mem.add(d)
        elif any(v is None for v in vs):
            unk += 1
    tp = len(mem & gold)
    P = tp / len(mem) if mem else 0.0
    R = tp / len(gold)
    sF = 2 * P * R / (P + R) if P + R else 0.0
    ans = max(mem) if mem else None
    print(f"  OR時: |mem|={len(mem)} 未検証={unk} setF1={sF:.2f}"
        f"  max_id予測={ans} (gold答え: {r['answer']})")
    for r_ in [10, 15, 20]:
        bm_ = [d for d in mem if minrank(preds, d) < r_]
        print(f"    束縛r{r_}: |mem|={len(bm_)} max_id={max(bm_) if bm_ else None}")

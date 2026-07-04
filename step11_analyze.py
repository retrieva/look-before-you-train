# step11_analyze.py - v2パイロットの誤り分析 (API課金ゼロ、キャッシュのみで解剖)
#   1) sort/count: 実体化された集合の precision/recall
#   2) min/max: 答えの文書が「候補生成/プロファイル判定/原文確認」のどの段で消えたか
#   3) topk: 選ばれたIDのうちgolden集合に入っている割合
import json, os, re, collections
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import pred_scores

DATA = "./data/GlobalQA"
N = 200
ids_re = re.compile(r"\d+")

parsed = {}
for l in open("parsed_train60.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        parsed[r["idx"]] = r

recs = [json.loads(l) for l in open("results_v2/v2_train60.jsonl", encoding="utf-8")
        if l.strip()]

def load_cache(path):
    c = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            if l.strip():
                try:
                    r = json.loads(l)
                    c[(r["p"], r["d"])] = r["v"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return c

judge = load_cache("judge_cache.jsonl")
confirm = load_cache("confirm_cache.jsonl")

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

def cands_for(r):
    c = set()
    for p in r["parsed"]["predicates"]:
        s = pred_scores(p)
        c.update(sorted(s, key=s.get, reverse=True)[:N])
        c.update([doc_ids[i] for i in np.argsort(bm25.get_scores(tok(p)))[::-1][:N]])
    return c

def member_of(r, d):
    preds = r["parsed"]["predicates"]
    vals = [judge.get((p, d), None) for p in preds]
    if any(v is None for v in vals):
        return None
    return any(vals) if r["parsed"]["combine"] == "or" else all(vals)

# ---- 1) sort/count: 集合品質 ----
print("\n=== 1) 実体化された集合の品質 (sort=リスト全体 / count=サイズ) ===")
setP, setR = [], []
for rec in recs:
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    if rec["op"].startswith("sort"):
        pred_ids = set(map(int, ids_re.findall(rec["pred"])))
        tp = len(pred_ids & gold)
        P = tp / len(pred_ids) if pred_ids else 0.0
        R = tp / len(gold)
        setP.append(P); setR.append(R)
        print(f"  qid{rec['qid']:>5} {rec['op']:9s}: P={P:.2f} R={R:.2f}"
              f"  |pred|={len(pred_ids)} |gold|={len(gold)}")
    elif rec["op"] == "count":
        print(f"  qid{rec['qid']:>5} count    : pred={rec['pred']:>4} gold={rec['gold']:>4}"
              f"  (|gold set|={len(gold)})")
if setP:
    print(f"  -- sort平均: precision={np.mean(setP):.2f}  recall={np.mean(setR):.2f}")

# ---- 2) min/max: 敗因の段特定 ----
print("\n=== 2) min/max: 答えの文書はどの段で消えたか ===")
for rec in recs:
    if rec["op"] not in ("min_id", "max_id"):
        continue
    r = parsed[rec["qid"]]
    try:
        g = int(str(r["answer"]).strip())
    except ValueError:
        continue
    stage_c = g in cands_for(r)
    stage_j = member_of(r, g)
    confs = [confirm.get((p, g)) for p in r["parsed"]["predicates"]]
    verdict = ("候補生成でMISS" if not stage_c else
               "プロファイル判定でfalse" if stage_j is False else
               "判定キャッシュに無し(未判定)" if stage_j is None else
               "memberには居た(confirm/選択の問題)")
    print(f"  qid{rec['qid']:>5} {rec['op']:6s}: 予測={rec['pred']:>5} 正解={g:>5}"
          f"  -> {verdict}  confirm={confs}")

# ---- 3) topk: 選択IDの正答率 ----
print("\n=== 3) topk: 選ばれたIDがgolden集合に入っている割合 ===")
for rec in recs:
    if not rec["op"].startswith("topk"):
        continue
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    pred_ids = list(map(int, ids_re.findall(rec["pred"])))
    hit = sum(d in gold for d in pred_ids)
    print(f"  qid{rec['qid']:>5} {rec['op']:13s}: {hit}/{len(pred_ids)} in-gold"
          f"  pred={pred_ids} gold_ans={rec['gold']}")

# ---- 4) プロファイル判定の偽陰性率 (golden docベース, キャッシュのみ) ----
print("\n=== 4) プロファイル判定の偽陰性率 (判定済みgolden docのみ) ===")
n_true = n_false = 0
for rec in recs:
    r = parsed[rec["qid"]]
    for g in r["golden_doc_ids"]:
        m = member_of(r, g)
        if m is True: n_true += 1
        elif m is False: n_false += 1
tot = n_true + n_false
if tot:
    print(f"  golden doc 判定済み {tot}件中、member判定true={n_true} false={n_false}"
          f"  -> 偽陰性率 {n_false/tot*100:.1f}%")

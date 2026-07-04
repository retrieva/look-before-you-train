# step16_analyze22.py - v2.2の誤り分析 (課金ゼロ、キャッシュのみ)
#   gold docごとに 候補生成 / トリアージ / verify / (member在籍) の段階特定 +
#   全クエリの member集合 P/R + 証拠引用の無効化率
import json, os, re, collections
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import pred_scores

PV = "v22a"          # step15と一致させること
DATA = "./data/GlobalQA"
N = 200

parsed = {}
for l in open("parsed_train60.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        parsed[r["idx"]] = r

recs = [json.loads(l) for l in open("results_v22/v22_train60.jsonl", encoding="utf-8")
        if l.strip()]

def load(path):
    c = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            if l.strip():
                try:
                    r = json.loads(l)
                    if r.get("pv") == PV:
                        c[(r["p"], r["d"])] = r
                except (json.JSONDecodeError, KeyError):
                    pass
    return c

triage = load("triage_cache.jsonl")
verify = load("verify_cache.jsonl")

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
    """キャッシュからmember判定を再構成 (None=判定不能)"""
    preds = r["parsed"]["predicates"]
    comb = r["parsed"]["combine"]
    vals = []
    for p in preds:
        t = triage.get((p, d))
        if t and t["v"] == "no":
            vals.append(False)
            continue
        v = verify.get((p, d))
        vals.append(v["v"] if v else None)
    if comb == "or":
        if any(v is True for v in vals):
            return True
        return None if any(v is None for v in vals) else False
    if any(v is False for v in vals):
        return False
    return None if any(v is None for v in vals) else True

# ---- 1) gold docの段階特定 ----
print("\n=== 1) gold doc はどの段で消えたか (member=falseのgoldのみ) ===")
stage_cnt = collections.Counter()
for rec in recs:
    r = parsed[rec["qid"]]
    cs = cands_for(r)
    preds = r["parsed"]["predicates"]
    comb = r["parsed"]["combine"]
    lost = []
    for g in r["golden_doc_ids"]:
        if member_of(r, g) is True:
            stage_cnt["member在籍"] += 1
            continue
        if g not in cs:
            stage_cnt["候補生成MISS"] += 1
            lost.append((g, "候補MISS", ""))
            continue
        ts = [triage.get((p, g), {}).get("v") for p in preds]
        vs = [verify.get((p, g), {}).get("v") for p in preds]
        raws = [verify.get((p, g), {}).get("raw") for p in preds]
        if comb == "or":
            if all(t == "no" for t in ts):
                stage_cnt["トリアージ全no"] += 1
                lost.append((g, "triage全no", str(ts)))
            elif any(rv is True and vv is False
                     for rv, vv in zip(raws, vs)):
                stage_cnt["証拠引用の無効化でfalse"] += 1
                lost.append((g, "証拠無効化", str(vs)))
            else:
                stage_cnt["verifyでfalse"] += 1
                lost.append((g, "verify false", f"t={ts} v={vs}"))
        else:
            if any(t == "no" for t in ts):
                stage_cnt["トリアージno(and)"] += 1
                lost.append((g, "triage no(and)", str(ts)))
            elif any(rv is True and vv is False
                     for rv, vv in zip(raws, vs)):
                stage_cnt["証拠引用の無効化でfalse"] += 1
                lost.append((g, "証拠無効化", str(vs)))
            else:
                stage_cnt["verifyでfalse"] += 1
                lost.append((g, "verify false", f"t={ts} v={vs}"))
    for g, st, detail in lost[:3]:
        print(f"  qid{rec['qid']:>5} {rec['op']:13s} gold doc{g:>5}: {st}  {detail[:80]}")

tot = sum(stage_cnt.values())
print("\n  -- 段階別集計 (gold doc単位) --")
for k, v in stage_cnt.most_common():
    print(f"  {k:24s} {v:>4} ({v/tot*100:.0f}%)")

# ---- 2) member集合の品質 (全op) ----
print("\n=== 2) member集合 P/R (全クエリ、キャッシュから再構成) ===")
Ps, Rs = [], []
for rec in recs:
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    member = {d for d in cands_for(r) if member_of(r, d) is True}
    tp = len(member & gold)
    P = tp / len(member) if member else 0.0
    R = tp / len(gold) if gold else 1.0
    Ps.append(P); Rs.append(R)
    print(f"  qid{rec['qid']:>5} {rec['op']:13s}: P={P:.2f} R={R:.2f}"
          f"  |mem|={len(member)} |gold|={len(gold)}")
print(f"  -- 平均: precision={np.mean(Ps):.2f}  recall={np.mean(Rs):.2f}"
      f"  (v2.1: P=0.02 R=0.40)")

# ---- 3) 証拠引用の無効化率 (verify全体) ----
raw_true = sum(1 for v in verify.values() if v.get("raw"))
inval = sum(1 for v in verify.values() if v.get("raw") and not v["v"])
if raw_true:
    print(f"\n=== 3) 証拠引用: LLMがtrueと言った {raw_true}件中 "
          f"{inval}件 ({inval/raw_true*100:.1f}%) を引用不一致で無効化 ===")
    ex = [v for v in verify.values() if v.get("raw") and not v["v"]][:5]
    for v in ex:
        print(f"  無効化例: evidence=\"{v.get('ev','')[:90]}\"")

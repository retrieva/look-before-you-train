# step24_combine.py - combine (and/or) 取り違えの検証と適応的combineの効果測定 (課金ゼロ)
#   1) and指定クエリの元の質問文を表示 (パーサ誤りの目視確認用)
#   2) gold docの述語充足構造: 全述語充足(and整合) vs 1つ以上充足(or整合) の比率
#   3) 適応的combine (orで実体化 -> |member|>50ならand) をキャッシュからシミュレートし、
#      set F1とcount予測が現行 (パーサのcombineに従う) からどう動くか測る
import json, re, collections, argparse
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import pred_scores

PVS = ["v22a", "ds-deepseek-v4-flash"]   # gpt優先、無ければds
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

rows = [json.loads(l) for l in open("parsed_train60.jsonl", encoding="utf-8")
        if l.strip()]
parsed = {r["idx"]: r for r in rows}

# 元の質問文のフィールド名を自動検出
sample = rows[0]
qfield = None
for k, v in sample.items():
    if k not in ("parsed", "answer", "golden_doc_ids", "idx") and \
       isinstance(v, str) and len(v) > 20:
        qfield = k
        break
print(f"質問文フィールド: {qfield}  (レコードのキー: {list(sample.keys())})")

V = {}
for l in open("verify_cache.jsonl", encoding="utf-8"):
    if l.strip():
        try:
            r = json.loads(l)
            if r.get("pv") in PVS and (r["p"], r["d"]) not in V:
                pass
            if r.get("pv") == "v22a":
                V[(r["p"], r["d"])] = r["v"]
        except (json.JSONDecodeError, KeyError):
            pass
# dsで補完 (gptに無いペアのみ)
for l in open("verify_cache.jsonl", encoding="utf-8"):
    if l.strip():
        try:
            r = json.loads(l)
            if r.get("pv") == "ds-deepseek-v4-flash" and (r["p"], r["d"]) not in V:
                V[(r["p"], r["d"])] = r["v"]
        except (json.JSONDecodeError, KeyError):
            pass
print(f"verify判定 (gpt優先+ds補完): {len(V)}ペア")

_pools = {}
def pool(p):
    if p not in _pools:
        br = [doc_ids[i] for i in np.argsort(bm25.get_scores(tok(p)))[::-1][:M]]
        _pools[p] = set(br) | set(bge[p][:M])
    return _pools[p]

recs = [json.loads(l) for l in
        open(f"results_v24/v24_train60_bge_deepseek_m{M}.jsonl", encoding="utf-8")
        if l.strip()]

# ---- 1) and指定クエリの質問文 ----
print("\n=== 1) and指定クエリの元の質問文 ===")
for rec in recs:
    r = parsed[rec["qid"]]
    if r["parsed"]["combine"] != "and":
        continue
    q = r.get(qfield, "(質問文フィールド不明)") if qfield else "(不明)"
    print(f"\n  qid{rec['qid']} {rec['op']} |gold|={len(r['golden_doc_ids'])}")
    print(f"    Q: {str(q)[:200]}")
    for i, p in enumerate(r["parsed"]["predicates"]):
        print(f"    述語[{i}]: {p[:90]}")

# ---- 2) goldの述語充足構造 ----
print("\n=== 2) gold docの述語充足構造 (verify済みgoldのみ) ===")
for rec in recs:
    r = parsed[rec["qid"]]
    preds = r["parsed"]["predicates"]
    if len(preds) < 2:
        continue
    n_all = n_some = n_known = 0
    for g in r["golden_doc_ids"]:
        vs = [V.get((p, g)) for p in preds]
        if all(v is not None for v in vs):
            n_known += 1
            if all(vs):
                n_all += 1
            elif any(vs):
                n_some += 1
    if n_known:
        tag = "→ or整合" if n_some > n_all else ("→ and整合" if n_all and not n_some else "")
        print(f"  qid{rec['qid']:>5} ({r['parsed']['combine']:>3}指定): 判定済gold {n_known}件中"
              f" 全述語充足={n_all} 一部のみ充足={n_some} {tag}")

# ---- 3) 適応的combineのシミュレーション ----
print(f"\n=== 3) 適応的combine (or優先, |member|>50ならand) の効果 (m={M}) ===")
def member(r, comb):
    preds = r["parsed"]["predicates"]
    out = set()
    for d in set().union(*[pool(p) for p in preds]):
        if comb == "or":
            need = [p for p in preds if d in pool(p)]
            vs = [V.get((p, d)) for p in need]
            if any(v is True for v in vs):
                out.add(d)
        else:
            vs = [V.get((p, d)) for p in preds]
            if all(v is True for v in vs):
                out.add(d)
    return out

def prf(mem, gold):
    tp = len(mem & gold)
    P = tp / len(mem) if mem else 0.0
    R = tp / len(gold)
    F = 2 * P * R / (P + R) if P + R else 0.0
    return F

cur_F, ada_F = [], []
for rec in recs:
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    cur = member(r, r["parsed"]["combine"])
    m_or = member(r, "or")
    ada = m_or if len(m_or) <= 50 else member(r, "and")
    ada_comb = "or" if len(m_or) <= 50 else "and"
    f_cur, f_ada = prf(cur, gold), prf(ada, gold)
    cur_F.append(f_cur); ada_F.append(f_ada)
    mark = " <<改善" if f_ada > f_cur + 1e-9 else (" <<悪化" if f_ada < f_cur - 1e-9 else "")
    print(f"  qid{rec['qid']:>5} {rec['op']:13s} 現行({r['parsed']['combine']}):F={f_cur:.2f}(|{len(cur)}|)"
          f"  適応({ada_comb}):F={f_ada:.2f}(|{len(ada)}|){mark}")
    if rec["op"] == "count":
        print(f"      count: gold={rec['gold']} 現行={len(cur)} 適応={len(ada)}")
print(f"\n  set F1平均: 現行={np.mean(cur_F):.2f} -> 適応={np.mean(ada_F):.2f}")
print("  注: 極値系はverify未実施docがあるため下振れ (count/sortの行が最も信頼できる)")

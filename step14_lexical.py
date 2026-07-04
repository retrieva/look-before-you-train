# step14_lexical.py - 字句カバレッジだけのベースライン (課金ゼロ, train60全60問)
#   目的:
#   1) 述語の内容語カバレッジ閾値 τ による member 再現度 (集合P/R/F1) と
#      最終Answer F1 を τ スイープで測定 -> LLM判定をどこに配分すべきか決める
#   2) gold集合サイズの分布を確認 (論文の構築規則: gold は常に50件以下)
#   3) gold docのカバレッジ天井 (τごとのリコール上限) を測定
#   論文ベースライン行 "lexical coverage baseline" としてもそのまま使用可
import json, re, collections
import numpy as np
from step3_eval import token_f1

DATA = "./data/GlobalQA"

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
STOP = set(("a an the of in with and or for to on at by is are who has have had "
            "experience experienced expertise skilled skills background work worked "
            "working knowledge proficient familiar strong someone people person "
            "candidates resumes years using used use related relate").split())
def content_toks(s):
    return [t for t in tok(s) if t not in STOP and len(t) > 2]

print("コーパス読み込み中...")
docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_tok = {d["id"]: set(tok(d["contents"])) for d in docs}
all_ids = sorted(doc_tok)

rows = [json.loads(l) for l in open("parsed_train60.jsonl", encoding="utf-8")
        if l.strip()]
print(f"クエリ数: {len(rows)}")

# ---- 述語ごとのカバレッジを前計算 ----
uniq_preds = sorted({p for r in rows for p in r["parsed"]["predicates"]})
cov = {}
for p in uniq_preds:
    ts = content_toks(p)
    if not ts:
        cov[p] = {d: 1.0 for d in all_ids}
    else:
        cov[p] = {d: sum(t in doc_tok[d] for t in ts) / len(ts) for d in all_ids}
print(f"述語数 (unique): {len(uniq_preds)}")

# ---- 0) gold集合サイズの分布 (論文: 常に <=50) ----
sizes = [len(r["golden_doc_ids"]) for r in rows]
print(f"\n=== 0) gold集合サイズ (train60) ===")
print(f"  min={min(sizes)} median={int(np.median(sizes))} max={max(sizes)}"
      f"  mean={np.mean(sizes):.1f}  >50件: {sum(s > 50 for s in sizes)}問")

# ---- 3) gold docカバレッジ天井 ----
print("\n=== gold doc のカバレッジ天井 (max over 述語) ===")
maxcovs = []
for r in rows:
    for g in r["golden_doc_ids"]:
        if g in doc_tok:
            maxcovs.append(max(cov[p][g] for p in r["parsed"]["predicates"]))
maxcovs = np.array(maxcovs)
for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"  τ={tau:.1f}: gold docの {np.mean(maxcovs >= tau)*100:5.1f}% がリコール可能")

# ---- 1) τスイープ: 集合品質と最終F1 ----
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
    return ", ".join(map(str, ids[::-1]))    # sort_desc

print("\n=== τスイープ: lexical-only の集合再現度と Answer F1 (train60全問) ===")
print(f"{'τ':>5} {'setP':>6} {'setR':>6} {'|mem|中央値':>10} {'ALL F1':>7}  per-op F1")
best = (-1, None)
for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    per_op = collections.defaultdict(list)
    Ps, Rs, msizes = [], [], []
    for r in rows:
        preds = r["parsed"]["predicates"]
        comb = r["parsed"]["combine"]
        op = r["parsed"]["agg"]
        k = r["parsed"].get("k") or 1
        member = []
        for d in all_ids:
            vals = [cov[p][d] >= tau for p in preds]
            if (any(vals) if comb == "or" else all(vals)):
                member.append(d)
        gold = set(r["golden_doc_ids"])
        tp = len(set(member) & gold)
        Ps.append(tp / len(member) if member else 0.0)
        Rs.append(tp / len(gold) if gold else 1.0)
        msizes.append(len(member))
        f1 = token_f1(fmt(op, member, k), r["answer"])
        per_op[op].append(f1)
        per_op["_all"].append(f1)
    allf1 = np.mean(per_op["_all"]) * 100
    ops = "  ".join(f"{o}:{np.mean(v)*100:.1f}"
                    for o, v in sorted(per_op.items()) if not o.startswith("_"))
    print(f"{tau:>5.1f} {np.mean(Ps):>6.2f} {np.mean(Rs):>6.2f}"
          f" {int(np.median(msizes)):>10} {allf1:>7.2f}  {ops}")
    if allf1 > best[0]:
        best = (allf1, tau)

print(f"\n最良: τ={best[1]} で ALL F1={best[0]:.2f}"
      f"  (参考: v2パイロット10.92 / v0 1.80 / GlobalRAG SOTA 6.63)")
print("読み方:")
print("  - setPが高くsetRも保てるτがある -> lexicalを厳格プレフィルタの主軸にし、")
print("    LLM判定は境界帯 (τ未満だがBM25/dense上位) の救済に限定できる = 安い")
print("  - どのτでもsetPが低い -> gold基準はlexicalで近似不可。厳格フルテキスト")
print("    LLM判定 (証拠引用検証つき) を全候補に回す設計になる = 高い")
print("  - |mem|中央値はgold分布 (中央値~15-20, 最大50) と比較すること")

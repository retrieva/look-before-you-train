# step7_diagnose.py - v1の故障箇所を切り分ける
#   Part A (無料): 正解文書の類似度がどの帯にいるか -> リコール上限が分かる
#   Part B (--probe, 目安$0.1): 正解文書をLLMに判定させ偽陰性率を測る
# 使い方:
#   python step7_diagnose.py            # Part Aのみ (API課金ゼロ, 全キャッシュ済み)
#   python step7_diagnose.py --probe    # Part A + B
import json, argparse, random, collections
import numpy as np
from step3_eval import pred_scores  # 述語embeddingはpred_cache済み -> 無料

LOWS = [0.56, 0.50, 0.46, 0.42, 0.38, 0.34, 0.30]

rows = [json.loads(l) for l in open("parsed_train60.jsonl", encoding="utf-8") if l.strip()]

def golden_soft(r):
    """各golden docの述語式ソフトスコア (or=max, and=min)"""
    per_pred = [pred_scores(p) for p in r["parsed"]["predicates"]]
    agg = max if r["parsed"]["combine"] == "or" else min
    return {g: agg(s.get(g, 0.0) for s in per_pred) for g in r["golden_doc_ids"]}

# ---------------- Part A: リコール分析 (無料) ----------------
print("=== Part A: golden docのソフトスコア分布 (recall上限の特定) ===")
by_op = collections.defaultdict(list)   # op -> [golden softスコア...]
ans_doc_scores = []                     # min/max: 「答えの文書」のスコア
for r in rows:
    gs = golden_soft(r)
    op = r["parsed"]["agg"]
    by_op[op] += list(gs.values())
    if op in ("min_id", "max_id"):
        try:
            ans_doc_scores.append((op, gs.get(int(str(r["answer"]).strip()), None)))
        except ValueError:
            pass

all_scores = [s for v in by_op.values() for s in v]
print(f"\ngolden docs total: {len(all_scores)}")
print(f"{'LOW':>6} | recall(golden soft >= LOW)")
for L in LOWS:
    rec = np.mean([s >= L for s in all_scores])
    print(f"{L:>6.2f} | {rec*100:5.1f}%")

print("\n-- op別 recall@0.42 (現行LOW) / @0.34 --")
for op in sorted(by_op):
    v = by_op[op]
    print(f"  {op:14s} @0.42={np.mean([s>=0.42 for s in v])*100:5.1f}%"
          f"  @0.34={np.mean([s>=0.34 for s in v])*100:5.1f}%  (n={len(v)})")

print("\n-- min/max: 『答えそのものの文書』のスコア --")
for op, s in ans_doc_scores:
    band = ("IN(>=0.56)" if s is not None and s >= 0.56 else
            "band[0.42,0.56)" if s is not None and s >= 0.42 else
            "BELOW LOW (<0.42)  <- 検証の目に触れない" if s is not None else
            "answerがgoldenに無い?")
    print(f"  {op}: soft={s if s is None else round(s,3)}  {band}")

# ---------------- Part B: 偽陰性プローブ (少額課金) ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--probe", action="store_true")
ap.add_argument("--max-calls", type=int, default=90)
args = ap.parse_args()
if not args.probe:
    print("\n(Part Bは --probe 指定時のみ実行。golden文書へのLLM判定で偽陰性率を測ります)")
    raise SystemExit

print("\n=== Part B: LLM検証の偽陰性プローブ ===")
from step5_verify_v1 import (load_chunk_texts, embed_predicate, llm_verify,
                             _load_verify_cache, _load_pred_emb_cache,
                             CORPUS_PATH)
_load_verify_cache(); _load_pred_emb_cache()
embs = np.load("chunk_embs.npy")
embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
owner = json.load(open("chunk_owner.json"))
texts = load_chunk_texts(CORPUS_PATH, n_expected=len(owner))
doc_chunks = collections.defaultdict(list)
for ci, d in enumerate(owner): doc_chunks[d].append(ci)

def best_chunk(pred, d):
    pv = embed_predicate(pred)
    idxs = doc_chunks[d]
    sims = embs[idxs] @ pv
    return idxs[int(np.argmax(sims))]

random.seed(0)
stats = collections.Counter()
fn_examples = []
calls = 0
sample = [r for r in rows if r["parsed"]["combine"] == "or"]
random.shuffle(sample)
for r in sample:
    if calls >= args.max_calls: break
    gs = golden_soft(r)
    # 帯内〜帯上のgolden docを対象 (帯の下はPart Aの守備範囲)
    cands = [g for g, s in gs.items() if s >= 0.42][:2]
    for g in cands:
        if calls >= args.max_calls: break
        preds = sorted(r["parsed"]["predicates"],
                       key=lambda p: pred_scores(p).get(g, 0), reverse=True)
        any_true = False
        for p in preds:
            c = collections.Counter()
            ok = llm_verify(p, g, texts[best_chunk(p, g)], c)
            calls += c["llm_calls"]
            if ok:
                any_true = True
                break
        stats["golden_checked"] += 1
        if not any_true:
            stats["false_negative"] += 1
            if len(fn_examples) < 3:
                fn_examples.append((r["question"][:70], g))

n = stats["golden_checked"]
print(f"golden docs judged: {n}, LLM calls: {calls}")
if n:
    print(f"偽陰性率 (golden なのに全述語 false): {stats['false_negative']/n*100:.1f}%")
for q, g in fn_examples:
    print(f"  FN例: doc {g} / Q: {q}")

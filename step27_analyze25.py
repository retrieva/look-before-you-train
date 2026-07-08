# step27_analyze25.py - v2.5 (DNF) パイロットの無料解剖 (課金ゼロ)
#   裁定する仮説: 「8.12 < 9.18 はDNF構造の欠陥ではなく、step25の述語文字列変更による
#   (a) BGEプール欠落 (step21未再実行 -> step26のbge_topが黙って[]を返す) と
#   (b) verifyキャッシュ失効、の交絡である」
#   1) BGE欠落述語の棚卸し (プールがBM25単独になっている述語)
#   2) per-qid Answer F1: v2.5 vs v2.4 (results_v24/ に存在する全judge変種)
#   3) gold損失段階 (DNF版): member / 露出ゼロ(rank-out) / verifyでFalse / 未検証(極値打ち切り)
#   4) count解剖: キャッシュ再構成member vs gold件数、FP上位の通過証拠
#   5) 極値: 採用docの正体 (FPなら通過句と証拠) + gold端docの死因
# 使い方: python step27_analyze25.py --m 30
#   step21再実行の「前後」で1回ずつ打つと、プール復元の効果が 1)/3) の数字で見える。
#   注: step20/23/24 は parsed_*.jsonl (フラット) 前提。v2.5系の解析は本スクリプトに集約。
import json, os, re, glob, argparse, collections
import numpy as np
from rank_bm25 import BM25Okapi

PV = "v22a"
DATA = "./data/GlobalQA"

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train60")
ap.add_argument("--judge", default="gpt", choices=["gpt", "deepseek"])
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

missing_bge = set()
_pools = {}
def pool(p):
    if p not in _pools:
        br = [doc_ids[i] for i in np.argsort(bm25.get_scores(tok(p)))[::-1][:M]]
        gr = bge.get(p)
        if gr is None:
            missing_bge.add(p)
            gr = []
        _pools[p] = set(br) | set(gr[:M])
    return _pools[p]

def clause_status(c, d):
    """句cのdでの評価: 'true' / 'false' / 'unknown' / 'unexposed'"""
    if not any(d in pool(p) for p in c):
        return "unexposed"
    vs = [verify.get((p, d)) for p in c]
    if any(v is not None and not v["v"] for v in vs):
        return "false"
    if any(v is None for v in vs):
        return "unknown"
    return "true"

def doc_status(r, d):
    """'member' / 'vfalse' / 'unknown' / 'rankout'"""
    sts = [clause_status(c, d) for c in r["parsed"]["clauses"]]
    if "true" in sts:
        return "member"
    if all(s == "unexposed" for s in sts):
        return "rankout"
    if "unknown" in sts:
        return "unknown"
    return "vfalse"

def members_from_cache(r):
    return {d for d in doc_ids if doc_status(r, d) == "member"}

# ---- 1) BGE欠落述語の棚卸し ----
print("\n=== 1) BGE欠落述語 (プールがBM25単独 = step21未較正) ===")
tot_p = tot_m = 0
for rec in recs:
    r = parsed[rec["qid"]]
    preds = sorted({p for c in r["parsed"]["clauses"] for p in c})
    for p in preds:
        pool(p)  # プール構築 (欠落検出を兼ねる)
    miss = [p for p in preds if p in missing_bge]
    tot_p += len(preds); tot_m += len(miss)
    if miss:
        print(f"  qid{rec['qid']:>5}: {len(miss)}/{len(preds)} 述語がBGE欠落")
        for p in miss:
            print(f"      - {p[:85]}")
print(f"  -- 合計: {tot_m}/{tot_p} 述語 ({tot_m/max(tot_p,1)*100:.0f}%) が欠落。"
      f"0でない限りstep21再実行が先 (このパイロットはDNFの公平測定になっていない)")

# ---- 2) per-qid Answer F1 diff ----
print("\n=== 2) per-qid Answer F1: v2.5 vs v2.4 ===")
v24 = {}
for path in sorted(glob.glob(f"results_v24/v24_{args.split}_bge_*_m{M}.jsonl")):
    tag = os.path.basename(path).split("_bge_")[1].rsplit("_m", 1)[0]
    v24[tag] = {j["qid"]: j for j in
                (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}
if not v24:
    print("  (results_v24/*.jsonl が見つからないためスキップ)")
else:
    hdr = "  qid    op              v25 "
    for t in v24:
        hdr += f"  v24:{t:>8s}"
    print(hdr)
    for rec in recs:
        row = f"  {rec['qid']:>5}  {rec['op']:13s} {rec['f1']:5.2f}"
        for t in v24:
            j = v24[t].get(rec["qid"])
            row += f"  {j['f1']:12.2f}" if j else f"  {'-':>12s}"
        print(row)
    qids = [rec["qid"] for rec in recs]
    line = f"  平均: v25={np.mean([rec['f1'] for rec in recs])*100:.2f}"
    for t in v24:
        vs = [v24[t][q]["f1"] for q in qids if q in v24[t]]
        if vs:
            line += f"  v24:{t}={np.mean(vs)*100:.2f} (n={len(vs)})"
    print(line)

# ---- 3) gold損失段階 (DNF版) ----
print("\n=== 3) gold docの損失段階 (DNF版) ===")
cnt = collections.Counter()
rankout_bgeflag = 0
for rec in recs:
    r = parsed[rec["qid"]]
    c = collections.Counter()
    for g in r["golden_doc_ids"]:
        s = doc_status(r, g)
        c[s] += 1
        cnt[s] += 1
        if s == "rankout" and any(any(p in missing_bge for p in cl)
                                  for cl in r["parsed"]["clauses"]):
            rankout_bgeflag += 1
    print(f"  qid{rec['qid']:>5} {rec['op']:13s} |gold|={len(r['golden_doc_ids']):>3}: "
          f"mem={c['member']:>3} rankout={c['rankout']:>3} "
          f"verifyF={c['vfalse']:>3} 未検証={c['unknown']:>3}")
tot = sum(cnt.values())
label = {"member": "member在籍", "rankout": "露出ゼロ(rank-out)",
         "vfalse": "verifyでFalse", "unknown": "未検証(打ち切り圏外)"}
print("  --- 集計 ---")
for k, v in cnt.most_common():
    print(f"  {label[k]:22s} {v:>4} ({v/tot*100:.0f}%)")
print(f"  rank-outのうち、BGE欠落述語を含むクエリ由来: {rankout_bgeflag} "
      f"(step21再実行で回復し得る上限の目安)")

# ---- 4) count解剖 ----
print("\n=== 4) count解剖 (キャッシュ再構成) ===")
for rec in recs:
    if rec["op"] != "count":
        continue
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    mem = members_from_cache(r)
    fp = sorted(mem - gold); fn = sorted(gold - mem)
    print(f"  qid{rec['qid']:>5}: pred={rec['pred']} gold={rec['gold']}"
          f"  |mem|={len(mem)} FP={len(fp)} FN={len(fn)}")
    for d in fp[:3]:
        for cl in r["parsed"]["clauses"]:
            if clause_status(cl, d) == "true":
                print(f"      FP doc{d:>5} 通過句: ({' AND '.join(p[:40] for p in cl)})")
                for p in cl:
                    v = verify.get((p, d))
                    print(f"        {p[:60]}  証拠:\"{(v or {}).get('ev','')[:80]}\"")
                break

# ---- 5) 極値: 採用docの正体 + gold端の死因 ----
print("\n=== 5) 極値: 採用docの正体 (FPなら通過証拠) ===")
ids_re = re.compile(r"\d+")
for rec in recs:
    if rec["op"] not in ("min_id", "max_id") and not rec["op"].startswith("topk"):
        continue
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    picked = [int(x) for x in ids_re.findall(rec["pred"] or "")]
    if not picked:
        print(f"  qid{rec['qid']:>5} {rec['op']:13s} 採用なし (member 0)")
    for d in picked:
        tag = "GOLD" if d in gold else "FP"
        print(f"  qid{rec['qid']:>5} {rec['op']:13s} 採用doc{d:>5} [{tag}]"
              f"  (gold答え: {rec['gold']})")
        if tag == "FP":
            for cl in r["parsed"]["clauses"]:
                if clause_status(cl, d) == "true":
                    for p in cl:
                        v = verify.get((p, d))
                        print(f"      通過: {p[:60]}  証拠:\"{(v or {}).get('ev','')[:80]}\"")
                    break
    ex = min(gold) if rec["op"] in ("min_id", "topk_smallest") else max(gold)
    print(f"      gold端 doc{ex:>5} の状態: {doc_status(r, ex)}")

print("\n判断基準: 1)の欠落が0%になるまで再パイロット禁止 / 3)のrank-out率がv2.4比で"
      "悪化していれば原因はプール、member率が同等以上でF1だけ低ければ原因は境界FP。")

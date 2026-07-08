# step20_analyze23.py - v2.3の誤り分析 + 公式仕様F1との整合検証 (課金ゼロ)
#   1) member集合のP/R/F1 (rank-bound + verifyキャッシュから再構成) と count誤差
#   2) gold docの損失段階: 順位圏外 / verify false / 未検証(早期打ち切り圏外)
#   3) min/max/topk: 採用された境界docの正体 (goldか、FPなら証拠引用を表示)
#   4) 公式仕様 (SQuAD系正規化・集合F1) で全結果を再採点し、自作token_f1と比較
import json, os, re, string, argparse, collections
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import pred_scores, token_f1

PV = "v22a"
DATA = "./data/GlobalQA"

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=20)
args = ap.parse_args()
M = args.m

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

recs = [json.loads(l) for l in
        open(f"results_v23/v23_train60_m{M}.jsonl", encoding="utf-8") if l.strip()]

verify = {}
for l in open("verify_cache.jsonl", encoding="utf-8"):
    if l.strip():
        try:
            v = json.loads(l)
            if v.get("pv") == PV:
                verify[(v["p"], v["d"])] = v
        except (json.JSONDecodeError, KeyError):
            pass

_pools = {}
def pool(p):
    if p not in _pools:
        br = [doc_ids[i] for i in np.argsort(bm25.get_scores(tok(p)))[::-1][:M]]
        s = pred_scores(p)
        dr = sorted(s, key=s.get, reverse=True)[:M]
        _pools[p] = set(br) | set(dr)
    return _pools[p]

def member_of(r, d):
    """None=未検証あり判定不能"""
    preds = r["parsed"]["predicates"]
    comb = r["parsed"]["combine"]
    if comb == "or":
        need = [p for p in preds if d in pool(p)]
        if not need:
            return False
        vs = [verify.get((p, d)) for p in need]
        if any(v is not None and v["v"] for v in vs):
            return True
        return None if any(v is None for v in vs) else False
    else:
        if not any(d in pool(p) for p in preds):
            return False
        vs = [verify.get((p, d)) for p in preds]
        if any(v is not None and not v["v"] for v in vs):
            return False
        return None if any(v is None for v in vs) else True

# ---- 公式仕様のAnswer F1 (SQuAD系正規化, トークン集合) ----
def normalize_text(text):
    if not text:
        return ""
    t = text.lower()
    t = "".join(ch if ch not in set(string.punctuation) else " " for ch in t)
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())

def official_f1(pred, gold):
    np_, ng = normalize_text(str(pred)), normalize_text(str(gold))
    pt, gt = np_.split(), ng.split()
    if not pt or not gt:
        return 1.0 if np_ == ng else 0.0
    same = len(set(pt) & set(gt))
    if same == 0:
        return 0.0
    P, R = same / len(pt), same / len(gt)
    return 2 * P * R / (P + R)

# ---- 1) member集合品質 ----
print(f"\n=== 1) member集合 P/R/F1 (m={M}) ===")
Fs = []
for rec in recs:
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    mem = {d for d in doc_ids if member_of(r, d) is True}
    unk = sum(1 for d in doc_ids if member_of(r, d) is None)
    tp = len(mem & gold)
    P = tp / len(mem) if mem else 0.0
    R = tp / len(gold)
    F = 2 * P * R / (P + R) if P + R else 0.0
    Fs.append(F)
    extra = f" count誤差={int(rec['pred'] or 0) - int(rec['gold']):+d}" \
        if rec["op"] == "count" else ""
    print(f"  qid{rec['qid']:>5} {rec['op']:13s}: P={P:.2f} R={R:.2f} F1={F:.2f}"
          f"  |mem|={len(mem)} |gold|={len(gold)} 未検証={unk}{extra}")
print(f"  -- set F1平均: {np.mean(Fs):.2f}")

# ---- 2) gold損失の段階 ----
print("\n=== 2) gold docの損失段階 ===")
cnt = collections.Counter()
for rec in recs:
    r = parsed[rec["qid"]]
    preds = r["parsed"]["predicates"]
    for g in r["golden_doc_ids"]:
        mv = member_of(r, g)
        if mv is True:
            cnt["member在籍"] += 1
        elif mv is None:
            cnt["未検証(打ち切り圏外)"] += 1
        elif not any(g in pool(p) for p in preds):
            cnt["順位圏外(rank-out)"] += 1
        else:
            cnt["verifyでfalse"] += 1
tot = sum(cnt.values())
for k, v in cnt.most_common():
    print(f"  {k:22s} {v:>4} ({v/tot*100:.0f}%)")

# ---- 3) 極値系: 採用された境界docの正体 ----
print("\n=== 3) min/max/topk: 採用docはgoldか (FPなら証拠を表示) ===")
ids_re = re.compile(r"\d+")
for rec in recs:
    if rec["op"] not in ("min_id", "max_id") and not rec["op"].startswith("topk"):
        continue
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    for d in map(int, ids_re.findall(rec["pred"])):
        tag = "GOLD" if d in gold else "FP"
        print(f"  qid{rec['qid']:>5} {rec['op']:13s} 採用doc{d:>5} [{tag}]"
              f"  (gold答え: {rec['gold']})")
        if tag == "FP":
            for p in r["parsed"]["predicates"]:
                v = verify.get((p, d))
                if v and v["v"]:
                    print(f"      通過述語: {p[:70]}")
                    print(f"      証拠: \"{v.get('ev','')[:100]}\"")

# ---- 4) 公式仕様F1で再採点 ----
print("\n=== 4) 公式仕様F1 vs 自作token_f1 ===")
ours, offs = [], []
diff = 0
for rec in recs:
    o = token_f1(rec["pred"], rec["gold"])
    f = official_f1(rec["pred"], rec["gold"])
    ours.append(o); offs.append(f)
    if abs(o - f) > 1e-6:
        diff += 1
        print(f"  qid{rec['qid']:>5}: 自作={o:.3f} 公式={f:.3f}  pred=\"{rec['pred'][:40]}\"")
print(f"  平均: 自作={np.mean(ours)*100:.2f} 公式={np.mean(offs)*100:.2f}  不一致{diff}件")
print("  (不一致0件なら自作token_f1は公式仕様と等価とみなせる)")

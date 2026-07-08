# step33b_scorejudge2.py - 較正ラウンド2b: 帯域ルーブリック採点 (q2b) + 拡張スイープ
#   ラウンド2 (step33, q2-gpt) の裁定: スコアが{0,5,10}に量子化 (87%)、5点帯は
#   gold率43%のコイン投げ291ペア。τの崖 (5→6でTPR 93.8→66.8) の正体。
#   -> 帯域定義ルーブリックで同一928ペアを再採点し、5点帯が割れるか裁定する。
#   ペア構成は step32/33 と完全同一 (seed7・同一乱数系列)。ただし極値系タスクは
#   gold答えdoc (min/max) がcap抽出で落ちた場合に追加ペアとして補完 (base928は不変)。
#   スイープは TPR/FPR に加え、pair-F1 と極値の答え生存率も直接表示する。
#
# 事前予測 (2026-07-06 登録):
#   P1: 旧5点帯291ペアが細分され、帯内で新スコアに対しgold率が単調になる
#   P2: countはτをFPR<40%側に押し込めるようになり個数較正が改善
#   P3: 満点非gold (count 38.8% / min_id 45.8%) は細分後も高得点に残る (判定不能成分は不変)
#   P4 (リスク): gold 7 vs 非gold 9 の逆転が増え、極値の答え生存がq2比で悪化しうる
#      -> 出たら「極値はτ=5相当の低閾値固定 + flood-fill」で確定
#
# 使い方:
#   python step33b_scorejudge2.py --limit 40 --judge gpt              (採点 ~$1-2 + 無料スイープ)
#   python step33b_scorejudge2.py --limit 40 --judge gpt --sweep-only (課金ゼロ)
#   python step33b_scorejudge2.py --limit 40 --judge deepseek         (ds比較, §8カード3)
import json, os, re, random, argparse, collections
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
from util import with_retry

DATA = "./data/GlobalQA"
random.seed(7)

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="gpt", choices=["gpt", "deepseek"])
ap.add_argument("--ds-model", default="deepseek-v4-flash", dest="ds_model")
ap.add_argument("--limit", type=int, default=40)
ap.add_argument("--cap", type=int, default=20)
ap.add_argument("--neg-pool", type=int, default=40, dest="neg_pool")
ap.add_argument("--sweep-only", action="store_true", dest="sweep_only")
ap.add_argument("--no-answer-doc", action="store_true", dest="no_answer_doc",
                help="極値gold答えdocの補完ペアを追加しない (base928のみ)")
args = ap.parse_args()

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
raw_text = {d["id"]: d["contents"] for d in docs}
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

rows = [json.loads(l) for l in open("calib_sample.jsonl", encoding="utf-8") if l.strip()]
rows.sort(key=lambda r: r["idx"])
def task_of(q):
    ql = q.lower()
    if "how many" in ql: return "count"
    m = re.search(r"top\s*(\d+)", ql)
    if m: return "topk_smallest" if ("smallest" in ql or "lowest" in ql) else "topk_largest"
    if "ascending" in ql: return "sort_asc"
    if "descending" in ql: return "sort_desc"
    if "smallest" in ql or "earliest" in ql or "lowest" in ql: return "min_id"
    return "max_id"
byt = collections.defaultdict(list)
for r in rows:
    byt[task_of(r["question"])].append(r)
per = max(1, args.limit // max(len(byt), 1))
sel = []
for t, rs in sorted(byt.items()):
    sel += rs[:per]
print(f"較正質問: {len(sel)}問 (step32/33と同一構成)")

# ---- q2bプロンプト: 帯域定義ルーブリック (量子化対策) ----
SCORE_SYS = (
    "You rate document relevance for a corpus-level query over a resume corpus. "
    "Output an integer 0-10. Rubric: 9-10 = unambiguously one of the documents the "
    "query asks about (matches the full profile/conditions described); 7-8 = very "
    "likely a match but one condition is implicit or only weakly evidenced; 4-6 = "
    "same occupation family or satisfies only part of the conditions; 2-3 = "
    "tangential overlap only; 0-1 = unrelated. Avoid defaulting to 0/5/10; first "
    "choose the band, then pick the exact integer within it to express confidence.")

S_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "score", "strict": True, "schema": {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"], "additionalProperties": False}}}

DS_SPEC = ' Respond ONLY with a JSON object {"score": <integer 0-10>}.'

class Judge:
    def __init__(self, kind, ds_model):
        self.kind = kind
        if kind == "deepseek":
            self.pv = "q2b-ds-" + ds_model
            self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                                 base_url="https://api.deepseek.com", timeout=180)
            self.model = ds_model
        else:
            self.pv = "q2b-gpt"
            self.client = OpenAI(timeout=180)
            self.model = "gpt-5-mini"

    def call(self, question, text):
        user = (f"QUERY: {question}\n\nDOCUMENT:\n{text}\n\n"
                "Score 0-10 how clearly this document is one of the documents "
                "the query is asking about.")
        if self.kind == "deepseek":
            r = with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SCORE_SYS + DS_SPEC},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"}))
        else:
            r = with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SCORE_SYS},
                          {"role": "user", "content": user}],
                response_format=S_SCHEMA))
        c = re.sub(r"^```(json)?|```$", "", r.choices[0].message.content.strip(),
                   flags=re.M)
        return max(0, min(10, int(json.loads(c)["score"])))

judge = Judge(args.judge, args.ds_model)
CACHE = "calib_cache.jsonl"
cache = {}  # (pv, idx, d) -> score  (q2/q2b両方読み込む: 比較解析に使用)
if os.path.exists(CACHE):
    for l in open(CACHE, encoding="utf-8"):
        if l.strip():
            j = json.loads(l)
            if "s" in j:
                cache[(j["pv"], j["idx"], j["d"])] = j["s"]
cf = open(CACHE, "a", encoding="utf-8")

def score(idx, question, d):
    k = (judge.pv, idx, d)
    if k not in cache:
        if args.sweep_only:
            return None
        s = judge.call(question, raw_text[d][:24000])
        cache[k] = s
        cf.write(json.dumps({"pv": judge.pv, "idx": idx, "d": d, "s": s},
                            ensure_ascii=False) + "\n")
        cf.flush()
    return cache[k]

EXTREME = {"min_id", "max_id", "topk_smallest", "topk_largest"}
def gold_answer_doc(task, gold):
    return min(gold) if task in ("min_id", "topk_smallest") else max(gold)

# ---- ペア構築 (step32/33と同一乱数系列) と採点 ----
pairs = []   # (task, idx, doc, label, score, is_extra)
for r in sel:
    gold = set(r["golden_doc_ids"])
    pos = sorted(gold)
    if len(pos) > args.cap:
        pos = sorted(random.sample(pos, args.cap))   # ← 乱数系列を変えないこと
    br = [doc_ids[i] for i in
          np.argsort(bm25.get_scores(tok(r["question"])))[::-1][:args.neg_pool]]
    negs = [d for d in br if d not in gold][:len(pos)]
    t = task_of(r["question"])
    extras = []
    if (not args.no_answer_doc) and t in EXTREME:
        ad = gold_answer_doc(t, gold)
        if ad not in pos:
            extras.append(ad)   # cap抽出で答えdocが落ちた場合の補完 (base928外)
    for d in pos:
        s = score(r["idx"], r["question"], d)
        if s is not None:
            pairs.append((t, r["idx"], d, 1, s, False))
    for d in extras:
        s = score(r["idx"], r["question"], d)
        if s is not None:
            pairs.append((t, r["idx"], d, 1, s, True))
    for d in negs:
        s = score(r["idx"], r["question"], d)
        if s is not None:
            pairs.append((t, r["idx"], d, 0, s, False))
    print(f"[idx {r['idx']:>5}] {t:13s} 採点済 "
          f"{sum(1 for p in pairs if p[1]==r['idx'])}ペア"
          + (f" (+答えdoc補完{len(extras)})" if extras else ""), flush=True)

base = [p for p in pairs if not p[5]]
tasks = sorted({p[0] for p in pairs})

# ---- スイープ1: TPR/FPR (base928, q2と直接比較可能) ----
print(f"\n=== ラウンド2b: {judge.pv} (base n={len(base)}, 補完込み n={len(pairs)}) ===")
sc_pos = [s for *_, l, s, _ in base if l == 1]
sc_neg = [s for *_, l, s, _ in base if l == 0]
print(f"  goldスコア   平均={np.mean(sc_pos):.1f} 分布 " +
      " ".join(f"{v}:{sc_pos.count(v)}" for v in range(11)))
print(f"  非goldスコア 平均={np.mean(sc_neg):.1f} 分布 " +
      " ".join(f"{v}:{sc_neg.count(v)}" for v in range(11)))
print(f"\n  τ(score>=τでmember)ごとの TPR/FPR (base):")
print(f"  {'τ':>3} | {'全体TPR':>7} {'全体FPR':>7} | タスク別TPR/FPR")
for tau in range(1, 11):
    tp = sum(1 for *_, l, s, _ in base if l == 1 and s >= tau)
    fn = sum(1 for *_, l, s, _ in base if l == 1 and s < tau)
    fp = sum(1 for *_, l, s, _ in base if l == 0 and s >= tau)
    tn = sum(1 for *_, l, s, _ in base if l == 0 and s < tau)
    bt = []
    for t in tasks:
        tpp = sum(1 for p in base if p[0] == t and p[3] == 1 and p[4] >= tau)
        fnn = sum(1 for p in base if p[0] == t and p[3] == 1 and p[4] < tau)
        fpp = sum(1 for p in base if p[0] == t and p[3] == 0 and p[4] >= tau)
        tnn = sum(1 for p in base if p[0] == t and p[3] == 0 and p[4] < tau)
        bt.append(f"{t[:6]}:{tpp/max(tpp+fnn,1)*100:.0f}/{fpp/max(fpp+tnn,1)*100:.0f}")
    print(f"  {tau:>3} | {tp/max(tp+fn,1)*100:6.1f}% {fp/max(fp+tn,1)*100:6.1f}% | "
          + " ".join(bt))

# ---- スイープ2: 問平均pair-F1 (member-F1代理) ----
byq = collections.defaultdict(lambda: {"pos": [], "neg": [], "task": None,
                                       "gold": None, "extra_pos": []})
sample_by_idx = {r["idx"]: r for r in rows}
for t, idx, d, l, s, ex in pairs:
    q = byq[idx]; q["task"] = t
    q["gold"] = set(sample_by_idx[idx]["golden_doc_ids"])
    if l == 1 and ex: q["extra_pos"].append((d, s))
    elif l == 1:      q["pos"].append((d, s))
    else:             q["neg"].append((d, s))

def pf1(pos, neg, tau):
    tp = sum(1 for _, s in pos if s >= tau)
    fp = sum(1 for _, s in neg if s >= tau)
    fn = len(pos) - tp
    return 2 * tp / max(2 * tp + fp + fn, 1)

print(f"\n  τ別 問平均pair-F1 (base):")
print(f"  {'τ':>3} | " + " ".join(f"{t[:7]:>8}" for t in tasks) + " |   全体")
for tau in range(1, 11):
    line, allf = [], []
    for t in tasks:
        fs = [pf1(q["pos"], q["neg"], tau) for q in byq.values() if q["task"] == t]
        line.append(np.mean(fs)); allf += fs
    print(f"  {tau:>3} | " + " ".join(f"{m*100:7.1f}%" for m in line)
          + f" | {np.mean(allf)*100:5.1f}%")

# ---- スイープ3: 極値の答え生存率 (補完doc込み = 実運用に近い) ----
print(f"\n  τ別 極値答え的中 (答えdoc補完込み。q2は補完なしなので参考比較):")
print(f"  {'τ':>3} | " + " ".join(f"{t[:7]:>8}" for t in tasks if t in EXTREME))
AGG = {"min_id": min, "topk_smallest": min, "max_id": max, "topk_largest": max}
for tau in range(3, 11):
    line = []
    for t in [t for t in tasks if t in EXTREME]:
        hit = tot = 0
        for idx, q in byq.items():
            if q["task"] != t: continue
            allpos = q["pos"] + q["extra_pos"]
            pred = [d for d, s in allpos + q["neg"] if s >= tau]
            if not pred: tot += 1; continue
            tot += 1
            hit += (AGG[t](pred) == AGG[t](q["gold"]))
        line.append(f"{hit}/{tot}")
    print(f"  {tau:>3} | " + " ".join(f"{x:>8}" for x in line))

# ---- 比較解析: q2 (旧5点帯) がq2bでどこに移ったか (P1裁定) ----
old_pv = "q2-gpt" if judge.kind == "gpt" else "q2-ds-" + args.ds_model
mig = {1: collections.Counter(), 0: collections.Counter()}
n_old5 = 0
for t, idx, d, l, s, ex in base:
    if cache.get((old_pv, idx, d)) == 5:
        n_old5 += 1
        mig[l][s] += 1
if n_old5:
    print(f"\n  旧{old_pv}=5点帯 ({n_old5}ペア) の新スコア分布 (P1裁定):")
    print("    gold  : " + " ".join(f"{v}:{mig[1][v]}" for v in range(11)))
    print("    非gold: " + " ".join(f"{v}:{mig[0][v]}" for v in range(11)))
    print("    帯別gold率: " + " ".join(
        f"{lo}-{hi}:{sum(mig[1][v] for v in range(lo,hi+1))}/"
        f"{sum(mig[1][v]+mig[0][v] for v in range(lo,hi+1))}"
        for lo, hi in [(0, 1), (2, 3), (4, 6), (7, 8), (9, 10)]))
else:
    print(f"\n  (旧{old_pv}キャッシュ無し: 5点帯移動解析はスキップ)")

# ---- P3裁定: 満点非goldの残存 ----
print(f"\n  P3裁定 満点(>=9)非gold率: " + " ".join(
    f"{t[:6]}:{sum(1 for p in base if p[0]==t and p[3]==0 and p[4]>=9)}"
    f"/{sum(1 for p in base if p[0]==t and p[3]==0)}"
    for t in tasks))
print("\n読み方: P1は帯別gold率の単調性、P2はcount行のFPR、P4は極値答え的中のq2比。")
print("済んだら calib_cache.jsonl を再解析へ (τ確定 → step34/v3.0実装)。")

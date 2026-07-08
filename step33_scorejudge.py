# step33_scorejudge.py - 較正ラウンド2: 採点式query-level判定 (0-10) + タスク別閾値スイープ
#   ラウンド1 (step32) の裁定: 二値J3は TPR70.3/FPR36.2 で基準未達。問ごとに
#   「分離可能/gold緩め/gold絞り」の3体制が混在し、単一二値では合わせられない。
#   -> 同一928ペアに0-10スコアを取り、タスク別に閾値τをスイープしてROCを引く。
#   ペア構成は step32 と完全同一 (同じseed・同じ層化・同じ負例規則)。
#   キャッシュ: calib_cache.jsonl (pv=q2-gpt / q2-ds-*)。スイープは無料で再実行可。
# 使い方:
#   python step33_scorejudge.py --limit 40 --judge gpt      (採点 ~$1-2 + 無料スイープ)
#   python step33_scorejudge.py --limit 40 --judge gpt --sweep-only   (課金ゼロ)
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
print(f"較正質問: {len(sel)}問 (step32と同一構成)")

SCORE_SYS = (
    "You rate document relevance for a corpus-level query over a resume corpus. "
    "Output an integer score 0-10: 10 = this resume is clearly one of the documents "
    "the query is asking about (it matches the profile/conditions described); "
    "5 = partial or borderline match (same occupation family or satisfies only part "
    "of the conditions); 0 = unrelated. Use the full range.")

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
            self.pv = "q2-ds-" + ds_model
            self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                                 base_url="https://api.deepseek.com", timeout=180)
            self.model = ds_model
        else:
            self.pv = "q2-gpt"
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
cache = {}
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

# ---- ペア構築 (step32と同一) と採点 ----
pairs = []  # (task, idx, doc, label, score)
for r in sel:
    gold = set(r["golden_doc_ids"])
    pos = sorted(gold)
    if len(pos) > args.cap:
        pos = sorted(random.sample(pos, args.cap))
    br = [doc_ids[i] for i in
          np.argsort(bm25.get_scores(tok(r["question"])))[::-1][:args.neg_pool]]
    negs = [d for d in br if d not in gold][:len(pos)]
    t = task_of(r["question"])
    for d in pos:
        s = score(r["idx"], r["question"], d)
        if s is not None:
            pairs.append((t, r["idx"], d, 1, s))
    for d in negs:
        s = score(r["idx"], r["question"], d)
        if s is not None:
            pairs.append((t, r["idx"], d, 0, s))
    print(f"[idx {r['idx']:>5}] {t:13s} 採点済 {sum(1 for p in pairs if p[1]==r['idx'])}ペア",
          flush=True)

# ---- 閾値スイープ (無料) ----
print(f"\n=== ラウンド2 スコア分布とスイープ: {judge.pv} (n={len(pairs)}) ===")
sc_pos = [s for *_, l, s in pairs if l == 1]
sc_neg = [s for *_, l, s in pairs if l == 0]
print(f"  goldスコア   平均={np.mean(sc_pos):.1f} 分布 " +
      " ".join(f"{v}:{sc_pos.count(v)}" for v in range(11)))
print(f"  非goldスコア 平均={np.mean(sc_neg):.1f} 分布 " +
      " ".join(f"{v}:{sc_neg.count(v)}" for v in range(11)))
print(f"\n  τ(score>=τでmember)ごとの TPR/FPR:")
print(f"  {'τ':>3} | {'全体TPR':>7} {'全体FPR':>7} | タスク別TPR/FPR")
for tau in range(1, 11):
    tp = sum(1 for *_, l, s in pairs if l == 1 and s >= tau)
    fn = sum(1 for *_, l, s in pairs if l == 1 and s < tau)
    fp = sum(1 for *_, l, s in pairs if l == 0 and s >= tau)
    tn = sum(1 for *_, l, s in pairs if l == 0 and s < tau)
    bytask = []
    for t in sorted({p[0] for p in pairs}):
        tpp = sum(1 for tt, *_, l, s in [(p[0], p[1], p[2], p[3], p[4]) for p in pairs]
                  if tt == t and l == 1 and s >= tau)
        fnn = sum(1 for p in pairs if p[0] == t and p[3] == 1 and p[4] < tau)
        fpp = sum(1 for p in pairs if p[0] == t and p[3] == 0 and p[4] >= tau)
        tnn = sum(1 for p in pairs if p[0] == t and p[3] == 0 and p[4] < tau)
        bytask.append(f"{t[:6]}:{tpp/max(tpp+fnn,1)*100:.0f}/{fpp/max(fpp+tnn,1)*100:.0f}")
    print(f"  {tau:>3} | {tp/max(tp+fn,1)*100:6.1f}% {fp/max(fp+tn,1)*100:6.1f}% | "
          + " ".join(bytask))
print("\n読み方: count/min_idは高τ (FPR<15%狙い)、sort系は低τ (TPR優先) を選ぶ。")
print("gold/非goldのスコア分布が重なりきっている問題型は判定不能成分 (露出限定gold)。")

# step32_calibrate.py - 較正ラウンド1: query-level判定 (J3, eq.(7)流) をtrainラベルで測る
#   step31でFN回復11/18を確認済み。本スクリプトはコインの裏 (FP率) を含む
#   総合成績を、calib_sample (train由来・train60と独立) の正負ペアで測定する。
#   正例: 各問のgold doc (上限20/問にサンプリング)
#   負例: 質問文BM25のtop-N内の非gold doc (正例と同数) = エンジンが実際に迷う領域
#   指標: pair TPR (goldをTrueと言えた率) / FPR (非goldをTrueと言った率) /
#         precision / pair-F1 / タスク別内訳 / 問ごとの成績ワースト
# 使い方:
#   python step32_calibrate.py --limit 40 --judge gpt   (~1,400ペア, 概算$1-2)
#   python step32_calibrate.py --limit 40 --judge deepseek
#   キャッシュ: calib_cache.jsonl (キー pv×idx×doc)。冪等。
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
ap.add_argument("--limit", type=int, default=40, help="使用する較正質問数")
ap.add_argument("--cap", type=int, default=20, help="1問あたりの正例上限 (負例も同数)")
ap.add_argument("--neg-pool", type=int, default=40, dest="neg_pool",
                help="負例を採るBM25順位の深さ")
args = ap.parse_args()

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
raw_text = {d["id"]: d["contents"] for d in docs}
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

rows = [json.loads(l) for l in open("calib_sample.jsonl", encoding="utf-8")
        if l.strip()]
rows.sort(key=lambda r: r["idx"])
# タスク層化を保ったままlimitに絞る (idx順で各タスク均等に)
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
print(f"較正質問: {len(sel)}問 (タスク層化, {per}/タスク)")

QUERY_SYS = (
    "You judge document relevance for a corpus-level query. Answer whether the "
    "document contains information to answer the query, i.e. whether it is one of "
    "the documents the query is asking about.")

Q_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "rel", "strict": True, "schema": {
        "type": "object",
        "properties": {"relevant": {"type": "boolean"},
                       "reason": {"type": "string"}},
        "required": ["relevant", "reason"], "additionalProperties": False}}}

DS_SPEC = (' Respond ONLY with a JSON object {"relevant": <true/false>, '
           '"reason": "<short reason>"}.')

class Judge:
    def __init__(self, kind, ds_model):
        self.kind = kind
        if kind == "deepseek":
            self.pv = "q1-ds-" + ds_model
            self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                                 base_url="https://api.deepseek.com", timeout=180)
            self.model = ds_model
        else:
            self.pv = "q1-gpt"
            self.client = OpenAI(timeout=180)
            self.model = "gpt-5-mini"

    def call(self, question, text):
        user = (f"QUERY: {question}\n\nDOCUMENT:\n{text}\n\n"
                "Does the document contain information to answer the query "
                "(is it one of the documents being asked about)?")
        if self.kind == "deepseek":
            r = with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": QUERY_SYS + DS_SPEC},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"}))
        else:
            r = with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": QUERY_SYS},
                          {"role": "user", "content": user}],
                response_format=Q_SCHEMA))
        c = re.sub(r"^```(json)?|```$", "", r.choices[0].message.content.strip(),
                   flags=re.M)
        return bool(json.loads(c)["relevant"])

judge = Judge(args.judge, args.ds_model)
CACHE = "calib_cache.jsonl"
cache = {}
if os.path.exists(CACHE):
    for l in open(CACHE, encoding="utf-8"):
        if l.strip():
            j = json.loads(l)
            cache[(j["pv"], j["idx"], j["d"])] = j["v"]
cf = open(CACHE, "a", encoding="utf-8")

def jcall(idx, question, d):
    k = (judge.pv, idx, d)
    if k not in cache:
        v = judge.call(question, raw_text[d][:24000])
        cache[k] = v
        cf.write(json.dumps({"pv": judge.pv, "idx": idx, "d": d, "v": v},
                            ensure_ascii=False) + "\n")
        cf.flush()
    return cache[k]

# ---- ペア構築と判定 ----
TP = FP = TN = FN = 0
by_task = collections.defaultdict(lambda: [0, 0, 0, 0])  # TP FP TN FN
perq = []
n_pairs = 0
for r in sel:
    gold = set(r["golden_doc_ids"])
    pos = sorted(gold)
    if len(pos) > args.cap:
        pos = sorted(random.sample(pos, args.cap))
    br = [doc_ids[i] for i in
          np.argsort(bm25.get_scores(tok(r["question"])))[::-1][:args.neg_pool]]
    negs = [d for d in br if d not in gold][:len(pos)]
    t = task_of(r["question"])
    q_tp = q_fp = q_tn = q_fn = 0
    for d in pos:
        v = jcall(r["idx"], r["question"], d)
        n_pairs += 1
        if v: TP += 1; by_task[t][0] += 1; q_tp += 1
        else: FN += 1; by_task[t][3] += 1; q_fn += 1
    for d in negs:
        v = jcall(r["idx"], r["question"], d)
        n_pairs += 1
        if v: FP += 1; by_task[t][1] += 1; q_fp += 1
        else: TN += 1; by_task[t][2] += 1; q_tn += 1
    acc = (q_tp + q_tn) / max(q_tp + q_tn + q_fp + q_fn, 1)
    perq.append((acc, r["idx"], t, q_tp, q_fn, q_fp, q_tn))
    print(f"[idx {r['idx']:>5}] {t:13s} TPR={q_tp}/{q_tp+q_fn}"
          f" FP={q_fp}/{q_fp+q_tn} acc={acc:.2f}", flush=True)

# ---- 集計 ----
tpr = TP / max(TP + FN, 1)
fpr = FP / max(FP + TN, 1)
prec = TP / max(TP + FP, 1)
f1 = 2 * prec * tpr / max(prec + tpr, 1e-9)
print(f"\n=== 較正ラウンド1: {judge.pv} (質問{len(sel)}問, ペア{n_pairs}) ===")
print(f"  TPR (gold検出率)     : {tpr*100:.1f}%   <- 現行strictの疫病はここが低いこと")
print(f"  FPR (非gold誤検出率) : {fpr*100:.1f}%   <- 負例はBM25 top-{args.neg_pool}の紛らわしい層")
print(f"  precision={prec*100:.1f}%  pair-F1={f1*100:.1f}")
print("\n  タスク別 (TPR / FPR):")
for t, (tp, fp, tn, fn) in sorted(by_task.items()):
    print(f"    {t:13s} TPR={tp/max(tp+fn,1)*100:5.1f}%  FPR={fp/max(fp+tn,1)*100:5.1f}%")
print("\n  成績ワースト5問 (プロンプト改善のネタ):")
for acc, idx, t, tp, fn, fp, tn in sorted(perq)[:5]:
    print(f"    idx{idx:>5} {t:13s} acc={acc:.2f}  FN={fn} FP={fp}")
print("\n判断基準: TPR 85%+ かつ FPR 20%以下なら v3.0 (query-level membership) へ移行。")
print("TPRが高くFPRも高い -> プロンプトに厳格化の一文を足してラウンド2 (キャッシュは判定不変分のみ再利用不可、PV更新を忘れずに)。")

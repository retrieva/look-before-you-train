# step34_v30.py - v3.0 混成エンジン: 辞書優先 + judgeフォールバック + flood-fill
#   設計 (HANDOFF v5 §7 を辞書発見で改訂):
#   1. DNFはparsed2から (パース済み)
#   2. member第一層 = step35辞書 (フレーズ→docプール, train逆推定, 課金ゼロ)
#   3. 未解決クローズのみ: 候補 = BM25 top-m ∪ BGE top-m (句ごと) → q2b採点判定
#      (query-level, 1doc=1判定, タスク別閾値τ)。calib_cache.jsonlのq2bスコアを再利用
#   4. flood-fill: member±wの隣人をq2b判定し新member枯渇まで反復 (職種ソート構造)
#      辞書由来memberは既にプール完結的なので、fillはjudge由来member種のみ (コスト制御)
#   5. コード集約 → answer F1 (token_f1, step20整合)
#
# 事前予測 (2026-07-07 登録, 課金前):
#   P1: train60 ALL answerF1 >= 35 (辞書32.9 + 未解決22.6%のjudge充填で sort/topk +5-10pt)
#   P2: countは<5のまま (個数完全一致の壁。次カード=個数較正で別途対処)
#   P3: judgeコール数は v2.5比で1/3以下 (辞書解決分はゼロコール)
#   P4 (リスク): judge充填はFPも足す。sort系でmember precision低下が純損になる問が出る
#
# 使い方:
#   python step34_v30.py --n 15                 (パイロット ~$1-2)
#   python step34_v30.py --n 60                 (本番 ~$3-5)
#   python step34_v30.py --n 60 --no-judge      (辞書のみ=step35と一致するか検算, 課金ゼロ)
#   python step34_v30.py --n 60 --no-fill       (flood-fill ablation)
# 注意: プール構成 (m/theta/tau) を変えたら results_v30_*.jsonl を退避 (運用ルール)
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
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--parsed", default="parsed2_train60.jsonl")
ap.add_argument("--theta", type=float, default=0.3)
ap.add_argument("--min-sup", type=int, default=3, dest="min_sup")
ap.add_argument("--m", type=int, default=30, help="句ごとBM25/BGE候補数")
ap.add_argument("--fill-w", type=int, default=3, dest="fill_w")
ap.add_argument("--fill-iters", type=int, default=2, dest="fill_iters")
ap.add_argument("--no-judge", action="store_true", dest="no_judge")
ap.add_argument("--no-fill", action="store_true", dest="no_fill")
args = ap.parse_args()

# タスク別閾値 (q2b較正, step33b。タスク別n=5なのでパイロットで再確認)
TAU = {"count": 6, "max_id": 6, "sort_desc": 6, "topk_largest": 6,
       "min_id": 5, "topk_smallest": 5, "sort_asc": 4}

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = sorted(d["id"] for d in docs)
id_set = set(doc_ids)
raw_text = {d["id"]: d["contents"] for d in docs}
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])
bm_ids = [d["id"] for d in docs]
bge = json.load(open("bge_ranks.json", encoding="utf-8")) if os.path.exists("bge_ranks.json") else {}

tr = json.load(open(f"{DATA}/train (1).json", encoding="utf-8"))
qtexts = [norm(t["question"]) for t in tr]
golds = [set(t["golden_doc_ids"]) for t in tr]
q_by_text = {qt: i for i, qt in enumerate(qtexts)}

# ---- 辞書 (step35と同一ロジック) ----
def support_idxs(p, excl):
    p = norm(p)
    return [i for i, qt in enumerate(qtexts) if p in qt and i not in excl]

def freq_map(idxs):
    cnt = collections.Counter()
    for i in idxs: cnt.update(golds[i])
    return {d: c / len(idxs) for d, c in cnt.items()}

def trim_variants(ph):
    p = norm(ph); outs = [p]
    for sep in (" with ", " and ", " or ", ", ", " in ", " for "):
        if sep in p: outs.append(p.split(sep)[0])
    return sorted(set(outs), key=len, reverse=True)

def resolve(ph, excl):
    for v in trim_variants(ph):
        idxs = support_idxs(v, excl)
        if len(idxs) >= args.min_sup:
            return freq_map(idxs)
    return None

# ---- q2b judge (step33bと同一プロンプト・同一PV → calib_cache再利用可) ----
SCORE_SYS = (
    "You rate document relevance for a corpus-level query over a resume corpus. "
    "Output an integer 0-10. Rubric: 9-10 = unambiguously one of the documents the "
    "query is asking about (matches the full profile/conditions described); 7-8 = very "
    "likely a match but one condition is implicit or only weakly evidenced; 4-6 = "
    "same occupation family or satisfies only part of the conditions; 2-3 = "
    "tangential overlap only; 0-1 = unrelated. Avoid defaulting to 0/5/10; first "
    "choose the band, then pick the exact integer within it to express confidence.")
S_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "score", "strict": True, "schema": {
        "type": "object", "properties": {"score": {"type": "integer"}},
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
CACHE = "calib_cache.jsonl"   # q2bスコアはPV+idx+docでキー共有 → 較正分を再利用
cache = {}
if os.path.exists(CACHE):
    for l in open(CACHE, encoding="utf-8"):
        if l.strip():
            j = json.loads(l)
            if "s" in j:
                cache[(j["pv"], j["idx"], j["d"])] = j["s"]
cf = open(CACHE, "a", encoding="utf-8")
n_calls = 0

def score(idx, question, d):
    global n_calls
    k = (judge.pv, idx, d)
    if k not in cache:
        if args.no_judge:
            return None
        s = judge.call(question, raw_text[d][:24000])
        n_calls += 1
        cache[k] = s
        cf.write(json.dumps({"pv": judge.pv, "idx": idx, "d": d, "s": s},
                            ensure_ascii=False) + "\n")
        cf.flush()
    return cache[k]

def token_f1(pred, gold):
    P = re.findall(r"[a-z0-9]+", str(pred).lower())
    G = re.findall(r"[a-z0-9]+", str(gold).lower())
    if not P or not G: return float(P == G)
    common = collections.Counter(P) & collections.Counter(G)
    o = sum(common.values())
    if o == 0: return 0.0
    pr, rc = o / len(P), o / len(G)
    return 2 * pr * rc / (pr + rc)

def aggregate(agg, k, members):
    m = sorted(members)
    if not m: return ""
    return {"count": str(len(m)), "min_id": str(m[0]), "max_id": str(m[-1]),
            "sort_asc": ", ".join(map(str, m)),
            "sort_desc": ", ".join(map(str, m[::-1])),
            "topk_smallest": ", ".join(map(str, m[:k or 0])),
            "topk_largest": ", ".join(map(str, sorted(members, reverse=True)[:k or 0]))
            }[agg]

# ---- 本体 ----
parsed = [json.loads(l) for l in open(args.parsed, encoding="utf-8") if l.strip()][:args.n]
OUT = f"results_v30_{judge.pv}_t{args.theta}_m{args.m}.jsonl"
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["idx"] for l in open(OUT, encoding="utf-8") if l.strip()}
of = open(OUT, "a", encoding="utf-8")

for p in parsed:
    if p["idx"] in done:
        continue
    q = p["question"]; task = p["parsed"]["agg"]; tau = TAU[task]
    i = q_by_text.get(norm(q)); excl = {i} if i is not None else set()
    gold = set(p["golden_doc_ids"])

    # 第一層: 辞書
    member = set(); unresolved = []
    for cl in p["parsed"]["clauses"]:
        fms = [resolve(ph, excl) for ph in cl]
        if any(f is None for f in fms):
            unresolved.append(cl)
            continue
        cand = set.intersection(*[set(f) for f in fms]) if fms else set()
        member |= {d for d in cand if min(f.get(d, 0) for f in fms) >= args.theta}

    # 第二層: 未解決クローズをjudge充填 (BM25∪BGE候補 → q2b >= τ)
    judged_members = set()
    for cl in unresolved:
        cands = set()
        for ph in cl:
            sc = bm25.get_scores(tok(ph))
            cands |= {bm_ids[j] for j in np.argsort(sc)[::-1][:args.m]}
            if ph in bge:
                cands |= set(bge[ph][:args.m])
        for d in sorted(cands - member):
            s = score(p["idx"], q, d)
            if s is not None and s >= tau:
                judged_members.add(d)
    member |= judged_members

    # 第三層: flood-fill (judge由来member種のみ, ±w, 反復)
    if not args.no_fill and judged_members:
        frontier = set(judged_members)
        for _ in range(args.fill_iters):
            neigh = set()
            for d in frontier:
                neigh |= {d + o for o in range(-args.fill_w, args.fill_w + 1)}
            neigh = (neigh & id_set) - member
            new = set()
            for d in sorted(neigh):
                s = score(p["idx"], q, d)
                if s is not None and s >= tau:
                    new.add(d)
            member |= new
            frontier = new
            if not new:
                break

    pred = aggregate(task, p["parsed"].get("k"), member)
    af = token_f1(pred, p["answer"])
    mf = 2 * len(member & gold) / max(len(member) + len(gold), 1)
    rec = {"idx": p["idx"], "task": task, "answer_f1": af, "member_f1": mf,
           "n_member": len(member), "n_gold": len(gold),
           "n_unresolved": len(unresolved), "n_judged": len(judged_members),
           "pred": pred, "gold_answer": p["answer"]}
    of.write(json.dumps(rec, ensure_ascii=False) + "\n"); of.flush()
    print(f"[idx {p['idx']:>5}] {task:13s} aF={af:.2f} mF={mf:.2f} "
          f"|member|={len(member):>3} (辞書{len(member)-len(judged_members)}"
          f"+judge{len(judged_members)}) 未解決句{len(unresolved)} "
          f"累計コール{n_calls}", flush=True)

# ---- 集計 ----
res = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
by = collections.defaultdict(list)
for r in res: by[r["task"]].append(r)
print(f"\n=== v3.0 ({judge.pv}, θ={args.theta}, τ={TAU}, n={len(res)}) ===")
print(f"{'task':>13} | {'n':>2} | {'memberF1':>8} | {'answerF1':>8}")
for t, rs in sorted(by.items()):
    print(f"{t:>13} | {len(rs):>2} | {np.mean([r['member_f1'] for r in rs])*100:7.1f}% "
          f"| {np.mean([r['answer_f1'] for r in rs])*100:7.1f}%")
print(f"{'ALL':>13} | {len(res):>2} | "
      f"{np.mean([r['member_f1'] for r in res])*100:7.1f}% | "
      f"{np.mean([r['answer_f1'] for r in res])*100:7.1f}%")
print(f"judgeコール数 (今回新規): {n_calls}")
print("参考: 辞書のみ32.9 / v2.5=8.85 / MARAG-7B=28.60, 14B=31.22 (train60比較)")

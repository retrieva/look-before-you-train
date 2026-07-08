# step34d_v30d.py - v3.0d: 空クローズ規則の修正 (step34cの後継)
#   v30cの新故障 (idx549で確認): 「解決済みだが空」のクローズをjudgeに回した結果、
#   満点非gold (判定不能成分) がmax_idの答えを強奪 (1.00->0.00)。
#   空は情報である: 辞書が支持フレーズ付きで空を返す = 構築が何も通さなかった連言。
#   v30d規則:
#     - 真の未解決 (支持変種なし) のみ judgeフォールバック (2444救済は維持)
#     - 解決済みだが空 -> 何も足さない (v30bの挙動, 549復元)
#     - 最終member空 -> 全句から候補生成して採点上位を救済 (1759カバー)
#
# 事前予測 (2026-07-07 登録):
#   P1d: --n 15 で ALL >= 70 (549復元 + 2444救済維持)
#   P2d: --n 60 で ALL >= 48 (辞書のみ36.3 + judge充填の純増)
#   パイロット2 (--n 15, v30b) 裁定: ALL=60.6 (P1'>=45 クリア) / コール125 (P2' クリア)
#   / count 33.3 (P4'良い方向に外れ, idx1113個数ピタリ) / P3' idx1759失敗 (形が変化)
#   新故障と修正:
#   - 部分文字列マッチ汚染: 'sourcing'が"outsourcing"等にヒットし偽プール合成 (idx1759)
#     -> 単語境界マッチ化 (train60辞書のみ36.3で副作用なし検証済み)
#   - 空クローズの死: 解決したが∩θ=0 (idx1759) / judge全員τ8未満 (idx2444) -> aF自動0
#     -> 空クローズは未解決扱いでjudgeへ / 最終member空なら採点上位を採用 (絶対に空で答えない)
#   辞書のみベースライン更新: train60 = 36.3 (メタ語除去+プール上限+単語境界込み)
#
# 事前予測 (2026-07-07 登録):
#   P1'': --n 15 で ALL >= 62 (1759/2444の少なくとも片方が0から復帰)
#   P2'': 新規コール < 400 (空クローズ再分類でjudge対象が増える分)
#   P3'': --n 60 で ALL >= 45 (辞書36.3 + judge充填)
#   パイロット (--n 15, 2026-07-07) の教訓:
#   - 辞書完全解決4問は aF 1.00/1.00/0.87/1.00。辞書層は本物
#   - 故障1: flood-fill が低τ (min_id τ5, FPR82%) で暴発 (idx1976: judge131件→~530件)
#     旧ログの「辞書」欄はfill分を誤合算していた (表示バグも修正)
#   - 故障2: メタ語句 ("documents about X","include Y") が全句未解決 → judge丸投げで膨張
#     (idx1759)。メタ語プレフィックス除去で未解決68句中13句救済 (idx1759は全句解決)
#   - 「変種2語以上」ガードは逆効果 (32.9→27.0) で不採用。「プール>60棄却」のみ採用
# 修正点: メタ語除去 / judgeフォールバック τ_fb=8+クローズ上位25件 / fillデフォルトOFF
#         (--fillで有効, 閾値8) / ログ集計修正 / 結果ファイル v30b_*
#
# 事前予測 (2026-07-07 登録, 課金前):
#   P1': train60パイロット15問 ALL answerF1 >= 45 (辞書完全解決問は維持, FP洪水停止)
#   P2': 新規judgeコール < 300/15問 (旧2334から激減)
#   P3': idx1759 が辞書で解決され aF > 0
#   P4': countは依然 <5 (次カード: 個数較正で別途)
# 使い方:
#   python step34b_v30b.py --n 15            (パイロット再走。q2bキャッシュ再利用で安い)
#   python step34b_v30b.py --n 60
#   python step34b_v30b.py --n 60 --no-judge (辞書のみ検算, 課金ゼロ)
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
ap.add_argument("--pool-cap", type=int, default=60, dest="pool_cap")
ap.add_argument("--m", type=int, default=30)
ap.add_argument("--tau-fb", type=int, default=8, dest="tau_fb",
                help="judgeフォールバック閾値 (q2b較正: τ8でFPR~32%, τ9で~22%)")
ap.add_argument("--fb-cap", type=int, default=25, dest="fb_cap",
                help="クローズあたりjudge採用上限 (スコア降順)")
ap.add_argument("--fill", action="store_true", help="flood-fill有効化 (デフォルトOFF)")
ap.add_argument("--fill-w", type=int, default=3, dest="fill_w")
ap.add_argument("--fill-iters", type=int, default=2, dest="fill_iters")
ap.add_argument("--rescue-top", type=int, default=5, dest="rescue_top",
                help="member空のとき採点上位N件を採用 (絶対に空で答えない)")
ap.add_argument("--no-judge", action="store_true", dest="no_judge")
args = ap.parse_args()

TAU = {"count": 6, "max_id": 6, "sort_desc": 6, "topk_largest": 6,
       "min_id": 5, "topk_smallest": 5, "sort_asc": 4}   # 辞書欠落時の参考値

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
id_set = {d["id"] for d in docs}
raw_text = {d["id"]: d["contents"] for d in docs}
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])
bm_ids = [d["id"] for d in docs]
bge = json.load(open("bge_ranks.json", encoding="utf-8")) if os.path.exists("bge_ranks.json") else {}

tr = json.load(open(f"{DATA}/train (1).json", encoding="utf-8"))
qtexts = [norm(t["question"]) for t in tr]
golds = [set(t["golden_doc_ids"]) for t in tr]
q_by_text = {qt: i for i, qt in enumerate(qtexts)}

# ---- 辞書 (メタ語除去 + プール上限ガード) ----
META = ("documents about ", "document about ", "include ", "mention a ",
        "mention ", "related to ", "are about ", "candidates who are ",
        "an ", "a ")

def trim_variants(ph):
    p = norm(ph)
    outs = [p]
    for pre in META:
        if p.startswith(pre):
            outs.append(p[len(pre):])
    more = []
    for o in outs:
        for sep in (" with ", " and ", " or ", ", ", " in ", " for "):
            if sep in o:
                more.append(o.split(sep)[0])
    return sorted(set(outs + more), key=len, reverse=True)

def support_idxs(v, excl):
    pat = re.compile(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])")
    return [i for i, qt in enumerate(qtexts) if pat.search(qt) and i not in excl]

def resolve(ph, excl):
    for v in trim_variants(ph):
        idxs = support_idxs(v, excl)
        if len(idxs) < args.min_sup:
            continue
        cnt = collections.Counter()
        for i in idxs:
            cnt.update(golds[i])
        fm = {d: c / len(idxs) for d, c in cnt.items()}
        if sum(1 for f in fm.values() if f >= args.theta) > args.pool_cap:
            continue   # 汎用語の巨大プールは棄却し次の変種へ
        return fm
    return None

# ---- q2b judge (step33b/34と同一プロンプト・PV → キャッシュ共有) ----
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
CACHE = "calib_cache.jsonl"
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
    return 2 * o / (len(P) + len(G))

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
OUT = f"results_v30d_{judge.pv}_t{args.theta}_fb{args.tau_fb}.jsonl"
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["idx"] for l in open(OUT, encoding="utf-8") if l.strip()}
of = open(OUT, "a", encoding="utf-8")

for p in parsed:
    if p["idx"] in done:
        continue
    q = p["question"]; task = p["parsed"]["agg"]
    i = q_by_text.get(norm(q)); excl = {i} if i is not None else set()
    gold = set(p["golden_doc_ids"])

    # 第一層: 辞書
    dict_members = set(); unresolved = []
    for cl in p["parsed"]["clauses"]:
        fms = [resolve(ph, excl) for ph in cl]
        if any(f is None for f in fms):
            unresolved.append(cl)
            continue
        cand = set.intersection(*[set(f) for f in fms]) if fms else set()
        mm = {d for d in cand if min(f.get(d, 0) for f in fms) >= args.theta}
        # 解決済みだが空 -> 空は情報。何も足さずjudgeにも回さない (v30d)
        dict_members |= mm

    # 第二層: 未解決クローズのみ judge充填 (τ_fb=8, クローズ上位fb_cap件)
    judge_members = set(); all_scored = []
    for cl in unresolved:
        cands = set()
        for ph in cl:
            sc = bm25.get_scores(tok(ph))
            cands |= {bm_ids[j] for j in np.argsort(sc)[::-1][:args.m]}
            if ph in bge:
                cands |= set(bge[ph][:args.m])
        scored = []
        for d in sorted(cands - dict_members):
            s = score(p["idx"], q, d)
            if s is not None:
                scored.append((s, d))
        scored.sort(reverse=True)
        judge_members |= {d for s, d in scored[:args.fb_cap] if s >= args.tau_fb}
        all_scored += scored

    member = dict_members | judge_members

    # 絶対に空で答えない: member空なら全句から候補を生成して採点上位を採用 (τ_fb不問)
    rescued = set()
    if not member:
        if not all_scored:
            cands = set()
            for cl in p["parsed"]["clauses"]:
                for ph in cl:
                    sc = bm25.get_scores(tok(ph))
                    cands |= {bm_ids[j] for j in np.argsort(sc)[::-1][:args.m]}
                    if ph in bge:
                        cands |= set(bge[ph][:args.m])
            for d in sorted(cands):
                s = score(p["idx"], q, d)
                if s is not None:
                    all_scored.append((s, d))
        if all_scored:
            all_scored.sort(reverse=True)
            n_take = p["parsed"].get("k") or args.rescue_top
            rescued = {d for _, d in all_scored[:n_take]}
            member |= rescued

    # 第三層: flood-fill (オプトイン, 閾値はmax(τ_fb,8))
    fill_members = set()
    if args.fill and judge_members:
        thr = max(args.tau_fb, 8)
        frontier = set(judge_members)
        for _ in range(args.fill_iters):
            neigh = set()
            for d in frontier:
                neigh |= {d + o for o in range(-args.fill_w, args.fill_w + 1)}
            neigh = (neigh & id_set) - member
            new = set()
            for d in sorted(neigh):
                s = score(p["idx"], q, d)
                if s is not None and s >= thr:
                    new.add(d)
            member |= new; fill_members |= new; frontier = new
            if not new:
                break

    pred = aggregate(task, p["parsed"].get("k"), member)
    af = token_f1(pred, p["answer"])
    mf = 2 * len(member & gold) / max(len(member) + len(gold), 1)
    rec = {"idx": p["idx"], "task": task, "answer_f1": af, "member_f1": mf,
           "n_member": len(member), "n_gold": len(gold),
           "n_dict": len(dict_members), "n_judge": len(judge_members),
           "n_fill": len(fill_members), "n_rescued": len(rescued),
           "n_unresolved": len(unresolved),
           "pred": pred, "gold_answer": p["answer"]}
    of.write(json.dumps(rec, ensure_ascii=False) + "\n"); of.flush()
    print(f"[idx {p['idx']:>5}] {task:13s} aF={af:.2f} mF={mf:.2f} "
          f"|member|={len(member):>3} (辞書{len(dict_members)}/judge{len(judge_members)}"
          f"/fill{len(fill_members)}/救済{len(rescued)}) 未解決句{len(unresolved)} "
          f"累計コール{n_calls}",
          flush=True)

res = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
by = collections.defaultdict(list)
for r in res: by[r["task"]].append(r)
print(f"\n=== v3.0d ({judge.pv}, θ={args.theta}, τ_fb={args.tau_fb}, "
      f"fill={'ON' if args.fill else 'OFF'}, n={len(res)}) ===")
print(f"{'task':>13} | {'n':>2} | {'memberF1':>8} | {'answerF1':>8}")
for t, rs in sorted(by.items()):
    print(f"{t:>13} | {len(rs):>2} | {np.mean([r['member_f1'] for r in rs])*100:7.1f}% "
          f"| {np.mean([r['answer_f1'] for r in rs])*100:7.1f}%")
print(f"{'ALL':>13} | {len(res):>2} | "
      f"{np.mean([r['member_f1'] for r in res])*100:7.1f}% | "
      f"{np.mean([r['answer_f1'] for r in res])*100:7.1f}%")
print(f"新規judgeコール: {n_calls}")
print("参考: 辞書のみ36.3 / v2.5=8.85 / MARAG-7B=28.60, 14B=31.22")

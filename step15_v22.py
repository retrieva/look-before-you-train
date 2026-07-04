# step15_v22.py - CORE v2.2a: recall-triage + full-text strict verification
#                  + aggregation-aware verification budget (min/max/topkは早期打ち切り)
# 設計根拠:
#   - GlobalQAのgoldは「keyword/semantic検索候補 + DeepSeekの全コーパス判定」で構築
#     (arXiv:2510.26205 §3.2)。判定基準 = 中立・厳格なフルテキストLLM判定の再現
#   - step14: 字句カバレッジではgoldを近似不可 (τ=1.0でもsetP=0.18) -> 意味判定必須
#   - step11/13: 緩和プロンプトはFP epidemic。gold集合は常に<=50件 (train60中央値16)
#   - v2.2a: min/max/topk は該当端から歩き、必要数確定で打ち切り (HANDOFF §3 の
#     aggregation-aware selective verification の復活)。count/sort のみ全件検証。
#     プロンプト不変・制御フローのみの変更なので PV=v22a のまま既存キャッシュを全額再利用
# 三段構成:
#   1) 候補: dense top-N ∪ BM25 top-N (N=200, union recall 96.8%)
#   2) トリアージ (リコール志向): プロファイルを yes/maybe/no の3値で判定。
#      全(profile,predicate)ペアの列挙を強制 (挙げ忘れ=FN の故障モードを排除)。noのみ落とす
#   3) 最終判定 (精度担当): yes/maybe を原文全体で厳格判定 + 証拠の逐語引用を
#      コード側で部分文字列検証。引用が実在しなければtrue無効
#   4) 集約: コード。memberは検証済みなので min/max/topk の別confirm段は廃止
# 使い方:
#   python step15_v22.py --split train60 --limit 15    # パイロット (目安 $3-5)
#   python step15_v22.py --split train60               # 全60問
# 冪等: triage_cache / verify_cache / results_v22 すべてチェックポイント式
# キャッシュキーはプロンプト版 PV を含む。プロンプトを変えたら PV を必ず上げること
import json, os, re, argparse, collections, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
from util import with_retry
from step3_eval import pred_scores, token_f1

PV = "v22a"          # プロンプト版 (キャッシュキーの一部)
client = OpenAI(timeout=180)
DATA = "./data/GlobalQA"
MODEL = "gpt-5-mini"
TRIAGE_CACHE = "triage_cache.jsonl"
VERIFY_CACHE = "verify_cache.jsonl"

# ---------------- データ読み込み ----------------
docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
raw_text = {d["id"]: d["contents"] for d in docs}

profiles = {}
for l in open("profiles.jsonl", encoding="utf-8"):
    if l.strip():
        p = json.loads(l)
        profiles[p["id"]] = p

def profile_text(pid, max_chars=1800):
    p = profiles[pid]
    parts = [
        "TITLES: " + "; ".join(p["titles"][:12]),
        "SKILLS: " + "; ".join(p["skills"][:25]),
        "TOOLS: " + "; ".join(p["tools"][:25]),
        "DOMAINS: " + "; ".join(p["domains"][:10]),
    ]
    if p.get("certifications"):
        parts.append("CERTS: " + "; ".join(p["certifications"][:8]))
    if p.get("years_experience") is not None:
        parts.append(f"YEARS: {p['years_experience']}")
    if p.get("notable"):
        parts.append("NOTABLE: " + "; ".join(p["notable"][:10]))
    parts.append("SUMMARY: " + p.get("summary", ""))
    return "\n".join(parts)[:max_chars]

# ---------------- 候補生成 (dense ∪ BM25) ----------------
def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
print("BM25インデックス構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

_bm, _dn = {}, {}
def bm25_top(pred, n):
    if pred not in _bm:
        _bm[pred] = list(np.argsort(bm25.get_scores(tok(pred)))[::-1])
    return [doc_ids[i] for i in _bm[pred][:n]]

def dense_top(pred, n):
    if pred not in _dn:
        s = pred_scores(pred)
        _dn[pred] = sorted(s, key=s.get, reverse=True)
    return _dn[pred][:n]

# ---------------- キャッシュ (キー: PV, 述語, doc) ----------------
def _load(path):
    c = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            if l.strip():
                try:
                    r = json.loads(l)
                    c[(r["pv"], r["p"], r["d"])] = r
                except (json.JSONDecodeError, KeyError):
                    pass
    return c

triage = _load(TRIAGE_CACHE)   # r["v"]: "yes"/"maybe"/"no"
verify = _load(VERIFY_CACHE)   # r["v"]: bool(検証済), r["raw"]: bool(LLM出力), r["ev"]: 引用
_tf = open(TRIAGE_CACHE, "a", encoding="utf-8")
_vf = open(VERIFY_CACHE, "a", encoding="utf-8")
_lock = threading.Lock()

def _write(f, rec):
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()

# ---------------- 2) トリアージ (yes/maybe/no, 全ペア列挙強制) ----------------
TRIAGE_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "triage", "strict": True, "schema": {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {
            "type": "object",
            "properties": {"profile": {"type": "integer"},
                           "predicate": {"type": "integer"},
                           "status": {"type": "string",
                                      "enum": ["yes", "maybe", "no"]}},
            "required": ["profile", "predicate", "status"],
            "additionalProperties": False}}},
        "required": ["results"], "additionalProperties": False}}}

TRIAGE_SYS = (
    "You triage resume profiles against predicates. For EVERY (profile, predicate) "
    "pair, output exactly one entry with a status: 'yes' if the profile clearly "
    "satisfies the predicate, 'maybe' if it plausibly might but the profile summary "
    "is not enough to be sure, 'no' only if it clearly does not. Profiles are "
    "labeled [PROFILE 0], [PROFILE 1], ... and predicates [0], [1], ...; refer to "
    "them ONLY by these bracketed index numbers. Output one entry per pair; do not "
    "skip any pair.")

def triage_batch(preds, batch_ids, stats):
    plist = "\n".join(f"[{i}] {p}" for i, p in enumerate(preds))
    body = "\n\n".join(f"[PROFILE {i}]\n{profile_text(d)}"
                       for i, d in enumerate(batch_ids))
    r = with_retry(lambda: client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": TRIAGE_SYS},
                  {"role": "user", "content":
                   f"PREDICATES:\n{plist}\n\nPROFILES:\n{body}"}],
        response_format=TRIAGE_SCHEMA))
    got = {}
    for it in json.loads(r.choices[0].message.content)["results"]:
        pos, pi = it["profile"], it["predicate"]
        if 0 <= pos < len(batch_ids) and 0 <= pi < len(preds):
            got[(pi, batch_ids[pos])] = it["status"]
    with _lock:
        stats["triage_calls"] += 1
        for d in batch_ids:
            for i, p in enumerate(preds):
                v = got.get((i, d), "maybe")   # 欠落はリコール側に倒す
                if (i, d) not in got:
                    stats["triage_missing"] += 1
                triage[(PV, p, d)] = {"v": v}
                _write(_tf, {"pv": PV, "p": p, "d": d, "v": v})

# ---------------- 3) フルテキスト厳格判定 (証拠引用検証) ----------------
VERIFY_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "verify", "strict": True, "schema": {
        "type": "object",
        "properties": {"judgments": {"type": "array", "items": {
            "type": "object",
            "properties": {"condition": {"type": "integer"},
                           "satisfies": {"type": "boolean"},
                           "evidence": {"type": "string"}},
            "required": ["condition", "satisfies", "evidence"],
            "additionalProperties": False}}},
        "required": ["judgments"], "additionalProperties": False}}}

VERIFY_SYS = (
    "You validate whether a resume satisfies given conditions. Judge strictly based "
    "on what the resume explicitly states or directly demonstrates; do not credit "
    "loose thematic similarity or generic plausibility. For each condition, output "
    "satisfies true/false. If true, copy a short verbatim evidence snippet EXACTLY "
    "as it appears in the resume (this will be string-matched against the resume; "
    "any paraphrase will invalidate the judgment). If false, use an empty string.")

def _norm(s):
    return re.sub(r"\s+", " ", s.lower())

def verify_doc(preds, need_idx, d, stats):
    """docの原文全体で、必要な述語だけ一括厳格判定。キャッシュ済みはスキップ"""
    todo = [i for i in need_idx if (PV, preds[i], d) not in verify]
    if todo:
        clist = "\n".join(f"[{j}] {preds[i]}" for j, i in enumerate(todo))
        r = with_retry(lambda: client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": VERIFY_SYS},
                      {"role": "user", "content":
                       f"CONDITIONS:\n{clist}\n\nRESUME:\n{raw_text[d][:24000]}"}],
            response_format=VERIFY_SCHEMA))
        got = {}
        for j in json.loads(r.choices[0].message.content)["judgments"]:
            if 0 <= j["condition"] < len(todo):
                got[todo[j["condition"]]] = j
        nt = _norm(raw_text[d])
        with _lock:
            stats["verify_calls"] += 1
            for i in todo:
                j = got.get(i, {"satisfies": False, "evidence": ""})
                raw_v = bool(j["satisfies"])
                ev = j.get("evidence", "")
                v = raw_v and bool(ev.strip()) and _norm(ev) in nt
                if raw_v and not v:
                    stats["ev_invalidated"] += 1
                verify[(PV, preds[i], d)] = {"v": v, "raw": raw_v, "ev": ev}
                _write(_vf, {"pv": PV, "p": preds[i], "d": d,
                             "v": v, "raw": raw_v, "ev": ev})
    return {i: verify[(PV, preds[i], d)]["v"] for i in need_idx}

# ---------------- クエリ実行 ----------------
def _is_member(preds, combine, advance, d, stats):
    vres = verify_doc(preds, advance[d], d, stats)
    if combine == "or":
        return any(vres.values())
    return all(vres.get(i, False) for i in range(len(preds)))

def run_query(r, args, stats):
    p = r["parsed"]
    preds, combine, op = p["predicates"], p["combine"], p["agg"]
    k = p.get("k") or 1
    cands = set()
    for pr in preds:
        cands.update(dense_top(pr, args.n))
        cands.update(bm25_top(pr, args.n))

    # --- トリアージ ---
    todo = [d for d in cands if d in profiles
            and any((PV, pr, d) not in triage for pr in preds)]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    if batches:
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda b: triage_batch(preds, b, stats), batches))

    def status(pr, d):
        rec = triage.get((PV, pr, d))
        return rec["v"] if rec else "maybe"   # profile欠損等はリコール側に倒す

    # --- 前進集合の決定 (verify対象) ---
    advance = {}   # d -> 検証すべき述語indexリスト
    for d in cands:
        st = [status(pr, d) for pr in preds]
        if combine == "or":
            need = [i for i, s in enumerate(st) if s != "no"]
            if need:
                advance[d] = need
        else:
            if all(s != "no" for s in st):
                advance[d] = list(range(len(preds)))
    stats["n_advance"] = len(advance)

    # --- 極値系: 該当端から歩いて必要数で打ち切り (aggregation-aware) ---
    if op in ("min_id", "max_id", "topk_smallest", "topk_largest"):
        want = 1 if op in ("min_id", "max_id") else k
        seq = sorted(advance)
        if op in ("max_id", "topk_largest"):
            seq = seq[::-1]
        found = []
        CH = 8
        for i in range(0, len(seq), CH):
            chunk = seq[i:i + CH]
            res = {}
            def h(d):
                res[d] = _is_member(preds, combine, advance, d, stats)
            with ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(h, chunk))
            for d in chunk:                    # 端からの順序で採用
                if res.get(d):
                    found.append(d)
            if len(found) >= want:
                break
        found = found[:want]
        stats["n_member"] = len(found)         # 部分探索: 確定分のみ
        if op in ("min_id", "max_id"):
            return str(found[0]) if found else ""
        return ", ".join(map(str, found))

    # --- count/sort: 全件検証が不可避。コストガードはここにのみ適用 ---
    if len(advance) > args.max_advance:
        raise RuntimeError(
            f"advance={len(advance)} > --max-advance {args.max_advance}: "
            f"count/sortの全件検証コストが上限超過。--max-advanceを上げるか要相談")
    member = set()
    def handle(item):
        d, need = item
        if _is_member(preds, combine, advance, d, stats):
            with _lock:
                member.add(d)
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(handle, list(advance.items())))

    ids = sorted(member)
    stats["n_member"] = len(ids)
    if len(ids) > 50:
        print(f"    [警告] |member|={len(ids)} > 50 (goldは常に<=50。判定が緩い疑い)")
    if op == "count":
        return str(len(ids))
    if op == "sort_asc":
        return ", ".join(map(str, ids))
    return ", ".join(map(str, ids[::-1]))    # sort_desc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train60", choices=["train60", "test100"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=10, help="トリアージのprofile数/呼び出し")
    ap.add_argument("--max-advance", type=int, default=1200, dest="max_advance",
                    help="count/sortの全件検証がこれを超えたら中断 (コスト暴走防止)")
    ap.add_argument("--limit", type=int, default=0, help="先頭N問のみ(パイロット用)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(f"parsed_{args.split}.jsonl", encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs("results_v22", exist_ok=True)
    out_path = f"results_v22/v22_{args.split}.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["qid"] for l in open(out_path, encoding="utf-8")
                if l.strip()}

    with open(out_path, "a", encoding="utf-8") as out:
        for r in rows:
            qid = r["idx"]
            if qid in done:
                continue
            stats = collections.Counter()
            pred = run_query(r, args, stats)
            f1 = token_f1(pred, r["answer"])
            rec = {"qid": qid, "op": r["parsed"]["agg"], "pred": pred,
                   "gold": r["answer"], "f1": f1,
                   "triage_calls": stats["triage_calls"],
                   "verify_calls": stats["verify_calls"],
                   "n_advance": stats["n_advance"],
                   "n_member": stats["n_member"],
                   "ev_invalidated": stats["ev_invalidated"],
                   "triage_missing": stats["triage_missing"]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[qid {qid}] {rec['op']:14s} f1={f1:.2f}"
                  f"  adv={rec['n_advance']} mem={rec['n_member']}"
                  f"  triage={rec['triage_calls']} verify={rec['verify_calls']}"
                  f"  ev_ng={rec['ev_invalidated']}", flush=True)

    res = collections.defaultdict(list)
    for l in open(out_path, encoding="utf-8"):
        if l.strip():
            rec = json.loads(l)
            res[rec["op"]].append(rec["f1"])
            res["_all"].append(rec["f1"])
    print(f"\n=== v2.2 {args.split} (n={len(res['_all'])}) ===")
    print(f"Answer F1: {np.mean(res['_all'])*100:.2f}"
          f"   (v2:10.92pilot / v1:4.77 / v0:1.80 / lexical:5.40 / SOTA:6.63)")
    for kk, v in sorted(res.items()):
        if not kk.startswith("_"):
            print(f"  {kk:14s} {np.mean(v)*100:6.1f} (n={len(v)})")

if __name__ == "__main__":
    main()

# step22_v24.py - CORE v2.4: BGE-aligned rank-bounded membership + 交換可能judge
#   v2.3からの変更:
#   - 順位束縛プールを BGE (構築と同じ検索器, step21のbge_ranks.json) ∪ BM25 に変更
#     (--retriever openai で旧 text-embedding-3-small に戻せる)
#   - 検証judgeを --judge gpt (gpt-5-mini, PV=v22a, キャッシュ再利用) /
#     --judge deepseek (deepseek-chat=V3系, PV=dsv3a, gold構築と同系のjudge) で切替
#   - それ以外 (厳格プロンプト・証拠引用の部分文字列検証・極値の早期打ち切り) は同一
# 事前: --judge deepseek には export DEEPSEEK_API_KEY=... が必要
# 使い方:
#   python step22_v24.py --split train60 --limit 15 --retriever bge --judge deepseek --m 30
import json, os, re, argparse, collections, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
from util import with_retry
from step3_eval import pred_scores, token_f1

DATA = "./data/GlobalQA"

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
raw_text = {d["id"]: d["contents"] for d in docs}

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
print("BM25インデックス構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

_bm = {}
def bm25_top(pred, n):
    if pred not in _bm:
        _bm[pred] = [doc_ids[i] for i in
                     np.argsort(bm25.get_scores(tok(pred)))[::-1]]
    return _bm[pred][:n]

_bge = None
_dn = {}
def retr_top(pred, n, retriever):
    global _bge
    if retriever == "bge":
        if _bge is None:
            _bge = json.load(open("bge_ranks.json", encoding="utf-8"))
        return _bge[pred][:n]
    if pred not in _dn:
        s = pred_scores(pred)
        _dn[pred] = sorted(s, key=s.get, reverse=True)
    return _dn[pred][:n]

# ---------------- judge抽象化 ----------------
VERIFY_SYS = (
    "You validate whether a resume satisfies given conditions. Judge strictly based "
    "on what the resume explicitly states or directly demonstrates; do not credit "
    "loose thematic similarity or generic plausibility. For each condition, output "
    "satisfies true/false. If true, copy a short verbatim evidence snippet EXACTLY "
    "as it appears in the resume (this will be string-matched against the resume; "
    "any paraphrase will invalidate the judgment). If false, use an empty string.")

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

DS_JSON_SPEC = (
    ' Respond ONLY with a JSON object of the form '
    '{"judgments": [{"condition": <int>, "satisfies": <true/false>, '
    '"evidence": "<verbatim snippet or empty string>"}]} with one entry per condition.')

class Judge:
    def __init__(self, kind, ds_model="deepseek-v4-flash"):
        self.kind = kind
        if kind == "deepseek":
            self.pv = "ds-" + ds_model      # モデル毎にキャッシュを分離
            self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                                 base_url="https://api.deepseek.com", timeout=180)
            self.model = ds_model
        else:
            self.pv = "v22a"           # v2.2a/v2.3のキャッシュを再利用
            self.client = OpenAI(timeout=180)
            self.model = "gpt-5-mini"

    def call(self, clist, text):
        if self.kind == "deepseek":
            r = with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": VERIFY_SYS + DS_JSON_SPEC},
                          {"role": "user", "content":
                           f"CONDITIONS:\n{clist}\n\nRESUME:\n{text}"}],
                response_format={"type": "json_object"}))
        else:
            r = with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": VERIFY_SYS},
                          {"role": "user", "content":
                           f"CONDITIONS:\n{clist}\n\nRESUME:\n{text}"}],
                response_format=VERIFY_SCHEMA))
        c = r.choices[0].message.content
        c = re.sub(r"^```(json)?|```$", "", c.strip(), flags=re.M)
        return json.loads(c)["judgments"]

# ---------------- verifyキャッシュ (キー: pv, 述語, doc) ----------------
VERIFY_CACHE = "verify_cache.jsonl"
verify = {}
if os.path.exists(VERIFY_CACHE):
    for l in open(VERIFY_CACHE, encoding="utf-8"):
        if l.strip():
            try:
                r = json.loads(l)
                verify[(r["pv"], r["p"], r["d"])] = r
            except (json.JSONDecodeError, KeyError):
                pass
_vf = open(VERIFY_CACHE, "a", encoding="utf-8")
_lock = threading.Lock()

def _norm(s):
    return re.sub(r"\s+", " ", s.lower())

def verify_doc(judge, preds, need_idx, d, stats):
    pv = judge.pv
    todo = [i for i in need_idx if (pv, preds[i], d) not in verify]
    if todo:
        clist = "\n".join(f"[{j}] {preds[i]}" for j, i in enumerate(todo))
        try:
            js = judge.call(clist, raw_text[d][:24000])
        except (json.JSONDecodeError, KeyError):
            js = []
            stats["parse_fail"] += 1
        got = {}
        for j in js:
            try:
                ci = int(j["condition"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= ci < len(todo):
                got[todo[ci]] = j
        nt = _norm(raw_text[d])
        with _lock:
            stats["verify_calls"] += 1
            for i in todo:
                j = got.get(i, {"satisfies": False, "evidence": ""})
                raw_v = bool(j.get("satisfies"))
                ev = str(j.get("evidence", "") or "")
                v = raw_v and bool(ev.strip()) and _norm(ev) in nt
                if raw_v and not v:
                    stats["ev_invalidated"] += 1
                verify[(pv, preds[i], d)] = {"v": v, "raw": raw_v, "ev": ev}
                _vf.write(json.dumps({"pv": pv, "p": preds[i], "d": d,
                                      "v": v, "raw": raw_v, "ev": ev},
                                     ensure_ascii=False) + "\n")
                _vf.flush()
    return {i: verify[(judge.pv, preds[i], d)]["v"] for i in need_idx}

# ---------------- クエリ実行 (v2.3と同一の制御フロー) ----------------
def run_query(r, args, judge, stats):
    p = r["parsed"]
    preds, combine, op = p["predicates"], p["combine"], p["agg"]
    k = p.get("k") or 1
    m = args.m
    pools = [set(bm25_top(pr, m)) | set(retr_top(pr, m, args.retriever))
             for pr in preds]
    cands = set().union(*pools)
    advance = {}
    for d in cands:
        if combine == "or":
            advance[d] = [i for i in range(len(preds)) if d in pools[i]]
        else:
            advance[d] = list(range(len(preds)))
    stats["n_advance"] = len(advance)

    def is_member(d):
        vres = verify_doc(judge, preds, advance[d], d, stats)
        if combine == "or":
            return any(vres.values())
        return all(vres.get(i, False) for i in range(len(preds)))

    if op in ("min_id", "max_id", "topk_smallest", "topk_largest"):
        want = 1 if op in ("min_id", "max_id") else k
        seq = sorted(advance)
        if op in ("max_id", "topk_largest"):
            seq = seq[::-1]
        found = []
        for i in range(0, len(seq), 8):
            chunk = seq[i:i + 8]
            res = {}
            def h(d):
                res[d] = is_member(d)
            with ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(h, chunk))
            for d in chunk:
                if res.get(d):
                    found.append(d)
            if len(found) >= want:
                break
        found = found[:want]
        stats["n_member"] = len(found)
        if op in ("min_id", "max_id"):
            return str(found[0]) if found else ""
        return ", ".join(map(str, found))

    member = set()
    def handle(d):
        if is_member(d):
            with _lock:
                member.add(d)
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(handle, list(advance)))
    ids = sorted(member)
    stats["n_member"] = len(ids)
    if len(ids) > 50:
        print(f"    [警告] |member|={len(ids)} > 50")
    if op == "count":
        return str(len(ids))
    if op == "sort_asc":
        return ", ".join(map(str, ids))
    return ", ".join(map(str, ids[::-1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train60", choices=["train60", "test100"])
    ap.add_argument("--m", type=int, default=30)
    ap.add_argument("--retriever", default="bge", choices=["bge", "openai"])
    ap.add_argument("--judge", default="gpt", choices=["gpt", "deepseek"])
    ap.add_argument("--ds-model", default="deepseek-v4-flash", dest="ds_model",
                    help="deepseek使用時のモデル (deepseek-v4-flash / deepseek-v4-pro)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    judge = Judge(args.judge, args.ds_model)

    rows = [json.loads(l) for l in
            open(f"parsed_{args.split}.jsonl", encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs("results_v24", exist_ok=True)
    out_path = (f"results_v24/v24_{args.split}_{args.retriever}"
                f"_{args.judge}_m{args.m}.jsonl")
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
            pred = run_query(r, args, judge, stats)
            f1 = token_f1(pred, r["answer"])
            rec = {"qid": qid, "op": r["parsed"]["agg"], "pred": pred,
                   "gold": r["answer"], "f1": f1,
                   "verify_calls": stats["verify_calls"],
                   "n_advance": stats["n_advance"],
                   "n_member": stats["n_member"],
                   "ev_invalidated": stats["ev_invalidated"],
                   "parse_fail": stats["parse_fail"]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[qid {qid}] {rec['op']:14s} f1={f1:.2f}"
                  f"  adv={rec['n_advance']} mem={rec['n_member']}"
                  f"  verify={rec['verify_calls']} ev_ng={rec['ev_invalidated']}",
                  flush=True)

    res = collections.defaultdict(list)
    for l in open(out_path, encoding="utf-8"):
        if l.strip():
            rec = json.loads(l)
            res[rec["op"]].append(rec["f1"])
            res["_all"].append(rec["f1"])
    print(f"\n=== v2.4 {args.split} {args.retriever}+{args.judge} m={args.m}"
          f" (n={len(res['_all'])}) ===")
    print(f"Answer F1: {np.mean(res['_all'])*100:.2f}"
          f"   (v2.3:6.43 / MARAG-R1 SOTA:26.4-31.22)")
    for kk, v in sorted(res.items()):
        if not kk.startswith("_"):
            print(f"  {kk:14s} {np.mean(v)*100:6.1f} (n={len(v)})")

if __name__ == "__main__":
    main()

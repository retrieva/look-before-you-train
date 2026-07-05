# step19_v23.py - CORE v2.3: rank-bounded membership + strict full-text verification
# 設計根拠 (step17/18 + DeepResearch構築調査で確定):
#   GlobalQAのgold = 「検索露出 (BGE top-20 x 最大10反復)」∩「DeepSeek-v3の判定」。
#   検索に浅く出ない文書は条件を満たしてもgoldにならない (H2確定)。
#   よって member = (いずれかの述語で BM25/dense 順位 top-m 以内) ∩ (厳格verify合格)。
#   - トリアージは廃止 (プールが小さく不要。FN源も消える)
#   - verifyプロンプトは v2.2a と同一 -> PV=v22a のまま既存キャッシュを全額再利用
#   - min/max/topk は該当端から早期打ち切り (aggregation-aware)
# 使い方:
#   python step19_v23.py --split train60 --limit 15 --m 50   # パイロット (大部分キャッシュ)
#   python step19_v23.py --split train60 --m 50              # 全60問
import json, os, re, argparse, collections, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
from util import with_retry
from step3_eval import pred_scores, token_f1

PV = "v22a"          # verifyプロンプトはv2.2aと同一 (変更したら必ず上げる)
client = OpenAI(timeout=180)
DATA = "./data/GlobalQA"
MODEL = "gpt-5-mini"
VERIFY_CACHE = "verify_cache.jsonl"

# ---------------- データ読み込み ----------------
docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
raw_text = {d["id"]: d["contents"] for d in docs}

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
print("BM25インデックス構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

_bm, _dn = {}, {}
def bm25_top(pred, n):
    if pred not in _bm:
        _bm[pred] = [doc_ids[i] for i in
                     np.argsort(bm25.get_scores(tok(pred)))[::-1]]
    return _bm[pred][:n]

def dense_top(pred, n):
    if pred not in _dn:
        s = pred_scores(pred)
        _dn[pred] = sorted(s, key=s.get, reverse=True)
    return _dn[pred][:n]

# ---------------- verifyキャッシュ ----------------
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

# ---------------- フルテキスト厳格判定 (v2.2aと同一プロンプト) ----------------
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
                _vf.write(json.dumps({"pv": PV, "p": preds[i], "d": d,
                                      "v": v, "raw": raw_v, "ev": ev},
                                     ensure_ascii=False) + "\n")
                _vf.flush()
    return {i: verify[(PV, preds[i], d)]["v"] for i in need_idx}

# ---------------- クエリ実行 ----------------
def run_query(r, args, stats):
    p = r["parsed"]
    preds, combine, op = p["predicates"], p["combine"], p["agg"]
    k = p.get("k") or 1
    m = args.m

    # 順位束縛プール: 述語ごとに BM25 top-m ∪ dense top-m
    pools = [set(bm25_top(pr, m)) | set(dense_top(pr, m)) for pr in preds]
    cands = set().union(*pools)
    advance = {}   # d -> 検証すべき述語index
    for d in cands:
        if combine == "or":
            advance[d] = [i for i in range(len(preds)) if d in pools[i]]
        else:
            advance[d] = list(range(len(preds)))
    stats["n_advance"] = len(advance)

    def is_member(d):
        vres = verify_doc(preds, advance[d], d, stats)
        if combine == "or":
            return any(vres.values())
        return all(vres.get(i, False) for i in range(len(preds)))

    # 極値系: 該当端から早期打ち切り
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

    # count/sort: プール全件検証
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
        print(f"    [警告] |member|={len(ids)} > 50 (goldは常に<=50)")
    if op == "count":
        return str(len(ids))
    if op == "sort_asc":
        return ", ".join(map(str, ids))
    return ", ".join(map(str, ids[::-1]))    # sort_desc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train60", choices=["train60", "test100"])
    ap.add_argument("--m", type=int, default=50, help="順位束縛の深さ (step18で較正)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(f"parsed_{args.split}.jsonl", encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs("results_v23", exist_ok=True)
    out_path = f"results_v23/v23_{args.split}_m{args.m}.jsonl"
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
                   "verify_calls": stats["verify_calls"],
                   "n_advance": stats["n_advance"],
                   "n_member": stats["n_member"],
                   "ev_invalidated": stats["ev_invalidated"]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[qid {qid}] {rec['op']:14s} f1={f1:.2f}"
                  f"  adv={rec['n_advance']} mem={rec['n_member']}"
                  f"  verify={rec['verify_calls']} (新規呼び出しのみ)", flush=True)

    res = collections.defaultdict(list)
    for l in open(out_path, encoding="utf-8"):
        if l.strip():
            rec = json.loads(l)
            res[rec["op"]].append(rec["f1"])
            res["_all"].append(rec["f1"])
    print(f"\n=== v2.3 {args.split} m={args.m} (n={len(res['_all'])}) ===")
    print(f"Answer F1: {np.mean(res['_all'])*100:.2f}"
          f"   (v2.2a部分:? / v2:10.92pilot / v0:1.80 / lexical:5.40 / SOTA:6.63)")
    for kk, v in sorted(res.items()):
        if not kk.startswith("_"):
            print(f"  {kk:14s} {np.mean(v)*100:6.1f} (n={len(v)})")

if __name__ == "__main__":
    main()

# step31_judgeprobe.py - H4の超小型プローブ (~$0.5): 極値系でgold答えdocがverifyで
#   死んだ事例を、2つの代替判定で再判定し、FNがどれだけ翻るかを測る。
#   H4: ベンチのgold規則は述語論理の厳格充足ではなく、eq.(7)流の
#       「クエリに答える情報を含むか」という文書レベルの緩い関連判定である。
#   変種A (lenient-pred): 述語判定のまま、意味的に妥当な合致を認める緩和プロンプト
#   変種B (query-level):  元の質問文そのものに対する関連判定 (eq.(7)の流儀)
#   注意: 本プローブはFN回復量のみを測る。FP増加コストは次段の較正ラウンドで測る。
#   結果は probe31.jsonl に保存 (冪等)。verify_cache.jsonl は汚さない。
# 使い方: python step31_judgeprobe.py --m 30
import json, os, re, argparse, collections
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
from util import with_retry

DATA = "./data/GlobalQA"
PV = "v22a"
MODEL = "gpt-5-mini"

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train60")
ap.add_argument("--m", type=int, default=30)
args = ap.parse_args()
M = args.m

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
raw_text = {d["id"]: d["contents"] for d in docs}
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])
bge = json.load(open("bge_ranks.json", encoding="utf-8"))

parsed = {}
for l in open(f"parsed2_{args.split}.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        parsed[r["idx"]] = r

recs = [json.loads(l) for l in
        open(f"results_v25/v25_{args.split}_gpt_m{M}.jsonl", encoding="utf-8")
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

_pl = {}
def poolc(p):
    if p not in _pl:
        br = [doc_ids[i] for i in np.argsort(bm25.get_scores(tok(p)))[::-1][:M]]
        _pl[p] = set(br) | set(bge.get(p, [])[:M])
    return _pl[p]

def doc_status(r, d):
    sts = []
    for c in r["parsed"]["clauses"]:
        if not any(d in poolc(p) for p in c):
            sts.append("unexposed")
            continue
        vs = [verify.get((p, d)) for p in c]
        if any(v is not None and not v["v"] for v in vs):
            sts.append("false")
        elif any(v is None for v in vs):
            sts.append("unknown")
        else:
            sts.append("true")
    if "true" in sts:
        return "member"
    if all(s == "unexposed" for s in sts):
        return "rankout"
    if "unknown" in sts:
        return "unknown"
    return "vfalse"

# ---- 対象の収集: 極値系でgold答えdocがvfalseの事例 ----
ids_re = re.compile(r"\d+")
targets = []  # (qid, question, gold_doc, failing_preds)
for rec in recs:
    if rec["op"] not in ("min_id", "max_id") and not rec["op"].startswith("topk"):
        continue
    r = parsed[rec["qid"]]
    for g in [int(x) for x in ids_re.findall(str(rec["gold"]))]:
        if doc_status(r, g) != "vfalse":
            continue
        fails = sorted({p for c in r["parsed"]["clauses"] for p in c
                        if (v := verify.get((p, g))) is not None and not v["v"]})
        targets.append({"qid": rec["qid"], "op": rec["op"], "doc": g,
                        "question": r["question"], "fails": fails})
print(f"対象: {len(targets)}件 (極値系のgold答えdoc vfalse)")

client = OpenAI(timeout=180)

LENIENT_SYS = (
    "You validate whether a resume can be described by given conditions. A condition "
    "is satisfied if the resume plausibly matches it, crediting reasonable semantic "
    "equivalents, synonyms, and clearly implied experience (e.g. a role that entails "
    "the skill). Do not require exact wording. For each condition output satisfies "
    "true/false and a short evidence snippet from the resume (paraphrase allowed).")

LEN_SCHEMA = {"type": "json_schema", "json_schema": {
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

OUT = "probe31.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        if l.strip():
            j = json.loads(l)
            done.add((j["qid"], j["doc"], j["variant"]))

res = collections.Counter()
with open(OUT, "a", encoding="utf-8") as out:
    for t in targets:
        text = raw_text[t["doc"]][:24000]
        # 変種A: lenient per-predicate (失敗述語のみ再判定)
        key = (t["qid"], t["doc"], "lenient")
        if key not in done and t["fails"]:
            clist = "\n".join(f"[{j}] {p}" for j, p in enumerate(t["fails"]))
            r_ = with_retry(lambda: client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": LENIENT_SYS},
                          {"role": "user", "content":
                           f"CONDITIONS:\n{clist}\n\nRESUME:\n{text}"}],
                response_format=LEN_SCHEMA))
            js = json.loads(r_.choices[0].message.content)["judgments"]
            flips = {t["fails"][int(j["condition"])]: bool(j["satisfies"])
                     for j in js if 0 <= int(j["condition"]) < len(t["fails"])}
            out.write(json.dumps({"qid": t["qid"], "doc": t["doc"],
                                  "variant": "lenient", "res": flips},
                                 ensure_ascii=False) + "\n")
            out.flush()
        # 変種B: query-level relevance (eq.(7)流)
        key = (t["qid"], t["doc"], "query")
        if key not in done:
            r_ = with_retry(lambda: client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": QUERY_SYS},
                          {"role": "user", "content":
                           f"QUERY: {t['question']}\n\nDOCUMENT:\n{text}\n\n"
                           "Does the document contain information to answer the "
                           "query (is it one of the documents being asked about)?"}],
                response_format=Q_SCHEMA))
            j = json.loads(r_.choices[0].message.content)
            out.write(json.dumps({"qid": t["qid"], "doc": t["doc"],
                                  "variant": "query",
                                  "res": bool(j["relevant"]),
                                  "reason": j.get("reason", "")[:200]},
                                 ensure_ascii=False) + "\n")
            out.flush()

# ---- 集計 ----
probe = collections.defaultdict(dict)
for l in open(OUT, encoding="utf-8"):
    if l.strip():
        j = json.loads(l)
        probe[(j["qid"], j["doc"])][j["variant"]] = j["res"]

print("\n=== 結果: gold答えdoc (strict=False) は代替判定で翻るか ===")
nA = nB = 0
for t in targets:
    p = probe.get((t["qid"], t["doc"]), {})
    lres = p.get("lenient", {})
    a = all(lres.get(x, False) for x in t["fails"]) if t["fails"] else None
    b = p.get("query")
    if a:
        nA += 1
    if b:
        nB += 1
    print(f"  qid{t['qid']:>5} doc{t['doc']:>5}: lenientで全述語通過={a}"
          f"  query-level関連={b}")
    for x in t["fails"]:
        print(f"      strict-F述語: {x[:70]} -> lenient={lres.get(x)}")
print(f"\n  -- lenient復活: {nA}/{len(targets)}   query-level復活: {nB}/{len(targets)}")
print("  判断: query-level復活が10+なら H4 濃厚 -> 較正ラウンドはquery-level系を軸に")
print("        lenientのみ高いなら述語判定の閾値問題 -> lenient系プロンプトを較正")

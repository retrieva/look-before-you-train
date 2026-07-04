# step10_v2.py - CORE v2: profile-based corpus scan engine
#   候補生成: dense(embedding) top-N ∪ BM25 top-N   (実測 union recall 96.8% @ N=200)
#   実体化:   候補プロファイルを25件/呼び出しで一括多値判定 (gpt-5-mini)
#   極値保険: min/max/top-k の答え候補だけ原文全体でLLM最終確認
#   集約:     コードで決定論的に実行
# 使い方:
#   python step10_v2.py --split train60 --limit 15    # パイロット (目安 $1-1.5)
#   python step10_v2.py --split train60               # 全60問 (目安 $5-6)
# 冪等: judge_cache / confirm_cache / results_v2 すべてチェックポイント式
import json, os, re, argparse, hashlib, collections
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
from util import with_retry
from step3_eval import pred_scores, token_f1

client = OpenAI(timeout=180)
DATA = "./data/GlobalQA"
MODEL = "gpt-5-mini"
JUDGE_CACHE = "judge_cache.jsonl"
CONFIRM_CACHE = "confirm_cache.jsonl"

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

# ---------------- 判定キャッシュ ----------------
def _load_jsonl_cache(path, keyf):
    c = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            if l.strip():
                try:
                    r = json.loads(l)
                    c[keyf(r)] = r["v"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return c

judge = _load_jsonl_cache(JUDGE_CACHE, lambda r: (r["p"], r["d"]))
confirm = _load_jsonl_cache(CONFIRM_CACHE, lambda r: (r["p"], r["d"]))
_jf = open(JUDGE_CACHE, "a", encoding="utf-8")
_cf = open(CONFIRM_CACHE, "a", encoding="utf-8")

def _save(f, pred, d, v):
    f.write(json.dumps({"p": pred, "d": d, "v": v}, ensure_ascii=False) + "\n")
    f.flush()

# ---------------- 一括プロファイル判定 ----------------
BATCH_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "judgments", "strict": True, "schema": {
        "type": "object",
        "properties": {"satisfying": {"type": "array", "items": {
            "type": "object",
            "properties": {"profile": {"type": "integer"},
                           "predicates": {"type": "array", "items": {"type": "integer"}}},
            "required": ["profile", "predicates"], "additionalProperties": False}}},
        "required": ["satisfying"], "additionalProperties": False}}}

JUDGE_SYS = (
    "You judge which resume profiles satisfy which predicates. A profile satisfies a "
    "predicate if the person's roles, skills, tools, domains or experience clearly "
    "match it, including obviously equivalent wording. Return, for each profile that "
    "satisfies at least one predicate, its number and the list of predicate indices "
    "it satisfies. Omit profiles that satisfy none.")

def judge_batch(preds, batch_ids, stats):
    plist = "\n".join(f"[{i}] {p}" for i, p in enumerate(preds))
    body = "\n\n".join(f"(profile {d})\n{profile_text(d)}" for d in batch_ids)
    r = with_retry(lambda: client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content":
                   f"PREDICATES:\n{plist}\n\nPROFILES:\n{body}"}],
        response_format=BATCH_SCHEMA))
    stats["judge_calls"] += 1
    sat = collections.defaultdict(set)
    for it in json.loads(r.choices[0].message.content)["satisfying"]:
        for pi in it["predicates"]:
            if 0 <= pi < len(preds):
                sat[it["profile"]].add(pi)
    for d in batch_ids:
        for i, p in enumerate(preds):
            v = i in sat.get(d, set())
            judge[(p, d)] = v
            _save(_jf, p, d, v)

def materialize(preds, combine, cands, stats, batch_size):
    todo = [d for d in cands
            if any((p, d) not in judge for p in preds) and d in profiles]
    for i in range(0, len(todo), batch_size):
        judge_batch(preds, todo[i:i + batch_size], stats)
    member = set()
    for d in cands:
        vals = [judge.get((p, d), False) for p in preds]
        if (any(vals) if combine == "or" else all(vals)):
            member.add(d)
    return member

# ---------------- 極値の原文最終確認 ----------------
CONFIRM_SYS = (
    "Based on the full resume below, answer whether it satisfies the condition. "
    "Consider synonyms, equivalent roles and clearly implied experience. "
    "Answer with JSON {\"satisfies\": true/false}.")
CONFIRM_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "confirm", "strict": True, "schema": {
        "type": "object", "properties": {"satisfies": {"type": "boolean"}},
        "required": ["satisfies"], "additionalProperties": False}}}

def confirm_doc(preds, combine, d, stats):
    """原文全体で式を確認 (or: どれか1つtrueで可 / and: 全部必要)"""
    order = sorted(preds, key=lambda p: judge.get((p, d), False), reverse=True)
    results = []
    for p in order:
        if (p, d) in confirm:
            v = confirm[(p, d)]
        else:
            r = with_retry(lambda: client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": CONFIRM_SYS},
                          {"role": "user", "content":
                           f"CONDITION: {p}\n\nRESUME:\n{raw_text[d][:24000]}"}],
                response_format=CONFIRM_SCHEMA))
            stats["confirm_calls"] += 1
            v = json.loads(r.choices[0].message.content)["satisfies"]
            confirm[(p, d)] = v
            _save(_cf, p, d, v)
        results.append(v)
        if combine == "or" and v:
            return True
        if combine == "and" and not v:
            return False
    return all(results) if combine == "and" else any(results)

# ---------------- クエリ実行 ----------------
def run_query(r, args, stats):
    p = r["parsed"]
    preds, combine, op = p["predicates"], p["combine"], p["agg"]
    k = p.get("k") or 1
    cands = set()
    for pr in preds:
        cands.update(dense_top(pr, args.n))
        cands.update(bm25_top(pr, args.n))
    member = materialize(preds, combine, cands, stats, args.batch)
    ids = sorted(member)
    if op == "count":
        return str(len(ids))
    if op in ("min_id", "max_id"):
        seq = ids if op == "min_id" else ids[::-1]
        for d in seq[:args.confirm_cap]:
            if confirm_doc(preds, combine, d, stats):
                return str(d)
        return str(seq[0]) if seq else ""       # 全滅なら未確認の端をフォールバック
    if op in ("topk_largest", "topk_smallest"):
        seq = ids[::-1] if op == "topk_largest" else ids
        out, checked = [], 0
        for d in seq:
            if len(out) >= k or checked >= 2 * k + 4:
                break
            checked += 1
            if confirm_doc(preds, combine, d, stats):
                out.append(d)
        for d in seq:                            # 確認合格が足りなければ未確認で補充
            if len(out) >= k:
                break
            if d not in out:
                out.append(d)
        return ", ".join(map(str, out))
    if op == "sort_asc":
        return ", ".join(map(str, ids))
    return ", ".join(map(str, ids[::-1]))        # sort_desc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train60", choices=["train60", "test100"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--confirm-cap", type=int, default=6, dest="confirm_cap")
    ap.add_argument("--limit", type=int, default=0, help="先頭N問のみ(パイロット用)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(f"parsed_{args.split}.jsonl", encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs("results_v2", exist_ok=True)
    out_path = f"results_v2/v2_{args.split}.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["qid"] for l in open(out_path, encoding="utf-8")
                if l.strip()}

    res = collections.defaultdict(list)
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
                   "judge_calls": stats["judge_calls"],
                   "confirm_calls": stats["confirm_calls"]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[qid {qid}] {rec['op']:14s} f1={f1:.2f}"
                  f"  judge={stats['judge_calls']} confirm={stats['confirm_calls']}",
                  flush=True)

    # サマリ (結果ファイル全体から集計)
    for l in open(out_path, encoding="utf-8"):
        if l.strip():
            rec = json.loads(l)
            res[rec["op"]].append(rec["f1"])
            res["_all"].append(rec["f1"])
    print(f"\n=== v2 {args.split} (n={len(res['_all'])}) ===")
    print(f"Answer F1: {np.mean(res['_all'])*100:.2f}"
          f"   (v1: 4.77 / v0: 1.80 on train60)")
    for kk, v in sorted(res.items()):
        if not kk.startswith("_"):
            print(f"  {kk:14s} {np.mean(v)*100:6.1f} (n={len(v)})")

if __name__ == "__main__":
    main()

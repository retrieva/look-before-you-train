# step13_autopsy.py - v2.1 敗因解剖 (課金ゼロ、キャッシュのみ)
#   A) FN golds: judge=falseだったgolden docについて、述語の内容語が
#      profile / raw のどちらに存在するかで原因を機械分類
#        - profileに証拠語あり  -> 判定ミス疑い (判定形式の変更で救える)
#        - rawのみに証拠語あり  -> 抽出ロス疑い (judge入力に原文チャンク追加が必要)
#        - どちらにも無し       -> 意味的マッチ必要 (字句では判別不能)
#   B) FP members: count/sortクエリの非gold memberをサンプル表示 (目視用)
#   C) confirm通過FP: min/max/topkで予測に採用された非gold文書のconfirm状況
import json, os, re, collections
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import pred_scores

DATA = "./data/GlobalQA"
N = 200

# ---------- 読み込み ----------
parsed = {}
for l in open("parsed_train60.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        parsed[r["idx"]] = r

recs = [json.loads(l) for l in open("results_v2/v2_train60.jsonl", encoding="utf-8")
        if l.strip()]

def load_cache(path):
    c = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            if l.strip():
                try:
                    r = json.loads(l)
                    c[(r["p"], r["d"])] = r["v"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return c

judge = load_cache("judge_cache.jsonl")
confirm = load_cache("confirm_cache.jsonl")

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

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

def cands_for(r):
    c = set()
    for p in r["parsed"]["predicates"]:
        s = pred_scores(p)
        c.update(sorted(s, key=s.get, reverse=True)[:N])
        c.update([doc_ids[i] for i in np.argsort(bm25.get_scores(tok(p)))[::-1][:N]])
    return c

def member_of(r, d):
    preds = r["parsed"]["predicates"]
    vals = [judge.get((p, d), None) for p in preds]
    if any(v is None for v in vals):
        return None
    return any(vals) if r["parsed"]["combine"] == "or" else all(vals)

# ---------- 字句カバレッジ ----------
STOP = set(("a an the of in with and or for to on at by is are who has have had "
            "experience experienced expertise skilled skills background work worked "
            "working knowledge proficient familiar strong someone people person "
            "candidates resumes years using used use").split())
def content_toks(s):
    return [t for t in tok(s) if t not in STOP and len(t) > 2]

def coverage(pred, text_tokens):
    ts = content_toks(pred)
    if not ts:
        return 1.0
    return sum(t in text_tokens for t in ts) / len(ts)

# ---------- A) FN golds の原因分類 ----------
print("\n=== A) golden doc の偽陰性: 原因分類 (judge=false の (述語,doc) ペア単位) ===")
counts = collections.Counter()
examples = []
for rec in recs:
    r = parsed[rec["qid"]]
    combine = r["parsed"]["combine"]
    for g in r["parsed"].get("golden_doc_ids", r.get("golden_doc_ids", [])) or r["golden_doc_ids"]:
        if g not in profiles:
            continue
        prof_toks = set(tok(profile_text(g, max_chars=10**6)))
        raw_toks = set(tok(raw_text[g]))
        for p in r["parsed"]["predicates"]:
            v = judge.get((p, g))
            if v is not False:
                continue
            cp = coverage(p, prof_toks)
            cr = coverage(p, raw_toks)
            if cp >= 0.6:
                cls = "判定ミス疑い(profileに証拠語あり)"
            elif cr >= 0.6:
                cls = "抽出ロス疑い(rawのみに証拠語)"
            else:
                cls = "字句証拠なし(意味マッチ必要)"
            counts[cls] += 1
            examples.append((cls, rec["qid"], combine, g, p, cp, cr))

tot = sum(counts.values())
for cls, n in counts.most_common():
    print(f"  {cls:32s} {n:>4} ({n/tot*100:.0f}%)" if tot else "")
print(f"  合計 judge=false の (述語,gold) ペア: {tot}")

print("\n  -- 例 (各分類から最大6件) --")
shown = collections.Counter()
for cls, qid, combine, g, p, cp, cr in examples:
    if shown[cls] >= 6:
        continue
    shown[cls] += 1
    print(f"  [{cls}] qid{qid} ({combine}) doc{g}  cov(prof)={cp:.2f} cov(raw)={cr:.2f}")
    print(f"      述語: {p[:110]}")

# ---------- B) FP members のサンプル (count/sort) ----------
print("\n=== B) FP member サンプル (count/sort, 非gold member) ===")
for rec in recs:
    if rec["op"] != "count" and not rec["op"].startswith("sort"):
        continue
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    members = [d for d in cands_for(r) if member_of(r, d) is True]
    fps = sorted(set(members) - gold)
    print(f"\n  qid{rec['qid']} {rec['op']} ({r['parsed']['combine']})"
          f"  |member|={len(members)} |gold|={len(gold)} FP={len(fps)}")
    for i, p in enumerate(r["parsed"]["predicates"]):
        print(f"    述語[{i}]: {p[:110]}")
    for d in fps[:4]:
        trues = [i for i, p in enumerate(r["parsed"]["predicates"])
                 if judge.get((p, d))]
        pt = profile_text(d)
        head = " / ".join(x for x in pt.split("\n")[:2])
        print(f"    FP doc{d} (true述語={trues}): {head[:150]}")

# ---------- C) confirm通過FP (min/max/topk の予測採用doc) ----------
print("\n=== C) 予測に採用された非gold文書のconfirm状況 (min/max/topk) ===")
ids_re = re.compile(r"\d+")
for rec in recs:
    if rec["op"] not in ("min_id", "max_id") and not rec["op"].startswith("topk"):
        continue
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    for d in map(int, ids_re.findall(rec["pred"])):
        if d in gold:
            continue
        cv = {i: confirm.get((p, d)) for i, p in enumerate(r["parsed"]["predicates"])
              if (p, d) in confirm}
        via = "confirm通過" if any(v for v in cv.values()) else \
              ("confirm全falseなのに採用(フォールバック)" if cv else "confirm未実施(フォールバック)")
        print(f"  qid{rec['qid']} {rec['op']:13s} 採用doc{d:>5} ({via}) confirm={cv}")
        head = " / ".join(profile_text(d).split("\n")[:2])
        print(f"      {head[:150]}")

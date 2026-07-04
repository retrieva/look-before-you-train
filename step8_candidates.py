# step8_candidates.py - 候補生成のリコール測定 (API課金ゼロ)
#   dense top-N (embedding, キャッシュ済み) vs BM25 top-N vs 和集合
#   ついでに述語の重複率も測る (キャッシュ償却の効き具合の見積り)
# 事前に: pip install rank_bm25
import json, re, collections
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import pred_scores   # キャッシュ済み述語embedding -> 無料

DATA = "./data/GlobalQA"
NS = [30, 50, 100, 200]

rows = [json.loads(l) for l in open("parsed_train60.jsonl", encoding="utf-8") if l.strip()]
docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]

def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())

print("BM25インデックス構築中 (初回のみ数十秒)...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

_bm_cache = {}
def bm25_rank(pred):
    if pred not in _bm_cache:
        scores = bm25.get_scores(tok(pred))
        _bm_cache[pred] = [doc_ids[i] for i in np.argsort(scores)[::-1]]
    return _bm_cache[pred]

_dense_cache = {}
def dense_rank(pred):
    if pred not in _dense_cache:
        s = pred_scores(pred)
        _dense_cache[pred] = sorted(s, key=s.get, reverse=True)
    return _dense_cache[pred]

def query_cands(r, N, mode):
    c = set()
    for p in r["parsed"]["predicates"]:
        if mode in ("dense", "union"):
            c.update(dense_rank(p)[:N])
        if mode in ("bm25", "union"):
            c.update(bm25_rank(p)[:N])
    return c

print(f"\n{'N':>4} | {'dense':>7} {'bm25':>7} {'union':>7} | union候補数/クエリ")
for N in NS:
    recs = {m: [] for m in ("dense", "bm25", "union")}
    sizes = []
    for r in rows:
        gold = set(r["golden_doc_ids"])
        for m in recs:
            c = query_cands(r, N, m)
            recs[m].append(len(c & gold) / len(gold))
            if m == "union": sizes.append(len(c))
    print(f"{N:>4} | {np.mean(recs['dense'])*100:6.1f}% {np.mean(recs['bm25'])*100:6.1f}%"
          f" {np.mean(recs['union'])*100:6.1f}% | {np.mean(sizes):6.0f}")

# min/max: 答えの文書が候補に入るか (union, N別)
print("\n-- min/max: 答えの文書が候補(union)に入る率 --")
mm = [r for r in rows if r["parsed"]["agg"] in ("min_id", "max_id")]
for N in NS:
    hit = []
    for r in mm:
        try: a = int(str(r["answer"]).strip())
        except ValueError: continue
        hit.append(a in query_cands(r, N, "union"))
    print(f"  N={N}: {np.mean(hit)*100:.0f}%  (n={len(hit)})")

# 述語の重複率 (test100+train60の160クエリで)
preds = []
for f in ("parsed_train60.jsonl", "parsed_test100.jsonl"):
    for l in open(f, encoding="utf-8"):
        if l.strip(): preds += json.loads(l)["parsed"]["predicates"]
c = collections.Counter(preds)
print(f"\n述語の重複: 総数={len(preds)}, ユニーク={len(c)}, "
      f"再利用率={100*(1-len(c)/len(preds)):.1f}%")
print("頻出述語トップ5:", c.most_common(5))

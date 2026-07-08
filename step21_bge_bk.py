# step21_bge.py - BGE-large-en (構築と同じ検索器) をローカルで実行 (API課金ゼロ)
#   1) コーパスをstep1と同じ6000文字チャンクに分割し、BAAI/bge-large-en-v1.5 で埋め込み
#   2) parsed_train60 (存在すればparsed_test100も) の全述語のランキングを作成し
#      bge_ranks.json にキャッシュ (エンジン側はtorch不要でこれを読むだけ)
#   3) gold docのBGE順位分布と BGE∪BM25 順位分布を測定し、mの推奨値を出す
# 事前: pip install sentence-transformers  (初回はモデル~1.3GBをDL。CPUで10-30分)
import json, os, re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = "./data/GlobalQA"
TOPN = 300
Q_PREFIX = "Represent this sentence for searching relevant passages: "

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]

# step1と同一のチャンク分割 (6000文字の機械分割)
chunks, owner = [], []
for d in docs:
    t = d["contents"]
    for i in range(0, len(t), 6000):
        chunks.append(t[i:i + 6000])
        owner.append(d["id"])
print(f"docs={len(docs)} chunks={len(chunks)}")

print("BGE-large-en-v1.5 ロード中 (初回はダウンロード)...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

if os.path.exists("bge_chunk_embs.npy"):
    embs = np.load("bge_chunk_embs.npy")
    print("チャンク埋め込みをキャッシュから読み込み")
else:
    print("チャンク埋め込み計算中 (CPUで10-30分。進捗バー表示)...")
    embs = model.encode(chunks, batch_size=16, show_progress_bar=True,
                        normalize_embeddings=True)
    np.save("bge_chunk_embs.npy", embs)
owner = np.array(owner)

# ---- 述語ランキング ----
preds = set()
for f in ["parsed_train60.jsonl", "parsed_test100.jsonl"]:
    if os.path.exists(f):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                preds.update(json.loads(l)["parsed"]["predicates"])
preds = sorted(preds)
print(f"述語数: {len(preds)}")

ranks = {}
if os.path.exists("bge_ranks.json"):
    ranks = json.load(open("bge_ranks.json", encoding="utf-8"))
todo = [p for p in preds if p not in ranks]
if todo:
    qe = model.encode([Q_PREFIX + p for p in todo], batch_size=16,
                      show_progress_bar=True, normalize_embeddings=True)
    for p, q in zip(todo, qe):
        sims = embs @ q                      # チャンク類似度
        best = {}
        for s, o in zip(sims, owner):        # doc = maxチャンク
            o = int(o)
            if s > best.get(o, -2):
                best[o] = float(s)
        order = sorted(best, key=best.get, reverse=True)[:TOPN]
        ranks[p] = order
    json.dump(ranks, open("bge_ranks.json", "w", encoding="utf-8"))
print("bge_ranks.json 保存済み")

# ---- gold順位の測定 (train60) ----
print("\nBM25構築中 (比較用)...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])
bm_rank = {}
def bm(p):
    if p not in bm_rank:
        order = list(np.argsort(bm25.get_scores(tok(p)))[::-1])
        bm_rank[p] = {doc_ids[i]: rk for rk, i in enumerate(order)}
    return bm_rank[p]

rows = [json.loads(l) for l in open("parsed_train60.jsonl", encoding="utf-8")
        if l.strip()]
bge_r, uni_r = [], []
for r in rows:
    ps = r["parsed"]["predicates"]
    rk_bge = {p: {d: i for i, d in enumerate(ranks[p])} for p in ps}
    for g in r["golden_doc_ids"]:
        b = min(rk_bge[p].get(g, 10**9) for p in ps)
        u = min(b, min(bm(p).get(g, 10**9) for p in ps))
        bge_r.append(b); uni_r.append(u)
bge_r, uni_r = np.array(bge_r), np.array(uni_r)

print(f"\n=== gold docの順位分布 (train60, n={len(bge_r)}) ===")
print(f"{'m':>5} {'BGEのみ':>8} {'BGE∪BM25':>9}   (参考: 旧dense∪BM25はtop-20~55-60%, top-50=79.5%)")
for m in [10, 20, 30, 50, 80, 100, 200]:
    print(f"{m:>5} {np.mean(bge_r < m)*100:7.1f}% {np.mean(uni_r < m)*100:8.1f}%")
print("\nm推奨: BGE∪BM25でリコール85%以上に達する最小のmを step22 の --m に使う")

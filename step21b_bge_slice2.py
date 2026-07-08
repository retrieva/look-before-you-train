# step21b_bge_slice2.py - step21にparsed2_slice2.jsonlを追加しただけの版 (課金ゼロ)
#   埋め込み計算・キャッシュ・ロジックはstep21と完全同一。新述語のみ差分計算
# 元: step21_bge.py
#   1) コーパスをstep1と同じ6000文字チャンクに分割し、BAAI/bge-large-en-v1.5 で埋め込み
#   2) parsed_* (旧フラット) と parsed2_* (v2.5 DNF) の全述語のランキングを作成し
#      bge_ranks.json にキャッシュ (エンジン側はtorch不要でこれを読むだけ)
#   3) gold docの順位分布 (BGE / BGE∪BM25) を旧パースとparsed2の両方で測定
#      -> 同一mでの露出率の差 = step25の述語書き換えによるリコール影響
# 事前: pip install sentence-transformers  (初回はモデル~1.3GBをDL)
# 冪等: チャンク埋め込みと既存述語ランキングはキャッシュ再利用、新述語のみ差分計算
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

# ---- 述語の収集 (旧フラット + v2.5 DNF の両形式) ----
def preds_of(rec):
    p = rec["parsed"]
    if "clauses" in p:                        # v2.5 DNF形式
        return {x for c in p["clauses"] for x in c}
    return set(p["predicates"])               # 旧フラット形式

PARSED_FILES = ["parsed_train60.jsonl", "parsed_test100.jsonl",
                "parsed2_train60.jsonl", "parsed2_test100.jsonl",
                "parsed2_slice2.jsonl"]
preds = set()
for f in PARSED_FILES:
    if os.path.exists(f):
        n0 = len(preds)
        for l in open(f, encoding="utf-8"):
            if l.strip():
                preds.update(preds_of(json.loads(l)))
        print(f"  {f}: +{len(preds) - n0} 述語")
preds = sorted(preds)
print(f"述語数 (合計): {len(preds)}")

# ---- 述語ランキング (差分のみ計算) ----
ranks = {}
if os.path.exists("bge_ranks.json"):
    ranks = json.load(open("bge_ranks.json", encoding="utf-8"))
todo = [p for p in preds if p not in ranks]
print(f"新規計算する述語: {len(todo)}")
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

# ---- gold順位の測定 (旧パース vs parsed2) ----
print("\nBM25構築中 (比較用)...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])
bm_rank = {}
def bm(p):
    if p not in bm_rank:
        order = list(np.argsort(bm25.get_scores(tok(p)))[::-1])
        bm_rank[p] = {doc_ids[i]: rk for rk, i in enumerate(order)}
    return bm_rank[p]

def measure(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    bge_r, uni_r = [], []
    for r in rows:
        ps = sorted(preds_of(r))
        rk_bge = {p: {d: i for i, d in enumerate(ranks[p])} for p in ps}
        for g in r["golden_doc_ids"]:
            b = min(rk_bge[p].get(g, 10**9) for p in ps)
            u = min(b, min(bm(p).get(g, 10**9) for p in ps))
            bge_r.append(b); uni_r.append(u)
    bge_r, uni_r = np.array(bge_r), np.array(uni_r)
    print(f"\n=== gold docの順位分布 ({path}, n={len(bge_r)}) ===")
    print(f"{'m':>5} {'BGEのみ':>8} {'BGE∪BM25':>9}")
    for m in [10, 20, 30, 50, 80, 100, 200]:
        print(f"{m:>5} {np.mean(bge_r < m)*100:7.1f}% {np.mean(uni_r < m)*100:8.1f}%")

for path in ["parsed2_train60.jsonl", "parsed2_slice2.jsonl"]:
    if os.path.exists(path):
        measure(path)
print("\nm推奨: BGE∪BM25でリコール85%以上に達する最小のmをエンジンの --m に使う")
print("(旧parsedとparsed2の同一mでの露出率の差 = step25述語書き換えのリコール影響)")

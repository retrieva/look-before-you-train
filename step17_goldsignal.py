# step17_goldsignal.py - goldと「忠実な意味判定の通過者」を分ける信号の探索 (課金ゼロ)
#   H1 (prominence): goldは述語が主要属性 (肩書き/中核スキル) の文書に限られる
#   H2 (bounded retrieval): goldは構築時の検索top-k候補に事実上束縛されている
#   裁定:
#     - Part1: 全train60でgold docの検索順位 (BM25/dense, 述語間min) の分布
#     - Part2: パイロットのverify通過FPとgoldの順位分布を直接比較 (H2の直接検定)
#     - Part3: 述語語のTITLES内出現・原文中頻度でgold/FPを比較 (H1の直接検定)
import json, os, re, collections
import numpy as np
from rank_bm25 import BM25Okapi
from step3_eval import pred_scores

PV = "v22a"
DATA = "./data/GlobalQA"

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
STOP = set(("a an the of in with and or for to on at by is are who has have had "
            "experience experienced expertise skilled skills background work worked "
            "working knowledge proficient familiar strong someone people person "
            "candidates resumes years using used use related relate").split())
def content_toks(s):
    return [t for t in tok(s) if t not in STOP and len(t) > 2]

docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8") if l.strip()]
doc_ids = [d["id"] for d in docs]
raw_text = {d["id"]: d["contents"] for d in docs}
print("BM25構築中...")
bm25 = BM25Okapi([tok(d["contents"]) for d in docs])

profiles = {}
for l in open("profiles.jsonl", encoding="utf-8"):
    if l.strip():
        p = json.loads(l)
        profiles[p["id"]] = p

rows = [json.loads(l) for l in open("parsed_train60.jsonl", encoding="utf-8")
        if l.strip()]
parsed = {r["idx"]: r for r in rows}

# ---- 述語ごとの順位表を前計算 ----
uniq = sorted({p for r in rows for p in r["parsed"]["predicates"]})
bm_rank, dn_rank = {}, {}
for p in uniq:
    order = list(np.argsort(bm25.get_scores(tok(p)))[::-1])
    bm_rank[p] = {doc_ids[i]: rk for rk, i in enumerate(order)}
    s = pred_scores(p)
    dn_rank[p] = {d: rk for rk, d in
                  enumerate(sorted(s, key=s.get, reverse=True))}

def min_rank(r, d):
    """クエリの全述語にわたる最良順位 (BM25とdenseの小さい方)"""
    best = 10**9
    for p in r["parsed"]["predicates"]:
        best = min(best, bm_rank[p].get(d, 10**9), dn_rank[p].get(d, 10**9))
    return best

# ---- Part 1: 全60クエリのgold順位分布 ----
print("\n=== Part1) gold docの検索順位 (述語間・BM25/dense間のmin, train60全問) ===")
ranks = []
for r in rows:
    for g in r["golden_doc_ids"]:
        ranks.append(min_rank(r, g))
ranks = np.array(ranks)
print(f"  gold doc数: {len(ranks)}  中央値={int(np.median(ranks))} "
      f"p90={int(np.percentile(ranks,90))} 最大={ranks.max()}")
for m in [10, 30, 50, 100, 200]:
    print(f"  top-{m:>3} 以内: {np.mean(ranks < m)*100:5.1f}%")

# ---- verify結果の読み込み (パイロット分) ----
verify = {}
if os.path.exists("verify_cache.jsonl"):
    for l in open("verify_cache.jsonl", encoding="utf-8"):
        if l.strip():
            try:
                v = json.loads(l)
                if v.get("pv") == PV:
                    verify[(v["p"], v["d"])] = v
            except (json.JSONDecodeError, KeyError):
                pass

recs = []
if os.path.exists("results_v22/v22_train60.jsonl"):
    recs = [json.loads(l) for l in open("results_v22/v22_train60.jsonl",
                                        encoding="utf-8") if l.strip()]

# ---- Part 2: verify通過FP vs gold の順位比較 (H2の直接検定) ----
print("\n=== Part2) verify通過doc: gold vs FP の順位分布 (パイロットのcount/sort中心) ===")
def or_pass(r, d):
    vs = [verify.get((p, d)) for p in r["parsed"]["predicates"]]
    vv = [v["v"] for v in vs if v is not None]
    if not vv:
        return None
    return any(vv) if r["parsed"]["combine"] == "or" else \
        (all(v["v"] for v in vs if v is not None) and all(v is not None for v in vs))

for rec in recs:
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    passed = [d for d in doc_ids if or_pass(r, d) is True]
    if len(passed) < 20:      # 早期打ち切りの極値系はサンプル不足なのでスキップ
        continue
    gr = [min_rank(r, d) for d in passed if d in gold]
    fr = [min_rank(r, d) for d in passed if d not in gold]
    if gr and fr:
        print(f"  qid{rec['qid']} {rec['op']}: 通過gold n={len(gr)} 順位中央値={int(np.median(gr))}"
              f" / 通過FP n={len(fr)} 順位中央値={int(np.median(fr))}")
        for m in [10, 30, 50]:
            print(f"     top-{m:>3}以内: gold {np.mean(np.array(gr)<m)*100:4.0f}%"
                  f"  FP {np.mean(np.array(fr)<m)*100:4.0f}%")

# ---- Part 3: prominence比較 (H1の直接検定) ----
print("\n=== Part3) prominence: 述語語のTITLES出現とTF (verify通過のgold vs FP) ===")
def prom(r, d):
    """クエリ述語ベスト: TITLES内容語ヒット率と原文TF密度"""
    if d not in profiles:
        return 0.0, 0.0
    titles = " ".join(profiles[d].get("titles", [])).lower()
    ttoks = set(tok(titles))
    rtoks = tok(raw_text[d])
    tf = collections.Counter(rtoks)
    best_t, best_f = 0.0, 0.0
    for p in r["parsed"]["predicates"]:
        cs = content_toks(p)
        if not cs:
            continue
        best_t = max(best_t, sum(c in ttoks for c in cs) / len(cs))
        best_f = max(best_f, sum(tf[c] for c in cs) / max(len(rtoks), 1) * 1000)
    return best_t, best_f

for rec in recs:
    r = parsed[rec["qid"]]
    gold = set(r["golden_doc_ids"])
    passed = [d for d in doc_ids if or_pass(r, d) is True]
    if len(passed) < 20:
        continue
    gp = [prom(r, d) for d in passed if d in gold]
    fp = [prom(r, d) for d in passed if d not in gold]
    if gp and fp:
        gt = np.mean([x[0] for x in gp]); ft = np.mean([x[0] for x in fp])
        gf = np.median([x[1] for x in gp]); ff = np.median([x[1] for x in fp])
        print(f"  qid{rec['qid']} {rec['op']}: TITLESヒット率 gold={gt:.2f} FP={ft:.2f}"
              f" / TF密度(中央値,/1000語) gold={gf:.2f} FP={ff:.2f}")

print("\n読み方:")
print("  - Part1でgoldの大半がtop-30~50以内 + Part2でFPが深い順位に分布 -> H2優勢:")
print("    構築時の検索深度を再現し「述語ごとに検索上位m件のみverify」へ (コスト激減)")
print("  - 順位で分離せず Part3のTITLESヒット率/TF密度で分離 -> H1優勢:")
print("    verify基準を「主要属性か」に変更 (PV=v22b)")
print("  - どちらでも分離しない -> gold規則は外形から同定不能。構築コード/プロンプトの")
print("    一次情報 (リポジトリ) の入手が最優先タスクに昇格")

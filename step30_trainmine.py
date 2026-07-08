# step30_trainmine.py - train 12.9k問の無料調査: 較正資産の棚卸し (課金ゼロ)
#   1) スキーマ自動検出とタスク分類 (question/answerパターンから)
#   2) gold規則の不変量検証: count答え==|gold|? min/max答え==端? sort答え==gold列?
#      (答えがgold集合から機械的に導出されているかの全数確認)
#   3) |gold| 分布 (タスク別): 2-50の確認、>20の割合 (D-F1@20上限問題の規模)
#   4) ブール形式の頻度調査: either/both/and also/related to X and Y ... (DNF難形の census)
#   5) 文書ごとの教師密度: 各docが何問のgoldに登場するか (較正データの厚み)
#   6) 較正サンプルの切り出し: タスク層化で120問 -> calib_sample.jsonl
#      (train60のidxは除外して独立性を保つ)
# 使い方: python step30_trainmine.py
import json, os, re, random, collections
import numpy as np

DATA = "./data/GlobalQA"
random.seed(42)

# ---- 0) 読み込みとスキーマ検出 ----
path = None
for cand in ["train (1).json", "train.json", "train(1).json"]:
    p = os.path.join(DATA, cand)
    if os.path.exists(p):
        path = p
        break
assert path, f"trainファイルが {DATA} に見つからない"
raw = json.load(open(path, encoding="utf-8"))
rows = raw
if isinstance(raw, dict):
    for v in raw.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            rows = v
            break
print(f"読み込み: {path}  n={len(rows)}")
print(f"先頭レコードのキー: {list(rows[0].keys())}")

def detect(rows):
    s = rows[0]
    qf = af = gf = None
    for k, v in s.items():
        if isinstance(v, str) and len(v) > 30 and qf is None:
            qf = k
        if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
            gf = k
    for k in s:
        if k.lower() in ("answer", "answers", "ans", "golden_answer"):
            af = k
    if af is None:
        for k, v in s.items():
            if k not in (qf, gf) and isinstance(v, (str, int)):
                af = k
                break
    return qf, af, gf

QF, AF, GF = detect(rows)
print(f"検出フィールド: question={QF} answer={AF} gold={GF}")

# ---- 1) タスク分類 ----
def task_of(q):
    ql = q.lower()
    if "how many" in ql:
        return "count"
    m = re.search(r"top\s*(\d+)", ql)
    if m:
        return "topk_smallest" if ("smallest" in ql or "lowest" in ql) else "topk_largest"
    if "ascending" in ql:
        return "sort_asc"
    if "descending" in ql:
        return "sort_desc"
    if "smallest" in ql or "earliest" in ql or "lowest" in ql:
        return "min_id"
    if "biggest" in ql or "largest" in ql or "most recent" in ql or "highest" in ql:
        return "max_id"
    if "sort" in ql:
        return "sort_unknown"
    return "unknown"

tasks = collections.Counter(task_of(r[QF]) for r in rows)
print("\n=== 1) タスク分布 ===")
for k, v in tasks.most_common():
    print(f"  {k:14s} {v:>6} ({v/len(rows)*100:.1f}%)")

# ---- 2) gold規則の不変量検証 ----
print("\n=== 2) 答え==gold集合の機械的導出か (全数検証) ===")
ids_re = re.compile(r"\d+")
ok = bad = 0
bad_ex = []
for r in rows:
    t = task_of(r[QF])
    g = r[GF]
    a = str(r[AF])
    exp = None
    if t == "count":
        exp = str(len(g))
    elif t == "min_id":
        exp = str(min(g)) if g else ""
    elif t == "max_id":
        exp = str(max(g)) if g else ""
    elif t in ("sort_asc", "sort_desc", "topk_smallest", "topk_largest"):
        ids = sorted(g)
        if t == "sort_desc":
            ids = ids[::-1]
        if t.startswith("topk"):
            m = re.search(r"top\s*(\d+)", r[QF].lower())
            k = int(m.group(1)) if m else 1
            ids = (ids if t == "topk_smallest" else ids[::-1])[:k]
        exp = [str(x) for x in ids]
    if exp is None:
        continue
    got = ids_re.findall(a) if isinstance(exp, list) else a.strip()
    tgt = exp if isinstance(exp, list) else exp
    if (got == tgt) or (isinstance(exp, list) and set(got) == set(tgt)):
        ok += 1
    else:
        bad += 1
        if len(bad_ex) < 5:
            bad_ex.append((t, a[:60], str(exp)[:60]))
print(f"  一致={ok} 不一致={bad}")
for t, a, e in bad_ex:
    print(f"    [{t}] answer=\"{a}\" 期待=\"{e}\"")

# ---- 3) |gold| 分布 ----
print("\n=== 3) |gold| 分布 (タスク別) ===")
by = collections.defaultdict(list)
for r in rows:
    by[task_of(r[QF])].append(len(r[GF]))
for t, v in sorted(by.items()):
    v = np.array(v)
    print(f"  {t:14s} n={len(v):>6} min={v.min():>3} 中央値={int(np.median(v)):>3}"
          f" max={v.max():>3}  >20:{np.mean(v > 20)*100:4.1f}%  >50:{np.mean(v > 50)*100:4.1f}%")

# ---- 4) ブール形式の census ----
print("\n=== 4) ブール難形の頻度 (質問文の字面) ===")
pats = {
    "either ... or": r"\beither\b.*\bor\b",
    "both ... and": r"\bboth\b.*\band\b",
    "and also / who are also": r"\band also\b|\bwho are also\b|\balso include\b",
    "related to X and Y": r"related to [^,?]+ and ",
    "or (単純)": r"\bor\b",
    "and (単純)": r"\band\b",
}
for name, pat in pats.items():
    n = sum(1 for r in rows if re.search(pat, r[QF].lower()))
    print(f"  {name:24s} {n:>6} ({n/len(rows)*100:.1f}%)")

# ---- 5) 文書ごとの教師密度 ----
print("\n=== 5) doc教師密度: 各docが何問のgoldに登場するか ===")
dc = collections.Counter()
for r in rows:
    for d in r[GF]:
        dc[d] += 1
freq = np.array(list(dc.values()))
print(f"  gold登場docのユニーク数: {len(dc)} / 2047")
print(f"  1docあたり登場回数: min={freq.min()} 中央値={int(np.median(freq))}"
      f" max={freq.max()}")
print("  -> (述語,doc,ラベル)ペアの理論在庫は数十万規模。judge較正の教師は十分")

# ---- 6) 較正サンプル切り出し (タスク層化120問, train60除外) ----
print("\n=== 6) 較正サンプル calib_sample.jsonl ===")
exclude = set()
if os.path.exists("parsed_train60.jsonl"):
    exclude = {json.loads(l)["idx"] for l in open("parsed_train60.jsonl",
               encoding="utf-8") if l.strip()}
pool = collections.defaultdict(list)
for i, r in enumerate(rows):
    if i in exclude:
        continue
    t = task_of(r[QF])
    if t not in ("unknown", "sort_unknown") and 2 <= len(r[GF]) <= 50:
        pool[t].append(i)
per = 120 // max(len(pool), 1)
with open("calib_sample.jsonl", "w", encoding="utf-8") as out:
    tot = 0
    for t, idxs in sorted(pool.items()):
        for i in sorted(random.sample(idxs, min(per, len(idxs)))):
            r = rows[i]
            out.write(json.dumps({"idx": i, "question": r[QF],
                                  "answer": str(r[AF]),
                                  "golden_doc_ids": r[GF]},
                                 ensure_ascii=False) + "\n")
            tot += 1
        print(f"  {t:14s} {min(per, len(idxs))}問")
print(f"  合計{tot}問 -> calib_sample.jsonl (train60のidxは除外済み)")
print("  次段: この標本を step25 でパース -> gold doc + 検索浅層の非gold docに対し")
print("  judge判定を取り、gold規則との一致率 (TPR/FPR) をjudge/プロンプト別に測る")

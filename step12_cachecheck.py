# step12_cachecheck.py - judge_cache の「位置インデックス/実ID混同」チェック (課金ゼロ)
#   仮説: judge_batch はプロファイルを「(profile 実ID)」で提示しているが、
#   モデルがバッチ内の位置番号 (0..24) を返している回がある。
#   その場合、実ID 0〜24 の文書にだけ偽の true が蓄積し、
#   実IDが返答に現れなかった文書 (golden含む) は一律 false になる。
#   判別法: 実IDの帯ごとに true 率を比較。[0,25) だけ突出していれば混同確定。
#   (クリーンな判定なら true 率は ID にほぼ依存しないはず)
import json, collections

tot = collections.Counter()
tru = collections.Counter()
n_pairs = 0
for l in open("judge_cache.jsonl", encoding="utf-8"):
    if not l.strip():
        continue
    try:
        r = json.loads(l)
    except json.JSONDecodeError:
        continue
    d = r["d"]
    n_pairs += 1
    tot[d] += 1
    if r["v"]:
        tru[d] += 1

print(f"cache entries: {n_pairs} / distinct docs: {len(tot)}")
print(f"\n{'ID帯':>16} {'true':>7} {'total':>7} {'true率':>8}")
bands = [(0, 25), (25, 50), (50, 100), (100, 500),
         (500, 1000), (1000, 1500), (1500, 2047)]
for lo, hi in bands:
    t = sum(v for d, v in tru.items() if lo <= d < hi)
    n = sum(v for d, v in tot.items() if lo <= d < hi)
    pct = t / n * 100 if n else 0.0
    print(f"[{lo:>6},{hi:>6}) {t:>7} {n:>7} {pct:7.1f}%")

print("\ntrue率が高い文書 top 20 (判定数5以上):")
rows = [(tru[d] / tot[d], tru[d], tot[d], d) for d in tot if tot[d] >= 5]
for rate, t, n, d in sorted(rows, reverse=True)[:20]:
    print(f"  doc {d:>5}: {t}/{n} = {rate*100:5.1f}%")

print("\n読み方:")
print("  - [0,25) の true率が他の帯より大きく高い -> 位置/ID混同が確定。")
print("    step10 v2.1 (位置インデックス方式) に差し替え、judge_cache を退避して再実行。")
print("  - 帯間で true率がほぼ均一 -> 混同ではなく判定品質の問題。")
print("    --batch 10 に縮小して再実行 (この場合もキャッシュ退避が必要)。")

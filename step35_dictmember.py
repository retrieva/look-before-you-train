# step35_dictmember.py - フレーズ→docプール辞書 (train逆推定) によるmember集合
#   発見 (2026-07-07, サンドボックス検証済み):
#   gold規則はプロンプトでは表現できないが、trainラベルから辞書として学べる。
#   条件フレーズはtrain12,875問で大量再利用される (支持数中央値89)。
#   フレーズpを含む他問のgold頻度 freq(d) を推定し、クローズ (AND句) のスコア
#   min_ph freq >= θ でmember化。DNF (OR) はクローズの和集合。
#   train60実測 (θ=0.3, sup=3, 自問除外): 解決率77.4% / memberF1 51.0 / answerF1 32.9
#   (v2.5=8.85, MARAG-7B=28.60, 14B=31.22。完全解決20問に限ればanswerF1 62.7)
#   タスク別: count 0.0 (個数完全一致が必要) / min_id 50.0 / topk_s 57.1 / 他 37-42
# 位置づけ: 論文貢献(c) 「train較正によるベンチ判定機の再現」の完全形。
#   ベンチ構造への適応なのでablation必須 (辞書のみ/judgeのみ/混成の3点比較)。
# 使い方:
#   python step35_dictmember.py                            (train60評価, 課金ゼロ)
#   python step35_dictmember.py --theta 0.3 --min-sup 3
#   python step35_dictmember.py --coverage-file "./data/GlobalQA/test (3).json"
#       (test質問テキストのフレーズ支持率を測る。ラベルは一切読まない)
import json, re, argparse, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="./data/GlobalQA/train (1).json")
ap.add_argument("--parsed", default="parsed2_train60.jsonl")
ap.add_argument("--theta", type=float, default=0.3)
ap.add_argument("--min-sup", type=int, default=3, dest="min_sup")
ap.add_argument("--coverage-file", default=None, dest="coverage_file",
                help="質問テキストのみ読み、フレーズ支持率を報告 (ラベル不使用)")
args = ap.parse_args()

def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()

tr = json.load(open(args.train, encoding="utf-8"))
qtexts = [norm(t["question"]) for t in tr]
golds = [set(t["golden_doc_ids"]) for t in tr]
q_by_text = {qt: i for i, qt in enumerate(qtexts)}

def support_idxs(phrase, excl):
    p = norm(phrase)
    return [i for i, qt in enumerate(qtexts) if p in qt and i not in excl]

def freq_map(idxs):
    cnt = collections.Counter()
    for i in idxs:
        cnt.update(golds[i])
    n = len(idxs)
    return {d: c / n for d, c in cnt.items()}

def trim_variants(ph):
    """未解決フレーズの縮約: 長い順に with/and/or/,/in/for の頭側を試す"""
    p = norm(ph)
    outs = [p]
    for sep in (" with ", " and ", " or ", ", ", " in ", " for "):
        if sep in p:
            outs.append(p.split(sep)[0])
    return sorted(set(outs), key=len, reverse=True)

def resolve(ph, excl, min_sup):
    """フレーズ→ (freq_map, 使用した変種, 支持数)。解決不能なら (None,None,0)"""
    for v in trim_variants(ph):
        idxs = support_idxs(v, excl)
        if len(idxs) >= min_sup:
            return freq_map(idxs), v, len(idxs)
    return None, None, 0

def dict_members(clauses, excl, theta, min_sup):
    """DNF評価。戻り: (member集合, 未解決クローズのlist)"""
    member = set()
    unresolved = []
    for cl in clauses:
        fms = []
        for ph in cl:
            fm, _, _ = resolve(ph, excl, min_sup)
            if fm is None:
                fms = None
                break
            fms.append(fm)
        if fms is None:
            unresolved.append(cl)
            continue
        docs = set.intersection(*[set(f) for f in fms]) if fms else set()
        member |= {d for d in docs if min(f.get(d, 0) for f in fms) >= theta}
    return member, unresolved

def token_f1(pred, gold):
    P = re.findall(r"[a-z0-9]+", str(pred).lower())
    G = re.findall(r"[a-z0-9]+", str(gold).lower())
    if not P or not G:
        return float(P == G)
    common = collections.Counter(P) & collections.Counter(G)
    o = sum(common.values())
    if o == 0:
        return 0.0
    pr, rc = o / len(P), o / len(G)
    return 2 * pr * rc / (pr + rc)

def aggregate(agg, k, members):
    m = sorted(members)
    if not m:
        return ""
    return {"count": str(len(m)), "min_id": str(m[0]), "max_id": str(m[-1]),
            "sort_asc": ", ".join(map(str, m)),
            "sort_desc": ", ".join(map(str, m[::-1])),
            "topk_smallest": ", ".join(map(str, m[:k or 0])),
            "topk_largest": ", ".join(map(str, sorted(members, reverse=True)[:k or 0]))
            }[agg]

# ---- カバレッジモード: 未知質問集合のフレーズ支持率 (ラベル不使用) ----
if args.coverage_file:
    tq = json.load(open(args.coverage_file, encoding="utf-8"))
    print(f"カバレッジ対象: {len(tq)}問 (質問テキストのみ使用)")
    # 正式パース前の代理: 雑抽出フレーズの支持率でtest汎化を先取り推定
    sup_ok = tot = 0
    for t in tq:
        for m in re.findall(
                r"(?:about|for|are|of|either)\s+([A-Za-z][A-Za-z /&+-]{6,60}?)"
                r"(?:,| or | and | with | in this| among|\?|$)", t["question"]):
            tot += 1
            excl = {q_by_text.get(norm(t["question"]))} - {None}
            if len(support_idxs(m, excl)) >= args.min_sup:
                sup_ok += 1
    print(f"代理フレーズ支持率 (sup>={args.min_sup}): {sup_ok}/{tot} "
          f"= {sup_ok/max(tot,1)*100:.1f}%")
    print("(train60の正式パース版解決率77.4%との比で汎化率を見積もる)")
    raise SystemExit

# ---- 評価モード: parsedファイルの全問を辞書のみで解く ----
parsed = [json.loads(l) for l in open(args.parsed, encoding="utf-8") if l.strip()]
res = []
for p in parsed:
    i = q_by_text.get(norm(p["question"]))
    excl = {i} if i is not None else set()   # 自問除外 (test時は自動的に空)
    gold = set(p["golden_doc_ids"])
    member, unresolved = dict_members(p["parsed"]["clauses"], excl,
                                      args.theta, args.min_sup)
    mf = 2 * len(member & gold) / max(len(member) + len(gold), 1)
    af = token_f1(aggregate(p["parsed"]["agg"], p["parsed"].get("k"), member),
                  p["answer"])
    res.append({"idx": p["idx"], "task": p["parsed"]["agg"], "mF": mf, "aF": af,
                "n_member": len(member), "n_gold": len(gold),
                "n_unresolved_clauses": len(unresolved)})

by = collections.defaultdict(list)
for r in res:
    by[r["task"]].append(r)
print(f"=== step35 辞書のみ (θ={args.theta}, sup>={args.min_sup}, n={len(res)}) ===")
print(f"{'task':>13} | {'n':>2} | {'memberF1':>8} | {'answerF1':>8}")
for t, rs in sorted(by.items()):
    print(f"{t:>13} | {len(rs):>2} | {np.mean([r['mF'] for r in rs])*100:7.1f}% "
          f"| {np.mean([r['aF'] for r in rs])*100:7.1f}%")
print(f"{'ALL':>13} | {len(res):>2} | {np.mean([r['mF'] for r in res])*100:7.1f}% "
      f"| {np.mean([r['aF'] for r in res])*100:7.1f}%")
with open("results_step35_dict.jsonl", "w", encoding="utf-8") as f:
    for r in res:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("results_step35_dict.jsonl に保存")

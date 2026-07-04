# -*- coding: utf-8 -*-
"""
step5_verify_v1.py — CORE v1: Aggregation-aware selective verification
=======================================================================
中核アイデア:
  述語マッチを3値 (IN / OUT / UNCERTAIN) で実体化し、
  「UNCERTAIN かつ 集約結果に影響する文書」だけを gpt-5-mini で Yes/No 精査する。
  検証予算の配分は集約演算子の感度に従う:
    - min_id / max_id : 極値側の境界候補をID端から逐次確認 (確定1件で即停止)
    - topk_*          : 端から k 件確定するまで
    - count / sort_*  : 境界帯 (θ±マージン) の文書を曖昧度順に解消

運用ルール準拠:
  - 冪等: 検証結果は verify_cache/verify_v1.jsonl に追記キャッシュ (述語×文書キー)。
    再実行しても課金済みの判定は再利用される。
  - APIキーは環境変数 OPENAI_API_KEY のみ。
  - --dry-run で LLM を一切呼ばずに帯サイズ・予想呼び出し数を確認できる。

使い方 (推奨順):
  1) step3_eval.py に `if __name__ == "__main__":` ガードを入れる (import副作用の分離)
  2) python step5_verify_v1.py --split train60 --dry-run   # 帯サイズと予算感の確認
  3) python step5_verify_v1.py --split train60             # 閾値/予算のチューニング
  4) python step5_verify_v1.py --split test100             # 本評価
"""

import os
import json
import argparse
import hashlib
from collections import defaultdict, Counter

import numpy as np
from openai import OpenAI

from util import with_retry

# =====================================================================
# ADAPT セクション: 既存資産との接続点はすべてここに集約
# 実リポジトリのスキーマに合わせてこのセクションだけ直せば動く想定
# =====================================================================

EMB_PATH = "chunk_embs.npy"          # (3116, 1536) 想定
OWNER_PATH = "chunk_owner.json"      # chunk index -> doc_id の対応
CORPUS_PATH = "./data/GlobalQA/corpus.jsonl"
PARSED = {
    "test100": "parsed_test100.jsonl",
    "train60": "parsed_train60.jsonl",
}
VERIFY_CACHE_DIR = "verify_cache"
RESULTS_DIR = "results_v1"

VERIFY_MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"

# --- 校正済み点閾値 (step3, train60でdoc-F1最大化) ---
THETA = 0.50
# --- 3値化の帯: [LOW, HIGH) が UNCERTAIN。train60 で要チューニング ---
#   HIGH を上げる → min/max の低ID偽陽性をより多く帯に落とせる (呼び出し増)
#   LOW  を下げる → 真の極値文書の取りこぼし (偽陰性) をより多く救える (呼び出し増)
HIGH_THETA = 0.56
LOW_THETA = 0.42

# --- 演算子別の LLM 判定予算 (1クエリあたりの文書数上限) ---
BUDGET = {
    "min_id": 15,
    "max_id": 15,
    "topk_largest": 25,
    "topk_smallest": 25,
    "count": 30,
    "sort_asc": 30,
    "sort_desc": 30,
}


def get_program(rec: dict):
    """parsed_*.jsonl の1レコードから (qid, op, k, expr, gold) を取り出す。

    [ADAPT済 2026-07-04] step2_parse.py の実出力スキーマに合わせて確定:
      {"idx", "question", "answer", "golden_doc_ids",
       "parsed": {"predicates": [...], "combine": "or"|"and", "agg", "k"}}
    combine はクエリごとに or/and が変わるため固定してはならない。
    """
    qid = rec["idx"]
    p = rec["parsed"]
    op = p["agg"]                       # min_id/max_id/count/sort_*/topk_* (step5と一致)
    k = p.get("k")
    expr = {p["combine"]: list(p["predicates"])}
    gold = rec["answer"]
    return qid, op, k, expr, gold


def load_chunk_texts(corpus_path: str, n_expected: int):
    """chunk index -> テキスト のリストを復元する。

    ADAPT: step1_embed.py の分割ロジックと完全一致させること。
    step1 に分割関数があるなら import で流用するのが最も安全:
        from step1_embed import build_chunks
    復元後 len(chunks) == n_expected (=3116) を必ず assert する。
    """
    # [ADAPT済 2026-07-04] step1_embed.py の分割はインライン実装のため同一ロジックを再現:
    #   contents を CHUNK=6000 文字で機械分割 (step1と値を変えないこと)
    CHUNK = 6000
    docs = [json.loads(l) for l in open(corpus_path, encoding="utf-8") if l.strip()]
    chunks = []
    for d in docs:
        t = d["contents"]
        for i in range(0, len(t), CHUNK):
            chunks.append(t[i:i + CHUNK])
    assert len(chunks) == n_expected, (
        f"chunk数不一致: {len(chunks)} != {n_expected} (step1と分割が一致していない)"
    )
    return chunks  # list[str], index が chunk_embs.npy と対応


def import_metrics():
    """[ADAPT済 2026-07-04] step3 の実関数は token_f1(pred, gold) のみ。
    step3_eval.py に __main__ ガードが入っていることが前提 (入っていないと
    import 時に校正ループが走る)。"""
    from step3_eval import token_f1
    return token_f1, token_f1


def format_answer(op: str, result):
    """集約結果 -> gold と比較可能な文字列/リスト表現。

    ADAPT: step3_eval.py の集約器・整形と同一にすること (token F1が整形に敏感なため)。
    可能なら step3 の集約器をそのまま import して置き換えるのが望ましい。
    """
    if op == "count":
        return str(result)
    if op in ("min_id", "max_id"):
        return str(result) if result is not None else ""
    return ", ".join(str(x) for x in result)  # [ADAPT済] v0(step3)と同じ", "結合

# =====================================================================
# ここから下は原則そのまま使える想定
# =====================================================================

client = OpenAI()

IN, OUT, UNC = 1, 0, None  # 3値


# ---------------------------------------------------------------
# 述語embedding (step3のpred_cacheと衝突しない独自キャッシュ)
# ---------------------------------------------------------------
_pred_emb_cache = {}
_PRED_EMB_PATH = os.path.join(VERIFY_CACHE_DIR, "pred_embs_v1.jsonl")


def _load_pred_emb_cache():
    if os.path.exists(_PRED_EMB_PATH):
        with open(_PRED_EMB_PATH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _pred_emb_cache[r["pred"]] = np.array(r["emb"], dtype=np.float32)


def embed_predicate(pred: str) -> np.ndarray:
    if pred in _pred_emb_cache:
        return _pred_emb_cache[pred]
    resp = with_retry(lambda: client.embeddings.create(model=EMBED_MODEL, input=pred))
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    _pred_emb_cache[pred] = v
    os.makedirs(VERIFY_CACHE_DIR, exist_ok=True)
    with open(_PRED_EMB_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"pred": pred, "emb": v.tolist()}) + "\n")
    return v


# ---------------------------------------------------------------
# LLM検証 (述語×文書, ディスクキャッシュで冪等)
# ---------------------------------------------------------------
_verify_cache = {}
_VERIFY_PATH = os.path.join(VERIFY_CACHE_DIR, "verify_v1.jsonl")

VERIFY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "predicate_check",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"satisfies": {"type": "boolean"}},
            "required": ["satisfies"],
            "additionalProperties": False,
        },
    },
}


def _vkey(pred: str, doc_id) -> str:
    return hashlib.sha1(f"{pred}\x00{doc_id}".encode("utf-8")).hexdigest()


def _load_verify_cache():
    if os.path.exists(_VERIFY_PATH):
        with open(_VERIFY_PATH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                _verify_cache[r["key"]] = r["satisfies"]


def llm_verify(pred: str, doc_id, chunk_text: str, stats: Counter) -> bool:
    """文書(該当チャンクのみ)が述語を満たすか gpt-5-mini でYes/No判定。"""
    key = _vkey(pred, doc_id)
    if key in _verify_cache:
        stats["cache_hit"] += 1
        return _verify_cache[key]
    stats["llm_calls"] += 1
    msgs = [
        {
            "role": "system",
            "content": (
                "You judge whether a document satisfies a predicate. "
                "Base your judgment strictly on the excerpt. "
                "Answer false unless the excerpt clearly satisfies the predicate."
            ),
        },
        {
            "role": "user",
            "content": f"Predicate: {pred}\n\nDocument excerpt:\n{chunk_text}",
        },
    ]
    resp = with_retry(
        lambda: client.chat.completions.create(
            model=VERIFY_MODEL, messages=msgs, response_format=VERIFY_SCHEMA
        )
    )
    ans = bool(json.loads(resp.choices[0].message.content)["satisfies"])
    _verify_cache[key] = ans
    os.makedirs(VERIFY_CACHE_DIR, exist_ok=True)
    with open(_VERIFY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "pred": pred, "doc_id": doc_id,
                            "satisfies": ans}) + "\n")
    return ans


# ---------------------------------------------------------------
# 述語式の評価 (3値 Kleene 論理)
# ---------------------------------------------------------------
def leaf_predicates(expr):
    if isinstance(expr, str):
        yield expr
    elif isinstance(expr, dict):
        for sub in next(iter(expr.values())):
            yield from leaf_predicates(sub)
    else:
        raise ValueError(f"未知の述語式: {expr!r}")


def eval_expr(expr, leaf_val):
    """leaf_val: pred文字列 -> IN/OUT/UNC。Kleene 3値で合成。"""
    if isinstance(expr, str):
        return leaf_val[expr]
    (op, subs), = expr.items()
    vals = [eval_expr(s, leaf_val) for s in subs]
    if op == "and":
        if OUT in vals:
            return OUT
        return UNC if UNC in vals else IN
    if op == "or":
        if IN in vals:
            return IN
        return UNC if UNC in vals else OUT
    raise ValueError(f"未知の結合子: {op}")


def soft_score(expr, leaf_sim):
    """フォールバック/優先順位付け用の連続スコア (and=min, or=max)。"""
    if isinstance(expr, str):
        return leaf_sim[expr]
    (op, subs), = expr.items()
    scores = [soft_score(s, leaf_sim) for s in subs]
    return min(scores) if op == "and" else max(scores)


# ---------------------------------------------------------------
# クエリ1件の処理
# ---------------------------------------------------------------
class Materializer:
    def __init__(self, chunk_embs, chunk_owner, chunk_texts, dry_run=False):
        self.embs = chunk_embs / (
            np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-12
        )
        self.owner = chunk_owner            # list: chunk idx -> doc_id
        self.texts = chunk_texts
        self.dry_run = dry_run
        # doc_id -> そのdocのchunk index列
        self.doc_chunks = defaultdict(list)
        for ci, d in enumerate(chunk_owner):
            self.doc_chunks[d].append(ci)
        self.doc_ids = sorted(self.doc_chunks.keys())

    def predicate_sims(self, pred: str):
        """述語との文書スコア = そのdocのチャンク類似度の最大値。bestチャンクも返す。"""
        pv = embed_predicate(pred)
        sims = self.embs @ pv               # (n_chunks,)
        doc_sim, doc_best = {}, {}
        for d in self.doc_ids:
            idxs = self.doc_chunks[d]
            local = sims[idxs]
            j = int(np.argmax(local))
            doc_sim[d] = float(local[j])
            doc_best[d] = idxs[j]
        return doc_sim, doc_best

    def resolve_doc(self, d, expr, leaf_val, doc_leaf_sim, doc_best, stats):
        """doc d の UNCERTAIN な葉述語をLLMで解消し、3値膜性を再評価して返す。"""
        for p in set(leaf_predicates(expr)):
            if leaf_val[p][d] is UNC:
                if self.dry_run:
                    stats["would_call"] += 1
                    # dry-run では点閾値で仮判定
                    leaf_val[p][d] = IN if doc_leaf_sim[p][d] >= THETA else OUT
                else:
                    ok = llm_verify(p, d, self.texts[doc_best[p][d]], stats)
                    leaf_val[p][d] = IN if ok else OUT
        return eval_expr(expr, {p: leaf_val[p][d] for p in leaf_val})

    def run_query(self, op, k, expr, stats):
        preds = list(set(leaf_predicates(expr)))
        doc_leaf_sim, doc_best = {}, {}
        for p in preds:
            doc_leaf_sim[p], doc_best[p] = self.predicate_sims(p)

        # 葉ごとの3値化
        leaf_val = {
            p: {
                d: (IN if s >= HIGH_THETA else OUT if s < LOW_THETA else UNC)
                for d, s in doc_leaf_sim[p].items()
            }
            for p in preds
        }
        status = {
            d: eval_expr(expr, {p: leaf_val[p][d] for p in preds})
            for d in self.doc_ids
        }
        soft = {
            d: soft_score(expr, {p: doc_leaf_sim[p][d] for p in preds})
            for d in self.doc_ids
        }
        stats["band_size"] += sum(1 for v in status.values() if v is UNC)

        budget = BUDGET.get(op, 20)
        confirmed = {d for d, v in status.items() if v is IN}
        uncertain = {d for d, v in status.items() if v is UNC}

        def spend_and_resolve(d):
            nonlocal budget
            budget -= 1
            v = self.resolve_doc(d, expr, leaf_val, doc_leaf_sim, doc_best, stats)
            status[d] = v
            uncertain.discard(d)
            if v is IN:
                confirmed.add(d)
            return v

        # ---- 演算子別ポリシー ----
        if op in ("min_id", "max_id"):
            rev = op == "max_id"
            for d in sorted(confirmed | uncertain, reverse=rev):
                if status[d] is IN:
                    return format_answer(op, d)  # ID端の確定文書 = 答え
                if budget <= 0:
                    break
                if spend_and_resolve(d) is IN:
                    return format_answer(op, d)
            # 予算枯渇/全滅: 点閾値フォールバック
            rest = [d for d in self.doc_ids
                    if status[d] is IN or (status[d] is UNC and soft[d] >= THETA)]
            ans = (max(rest) if rev else min(rest)) if rest else None
            return format_answer(op, ans)

        if op in ("topk_largest", "topk_smallest"):
            rev = op == "topk_largest"
            out = []
            for d in sorted(confirmed | uncertain, reverse=rev):
                if status[d] is UNC:
                    if budget <= 0:
                        if soft[d] >= THETA:  # フォールバック
                            out.append(d)
                    elif spend_and_resolve(d) is not IN:
                        continue
                if status[d] is IN or (budget <= 0 and d in out[-1:]):
                    if d not in out:
                        out.append(d)
                if len(out) >= (k or 1):
                    break
            return format_answer(op, out[: (k or 1)])

        # count / sort_*: 帯を曖昧度順 (|sim-θ| 昇順) に解消
        for d in sorted(uncertain.copy(), key=lambda d: abs(soft[d] - THETA)):
            if budget <= 0:
                break
            spend_and_resolve(d)
        member = [d for d in self.doc_ids
                  if status[d] is IN or (status[d] is UNC and soft[d] >= THETA)]
        if op == "count":
            return format_answer(op, len(member))
        if op == "sort_asc":
            return format_answer(op, sorted(member))
        if op == "sort_desc":
            return format_answer(op, sorted(member, reverse=True))
        raise ValueError(f"未知の演算子: {op}")


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    global HIGH_THETA, LOW_THETA
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(PARSED), default="test100")
    ap.add_argument("--dry-run", action="store_true",
                    help="LLMを呼ばず帯サイズと予想呼び出し数のみ算出")
    ap.add_argument("--high", type=float, default=HIGH_THETA)
    ap.add_argument("--low", type=float, default=LOW_THETA)
    args = ap.parse_args()

    HIGH_THETA, LOW_THETA = args.high, args.low

    _load_pred_emb_cache()
    _load_verify_cache()

    embs = np.load(EMB_PATH)
    with open(OWNER_PATH, encoding="utf-8") as f:
        owner = json.load(f)
    texts = load_chunk_texts(CORPUS_PATH, n_expected=len(owner))
    mat = Materializer(embs, owner, texts, dry_run=args.dry_run)

    answer_f1, token_f1 = import_metrics()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR,
        f"v1_{args.split}{'_dry' if args.dry_run else ''}.jsonl",
    )
    # 冪等: 処理済みqidはスキップ
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["qid"] for l in f}

    per_task = defaultdict(list)
    calls_per_task = defaultdict(list)
    with open(PARSED[args.split], encoding="utf-8") as f, \
         open(out_path, "a", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            qid, op, k, expr, gold = get_program(rec)
            if qid in done:
                continue
            stats = Counter()
            pred = mat.run_query(op, k, expr, stats)
            f1 = answer_f1(pred, gold)
            per_task[op].append(f1)
            calls_per_task[op].append(
                stats["llm_calls"] + stats["would_call"]
            )
            out.write(json.dumps({
                "qid": qid, "op": op, "pred": pred, "gold": gold, "f1": f1,
                "llm_calls": stats["llm_calls"],
                "would_call": stats["would_call"],
                "cache_hit": stats["cache_hit"],
                "band_size": stats["band_size"],
            }, ensure_ascii=False) + "\n")
            out.flush()

    # ---- サマリ ----
    print(f"\n=== v1 {args.split} {'(dry-run)' if args.dry_run else ''} ===")
    print(f"θ={THETA}  band=[{LOW_THETA}, {HIGH_THETA})")
    all_f1, all_calls = [], []
    for op in sorted(per_task):
        fs, cs = per_task[op], calls_per_task[op]
        all_f1 += fs
        all_calls += cs
        print(f"{op:15s}  n={len(fs):3d}  F1={np.mean(fs)*100:6.2f}"
              f"  calls/q={np.mean(cs):5.1f}")
    if all_f1:
        print(f"{'ALL':15s}  n={len(all_f1):3d}  F1={np.mean(all_f1)*100:6.2f}"
              f"  calls/q={np.mean(all_calls):5.1f}"
              f"  total_calls={sum(all_calls)}")
    print(f"結果: {out_path}")


if __name__ == "__main__":
    main()

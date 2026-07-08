# step25c_parse_test.py - test全問の並列パース (step25と同一プロンプト/モデル/スキーマ)
#   変更点: 入力 test (3).json (質問本文のみ使用) / 出力 parsed2_test.jsonl /
#   ThreadPoolExecutorで並列 (--workers) / idx = 200000+位置
#   【重要】idxオフセットの理由: 採点キャッシュのキーは (pv, idx, doc)。testのidxが
#   trainのidxと重なると、別質問のスコアを誤って再利用する事故が起きる。200000+で名前空間分離。
#   slice2裁定 (P1g) と並行実行してよい: testのラベルは読み込むが判定には一切使わず、
#   パースは質問本文のみを見る。エンジン実行 (step34f) はP1g合格まで禁止。
#   コスト: ~$1 (gpt-5-mini, 1160問)。冪等 (中断→再実行で続きから)
import json, os, argparse, threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from util import with_retry

client = OpenAI(timeout=120)
MODEL = "gpt-5-mini"
SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "program", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "agg": {"type": "string",
                    "enum": ["count", "min_id", "max_id", "sort_asc",
                             "sort_desc", "topk_largest", "topk_smallest"]},
            "k": {"type": ["integer", "null"]},
            "clauses": {"type": "array", "items": {
                "type": "array", "items": {"type": "string"}}}},
        "required": ["agg", "k", "clauses"], "additionalProperties": False}}}

SYS = """You compile a corpus-level query over a resume corpus into an aggregation program.

Output:
- agg: the aggregation. "biggest/most recent document ID" = max_id; "smallest/earliest" = min_id; "how many" = count; "sort ascending/descending" = sort_asc/sort_desc; "top k largest/smallest IDs" = topk_largest/topk_smallest (set k; else k=null).
- clauses: the document-selection condition in DISJUNCTIVE NORMAL FORM — a list of AND-groups; a document qualifies if ALL predicates in AT LEAST ONE group describe it. Each predicate is one self-contained description of a person/resume. NEVER put 'or'/'OR' inside a predicate string.

Boolean cues (follow these strictly):
- "both X and Y", "who are also Z", "and also include Z" -> Z joins the SAME AND-group: [[X, Y]] / distribute: "either A or B, and also C" -> [[A, C], [B, C]].
- "either X or Y", "X or Y" -> separate OR groups: [[X], [Y]].
- Topic enumerations like "documents related to X and Y" or "about X and Y" (coverage of multiple topics, not one person having both) -> separate OR groups: [[X], [Y]].
- Nested: "A with B and C, or D" -> [[A-with-B-and-C as its natural conjuncts], [D]]; split A/B/C into separate predicates within the group only if they are independent conditions.
- A single condition -> [[X]].

Keep predicate wording close to the question's own words (they are used for lexical retrieval)."""

def parse_one(q):
    r = with_retry(lambda: client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": q}],
        response_format=SCHEMA))
    return json.loads(r.choices[0].message.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default="./data/GlobalQA/test (3).json", dest="test_file")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tq = json.load(open(args.test_file, encoding="utf-8"))
    rows = [{"idx": 200000 + i, "question": t["question"], "answer": t["answer"],
             "golden_doc_ids": t["golden_doc_ids"]} for i, t in enumerate(tq)]
    out_path = "parsed2_test.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["idx"] for l in open(out_path, encoding="utf-8") if l.strip()}
    todo = [r for r in rows if r["idx"] not in done]
    print(f"test {len(rows)}問, 済{len(done)}, 残り{len(todo)}")

    lock = threading.Lock()
    out = open(out_path, "a", encoding="utf-8")
    n = [0]

    def work(r):
        p = parse_one(r["question"])
        if not p["clauses"] or not all(p["clauses"]):
            p = parse_one(r["question"])
        if not p["clauses"] or not all(p["clauses"]):
            p["clauses"] = [[r["question"]]]
            flag = " [警告:単一句化]"
        else:
            flag = ""
        rec = {"idx": r["idx"], "question": r["question"], "answer": r["answer"],
               "golden_doc_ids": r["golden_doc_ids"], "parsed": p}
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n[0] += 1
            if n[0] % 25 == 0 or flag:
                print(f"  {n[0]}/{len(todo)} 完了{flag}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print(f"パース完了: parsed2_test.jsonl ({len(rows)}問)")

if __name__ == "__main__":
    main()

# step25b_parse_slice2.py - step25のslice2版 (第2標本検証用, HANDOFF v7 §4-1)
#   プロンプト・モデル・スキーマはstep25と完全同一。変更は入出力のみ:
#   入力 slice2_sample.jsonl (生の質問, 旧パースなし) / 出力 parsed2_slice2.jsonl
#   旧パースが無いため空clauseフォールバックは「1回再試行→単一句化+警告」に置換
# 元: step25_parse2.py - クエリの再コンパイル
#   例: "either A or B, and also C"        -> [[A, C], [B, C]]
#       "related to X and Y" (トピック列挙) -> [[X], [Y]]
#       "A with B and C, or D"             -> [[A, B, C], [D]]
#       "both X and Y"                     -> [[X, Y]]
#   コスト: train60で ~$0.05 (gpt-5-mini)。冪等 (JSONL追記式)
import json, os, argparse
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
    rows = [json.loads(l) for l in
            open("slice2_sample.jsonl", encoding="utf-8") if l.strip()]
    out_path = "parsed2_slice2.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["idx"] for l in open(out_path, encoding="utf-8")
                if l.strip()}

    with open(out_path, "a", encoding="utf-8") as out:
        for r in rows:
            if r["idx"] in done:
                continue
            p = parse_one(r["question"])
            if not p["clauses"] or not all(p["clauses"]):
                print(f"  [警告] qid{r['idx']}: 空のclause -> 再試行")
                p = parse_one(r["question"])
            if not p["clauses"] or not all(p["clauses"]):
                print(f"  [警告] qid{r['idx']}: 依然空 -> 質問全文を単一句化 (要目視)")
                p["clauses"] = [[r["question"]]]
            rec = {"idx": r["idx"], "question": r["question"],
                   "answer": r["answer"],
                   "golden_doc_ids": r["golden_doc_ids"], "parsed": p}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            cl = " OR ".join("(" + " AND ".join(x[:40] for x in c) + ")"
                             for c in p["clauses"])
            print(f"[qid {r['idx']}] {p['agg']:14s} {cl[:140]}")

    print(f"パース完了: {len(rows)}問 -> {out_path}")

if __name__ == "__main__":
    main()

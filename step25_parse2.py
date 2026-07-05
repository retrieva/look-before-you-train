# step25_parse2.py - クエリの再コンパイル: フラットな{predicates, combine}から
#                    DNF (AND句のOR) へ表現力を拡張 (step24の発見に基づく)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train60", choices=["train60", "test100"])
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(f"parsed_{args.split}.jsonl", encoding="utf-8") if l.strip()]
    out_path = f"parsed2_{args.split}.jsonl"
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
                print(f"  [警告] qid{r['idx']}: 空のclause -> 旧パースで代替")
                p = {"agg": r["parsed"]["agg"], "k": r["parsed"].get("k"),
                     "clauses": [[x] for x in r["parsed"]["predicates"]]
                     if r["parsed"]["combine"] == "or"
                     else [r["parsed"]["predicates"]]}
            rec = {"idx": r["idx"], "question": r["question"],
                   "answer": r["answer"],
                   "golden_doc_ids": r["golden_doc_ids"], "parsed": p}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            cl = " OR ".join("(" + " AND ".join(x[:40] for x in c) + ")"
                             for c in p["clauses"])
            print(f"[qid {r['idx']}] {p['agg']:14s} {cl[:140]}")

    # 旧パースとの差分サマリ
    n_diff = 0
    old = {r["idx"]: r for r in rows}
    for l in open(out_path, encoding="utf-8"):
        rec = json.loads(l)
        o = old[rec["idx"]]["parsed"]
        flat_old = ([[x] for x in o["predicates"]] if o["combine"] == "or"
                    else [o["predicates"]])
        if rec["parsed"]["clauses"] != flat_old:
            n_diff += 1
    print(f"\n構造が旧パースから変わったクエリ: {n_diff}/{len(rows)}")

if __name__ == "__main__":
    main()

# step2_parse.py - parse queries into programs (checkpointed JSONL output)
import json, os, random, collections
from openai import OpenAI
from util import with_retry

client = OpenAI()
DATA = "./data/GlobalQA"
MODEL = "gpt-5-mini"  # if this errors, run client.models.list() to find current mini name

SCHEMA = {"type": "object", "properties": {
    "predicates": {"type": "array", "items": {"type": "string"},
        "description": "atomic topic conditions, verbatim from the question"},
    "combine": {"type": "string", "enum": ["or", "and"]},
    "agg": {"type": "string", "enum":
        ["min_id", "max_id", "count", "topk_largest", "topk_smallest",
         "sort_asc", "sort_desc"]},
    "k": {"type": ["integer", "null"]}},
    "required": ["predicates", "combine", "agg", "k"],
    "additionalProperties": False}

def parse(q):
    r = with_retry(lambda: client.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content":
         "Parse the corpus-level query into a program. Documents are resumes with "
         "integer IDs. 'most recent' means largest ID, 'earliest' means smallest ID. "
         "Extract each atomic topic condition as one predicate string."},
        {"role": "user", "content": q}],
        response_format={"type": "json_schema", "json_schema":
            {"name": "parsed", "schema": SCHEMA, "strict": True}}))
    return json.loads(r.choices[0].message.content)

def classify(q):
    q = q.lower()
    if "how many" in q: return "count"
    if "top" in q: return "topk"
    if "ascending" in q or "descending" in q or "sort" in q: return "sort"
    return "minmax"

def stratified(rows, per_type, seed):
    random.seed(seed)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows): by[classify(r["question"])].append(i)
    return sorted(sum([random.sample(v, min(per_type, len(v))) for v in by.values()], []))

for split, fname, per_type, out in [
        ("test", "test (3).json", 25, "parsed_test100.jsonl"),
        ("train", "train (1).json", 15, "parsed_train60.jsonl")]:
    rows = json.load(open(f"{DATA}/{fname}", encoding="utf-8"))
    if isinstance(rows, dict): rows = rows.get("data", list(rows.values())[0])
    idxs = stratified(rows, per_type, seed=42)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["idx"] for l in open(out, encoding="utf-8") if l.strip()}
    with open(out, "a", encoding="utf-8") as f:
        for n, i in enumerate(idxs):
            if i in done: continue
            p = parse(rows[i]["question"])
            f.write(json.dumps({"idx": i, **rows[i], "parsed": p}) + "\n")
            f.flush()
            print(f"{split} {n+1}/{len(idxs)}: {p['agg']}, {len(p['predicates'])} preds")
print("done")

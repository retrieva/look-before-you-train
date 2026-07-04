# step9_extract.py - 全履歴書の構造化プロファイル抽出 (v2の土台)
#   1回だけ実行して全クエリで償却する。目安: $1-2 / 20-40分 (4並列)
#   冪等: profiles.jsonl に1件ずつ追記。中断しても再実行で続きから (二重課金なし)
import json, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from util import with_retry

client = OpenAI()
DATA = "./data/GlobalQA"
OUT = "profiles.jsonl"
MODEL = "gpt-5-mini"
MAX_CHARS = 24000          # 超長文履歴書は先頭24k字に切る (中央値6k字なので影響は僅少)
WORKERS = 4                # Tier1のRPM/TPMに触れたらwith_retryが自動で待つ

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "resume_profile",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "titles": {"type": "array", "items": {"type": "string"},
                           "description": "all job titles/roles held"},
                "skills": {"type": "array", "items": {"type": "string"},
                           "description": "skills, competencies, methodologies"},
                "tools": {"type": "array", "items": {"type": "string"},
                          "description": "software, tools, platforms, languages"},
                "domains": {"type": "array", "items": {"type": "string"},
                            "description": "industries/domains worked in"},
                "certifications": {"type": "array", "items": {"type": "string"}},
                "years_experience": {"type": ["number", "null"]},
                "notable": {"type": "array", "items": {"type": "string"},
                            "description": "notable achievements, metrics, "
                                           "special experiences (e.g. retention rates, "
                                           "team sizes, budgets)"},
                "summary": {"type": "string",
                            "description": "2-3 sentence factual summary"},
            },
            "required": ["titles", "skills", "tools", "domains",
                         "certifications", "years_experience", "notable", "summary"],
            "additionalProperties": False,
        },
    },
}

SYS = ("Extract a structured profile from this resume. Be exhaustive with titles, "
       "skills, tools and domains - downstream systems match arbitrary predicates "
       "against this profile, so include specifics (versions, platforms, metrics, "
       "percentages) verbatim where present. Do not invent facts.")

def extract(doc):
    text = doc["contents"][:MAX_CHARS]
    r = with_retry(lambda: client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": text}],
        response_format=SCHEMA))
    prof = json.loads(r.choices[0].message.content)
    return {"id": doc["id"], **prof}

def main():
    docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl", encoding="utf-8")
            if l.strip()]
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [d for d in docs if d["id"] not in done]
    print(f"total={len(docs)}  done={len(done)}  todo={len(todo)}")
    if not todo:
        print("all done"); return

    lock = threading.Lock()
    n_done = len(done)
    with open(OUT, "a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(extract, d): d["id"] for d in todo}
        for fut in as_completed(futures):
            prof = fut.result()
            with lock:
                f.write(json.dumps(prof, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                if n_done % 20 == 0 or n_done == len(docs):
                    print(f"extracted {n_done}/{len(docs)}", flush=True)

if __name__ == "__main__":
    main()

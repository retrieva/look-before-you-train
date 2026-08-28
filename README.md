# Look Before You Train

Code, calibration caches, and per-question results for the paper:

> **Look Before You Train: Train–Test Contamination in GlobalQA and a Training-Free Method that Reverse-Engineers Its Gold Rules**
> Takao Morita (Retrieva, Inc.), 2026. Preprint DOI: [10.5281/zenodo.22147771](https://doi.org/10.5281/zenodo.22147771)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22147771.svg)](https://doi.org/10.5281/zenodo.22147771)

**TL;DR.** 1,073 of the 1,160 GlobalQA test questions (92.5%) appear verbatim in the training
set with identical answers and identical gold document sets; a trivial train-lookup baseline
scores 92.5 answer-F1, ~3x the published RL state of the art (MARAG-R1 14B: 31.22). The gold
labels contain a construction-pipeline component that content-based judging cannot reach but
training labels reveal. A training-free system (DNF compilation + a phrase-to-document-pool
dictionary estimated from training labels + an LLM-judge fallback) scores **48.8** answer-F1
under a twin-exclusion protocol, and **22.0** on the 87 uncontaminated test questions.

## Where each number in the paper comes from

| Paper number | File / procedure |
|---|---|
| Hybrid 48.8 (full test), 51.0 (twin-excluded), 22.0 (clean 87) | `results_FINAL_test_v30f_48.8.jsonl` — split rows by the clean-87 index list (below); `idx` = 200000 + 0-based test position |
| Dictionary-only 45.7 / 48.6 / 9.8 | `results_v30f_test_dictonly_45.7.jsonl` (frozen config, judge disabled). Also exactly derivable from `results_FINAL_...` by zeroing every question with `n_dict == 0`, since the v3.0f judge layer only activates when the dictionary returns zero members |
| Content-only verifier 17.7 (clean 87) | `v25_clean87_gpt_m30.jsonl` |
| Tuning slice 42.0 / fresh slice 34.6 | `results_v30f_train60_slice2.jsonl` (two samples in one file; split by `idx` against `parsed2_train60.jsonl`) |
| Judge calibration (928 pairs; perfect-scoring non-gold rates 37.5% / 45.8% / ...) | `calib_cache.jsonl` (`pv=q2b-gpt`, binary round `pv=q1-gpt`) + `step33b_scorejudge2.py --sweep-only` (zero API cost) |
| Contamination 1,073/1,160; lookup 92.5 | 10-line check in Appendix C of the paper (normalize whitespace/case, exact question-text match) |
| Pre-registered predictions | `preqa_20260707.md` |

The clean-87 index list (0-based positions in the official test file) is distributed as an
ancillary file with the arXiv submission and can be regenerated with the Appendix C snippet.

## Repository layout

- **Current entry points**: `step34f_v30f.py` (frozen v3.0f engine; `--no-judge` for the
  dictionary-only ablation), `step25c_parse_test.py` (DNF parsing), `step33b_scorejudge2.py`
  (judge calibration), `step34g_shard.py` (sharded test run).
- `step1`–`step35` (everything else): the development history, kept intact on purpose — the
  paper's pre-registration discipline (predict → pay → adjudicate) is documented in these
  files' header comments. They are superseded and should not be used for reproduction.
- **Note on `step35_dictmember.py`**: this is an early-generation dictionary (substring
  matching, no meta-word stripping, no pool cap). The paper's dictionary-only numbers come
  from the frozen engine (`step34f_v30f.py --no-judge`), **not** from step35.
  `results_step35_dict.jsonl` is the legacy step35 output, kept for the record only.
- `parsed2_*.jsonl`: cached DNF parses (train60 tuning slice, fresh slice, full test,
  clean 87). Reusing them makes reproduction cheap.
- `calib_cache*.jsonl`, `bge_ranks.json`: cached LLM judge scores and BGE retrieval ranks.

## Reproduction

```bash
pip install -r requirements.txt          # or requirements-lock.txt for exact versions
export OPENAI_API_KEY=...                # only needed for judge-layer reproduction
```

Data: download GlobalQA from https://huggingface.co/datasets/QiiLuoo/GlobalQA and place
`corpus.jsonl`, `train (1).json`, `test (3).json` under `./data/GlobalQA/`. The dataset is
not redistributed here.

Zero-cost reproductions (use shipped caches):

```bash
# Contamination check (Appendix C, ~1 second)
python - << 'EOF'
import json, re
tr = json.load(open("data/GlobalQA/train (1).json")); te = json.load(open("data/GlobalQA/test (3).json"))
n = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
by = {n(t["question"]) for t in tr}
print(sum(1 for t in te if n(t["question"]) in by), "/", len(te))   # -> 1073 / 1160
EOF

# Judge-calibration sweeps (Table 2 etc.)
python step33b_scorejudge2.py --limit 40 --judge gpt --sweep-only

# Dictionary-only test run (no API calls; ~minutes on CPU)
python step34f_v30f.py --parsed parsed2_test.jsonl --n 1160 --no-judge
```

Caveat for the dictionary-only mode: `--no-judge` still *reads* cached judge scores from
`calib_cache.jsonl` if present. The shipped `calib_cache.jsonl` contains train-side
calibration scores only (no test-question entries), so the run above is a pure dictionary
run; we additionally verified it is cell-for-cell identical to zeroing `n_dict == 0`
questions in the full-run results. If you merge the shard caches
(`calib_cache_shard*.jsonl`, which do contain test-question scores) into
`calib_cache.jsonl`, `--no-judge` ceases to be a pure dictionary ablation.

Paid reproduction (full hybrid test run, ~USD 25–40 with gpt-5-mini):

```bash
python step34f_v30f.py --parsed parsed2_test.jsonl --n 1160 --workers 12
# or 3 shards in parallel: step34g_shard.py --shard {0,1,2}/3
```

All scripts are idempotent (interrupt → rerun; scores are cached by
(prompt-version, question, document), so nothing is billed twice).

## Frozen configuration (paper Appendix A)

Dictionary: θ=0.3 (θ_count=0.45), min support 3, pool cap 60, word-boundary matching,
meta-prefix stripping + head-segment truncation, self-exclusion of the (twin) question.
Judge fallback: only when the dictionary yields zero members; gpt-5-mini, 0–10 rubric,
admit ≥8, ≤25 per clause; rescue top-5 (top-k) if still empty. Frozen on 2026-07-07 before
the single test run; no test-informed tuning.

## Citation

```bibtex
@misc{morita2026lookbeforeyoutrain,
  title     = {Look Before You Train: Train--Test Contamination in GlobalQA and a
               Training-Free Method that Reverse-Engineers Its Gold Rules},
  author    = {Morita, Takao},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22147771},
  url       = {https://doi.org/10.5281/zenodo.22147771}
}
```

## License

See `LICENSE`. The GlobalQA dataset itself is subject to its own terms on Hugging Face.

The paper text (main_edited.tex / PDF) is © Takao Morita; the license above applies to the code.

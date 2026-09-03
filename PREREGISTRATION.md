# Pre-registration record (predict → pay → adjudicate)

This file collects, in one place, the predictions that were written down **before** each paid
run during development, and how each was adjudicated. All entries below are transcribed
verbatim from the dated header comments of the step files in this repository (the original
Japanese is kept; an English gloss follows each block). Nothing here was written after the fact.

## Round 1 — v3.0 (`step34_v30.py`, header dated 2026-07-07, "課金前" = before paying)

- P1: train60 ALL answerF1 ≥ 35 (dictionary 32.9 + judge fill on the 22.6 % unresolved → sort/topk +5–10 pt)

## Round 2 — v3.0b (`step34b_v30b.py`, 2026-07-07, before paying)

- P1': 15-question train60 pilot, ALL answerF1 ≥ 45 (fully-resolved dictionary questions preserved, FP flood stopped)
- P2': new judge calls < 300 per 15 questions (down from 2,334)
- P3': idx1759 resolved by the dictionary, aF > 0
- P4': count still < 5 (count calibration deferred to a separate card)

Adjudication (recorded in `step34c/d` headers): pilot 2 ALL = 60.6 (P1' pass) / 125 calls (P2' pass)
/ count 33.3 (P4' missed in the good direction; idx1113 exact count) / P3' failed (form changed).

## Round 3 — v3.0c/d (`step34d_v30d.py`, 2026-07-07)

- P1'': --n 15, ALL ≥ 62 (at least one of 1759/2444 recovers from 0)
- P2'': new calls < 400
- P3'': --n 60, ALL ≥ 45 (dictionary 36.3 + judge fill)
- P1d: --n 15, ALL ≥ 70 (549 restored + 2444 rescue kept)
- P2d: --n 60, ALL ≥ 48

Adjudication (recorded in `step34f` header): train60 ALL 39.6 → **P2d failed**. Split accounting:
dictionary-only group aF 69.7 (87.4 excl. count) / judge-involved group 19.0 / rescue 20.0.
Counterfactual: the judge only helped on dictionary-empty questions (1078: 0→1.00, 10863: 0→0.50)
and hurt partially-resolved ones (6889: 0.92→0.37). This adjudication produced the two v3.0f
changes (judge only when the member set is empty; θ_count = 0.45 from a 126-question held-out
OR-form calibration set).

## Round 4 — v3.0f, frozen configuration (`step34f_v30f.py`, 2026-07-07)

- P1f: train60 ALL ≥ 43
- P2f: new calls < 200 (mostly cache hits)
- P3f: count aF within ±7 pt

Adjudication: train60 (tuning slice) 42.0; configuration frozen.

## Round 5 — fresh slice gate (`step25c_parse_test.py` header)

"エンジン実行 (step34f) はP1g合格まで禁止" — running the frozen engine on the test set was
prohibited until the fresh-slice adjudication (P1g) passed. Fresh slice result: 34.6
(`results_v30f_train60_slice2.jsonl`).

## Round 6 — single test run (registered before the run; adjudicated after)

Source: the project's internal hand-off notes, which were written before and after the test run
on the same day and are the log of record for this project.

- **Plan (hand-off note v7 §4-4, 2026-07-07, written when the fresh-slice gate was set):**
  "test一発勝負 … 実行前チェック: 全ハイパラが§1と一致 / results退避 / 事前予測登録
  (現実的予測: test ALL 33-40。SOTA 31.22超えが目標線)"
  — one-shot test run; pre-run checklist: hyper-parameters match the frozen §1, previous
  results moved aside, prediction registered (realistic prediction: test ALL 33–40; beating
  the published 31.22 is the target line). "testの結果を見た後の再実行・再調整は絶対禁止"
  — re-running or re-tuning after seeing the test result is forbidden.
- **Registered prediction (hand-off note v8 §3, 2026-07-07, after the fresh-slice gate passed
  at 34.6 and before the test run):**
  "事前予測 P1h (登録済み): test ALL >= 30 / SOTA更新線 >= 31.3 / 中心予測 33前後"
  — P1h: test ALL ≥ 30; SOTA-beating line ≥ 31.3; **predicted centre ≈ 33**.
  Cost estimate registered at the same time: ~50k judge calls, $25–40.
- **Adjudication (hand-off note v9 §4, 2026-07-07, after the run):**
  "P1h test>=30/SOTA線31.3/中心33: 48.8で大幅合格、ただし中心予測を+15.8上振れ
  → 原因究明の結果、発見A (汚染) に到達"
  — 48.8: passed by a wide margin, but +15.8 above the registered centre; investigating the
  overshoot led to Finding 1 (train–test contamination).

The three notes are separate files (v7, v8, v9) whose creation order and timestamps are
preserved in the authors' working directory; the sequence plan → prediction → adjudication is
also reflected in `step25c_parse_test.py` (gate) and `step34g_shard.py` (the run itself).

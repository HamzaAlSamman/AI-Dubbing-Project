# Lecture evaluation, same gate for every system

eval_id: `ev_20261124T091231Z_baseline_same_gate` · code: `7ad6d9a` (branch evaluation-v2) · Colab T4
41 units · MetricX-24 hybrid QE (sentence-level for units >= 60 words) · hunspell-ar · real timeline windows

| system | accept% | QE | fits% | speedup% | too_long% | too_short% | cover% |
|---|---|---|---|---|---|---|---|
| final_v1 | 90.2 | 3.216 | 80.5 | 14.6 | 2.4 | 2.4 | 100.0 |
| S7_best_of_12_habibi_gate | 87.8 | 2.434 | 80.5 | 14.6 | 2.4 | 2.4 | 100.0 |
| hy_mt15_7b_nocontext | 73.2 | 2.699 | 65.9 | 17.1 | 17.1 | 0.0 | 100.0 |
| gemini35_flash_lite_api | 61.0 | 4.081 | 80.5 | 0.0 | 2.4 | 17.1 | 100.0 |
| nllb_3b | 51.2 | 4.936 | 65.9 | 24.4 | 4.9 | 4.9 | 100.0 |
| qwen3_8b | 46.3 | 4.209 | 78.0 | 4.9 | 2.4 | 14.6 | 100.0 |
| qwen3_4b (baseline, model of run A) | 34.1 | 6.004 | 53.7 | 7.3 | 17.1 | 22.0 | 100.0 |

Full per-unit rows: Drive `translation_step/evaluations/ev_20261124T091231Z_baseline_same_gate/auto.json` (lecture text stays on Drive).

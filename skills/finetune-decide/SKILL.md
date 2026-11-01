---
name: finetune-decide
description: Evaluator ("Iterate?") - read a fine-tuning eval report and decide retrain with one changed setting, regenerate training data for weak categories, or stop
metadata:
---

# Fine-tune Evaluator: iterate or stop

Reads the eval report from `finetune-eval` and makes **one decision**: `retrain` (change one setting),
`regenerate_data` (ask DataGen for more data in weak categories) or `stop`. The decision comes from fixed rules
(`docs/evaluator-notes.md` in the finetuner repo), not an LLM, so it's free, instant and reproducible. Takes ~1 s.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--eval-report-path` | `report.json` from `finetune-eval` (its output key `eval_report_path`) | Required |
| `--spec` / `--task-spec-path` | `task_spec.json`: reads `targets.*`, `budget.max_gpu_hours`, `scoring.pass_threshold` | spec defaults |
| `--target-score` | Stop when the tuned dev score reaches this (0–1). Overrides the spec | none |
| `--max-loops` / `--min-gain` | Override `targets.max_loops` (3) / `targets.min_gain` (0.02) | from spec |
| `--extra-examples` | Extra training examples to request when regenerating data | 150 |

## Rules (first match wins)

1. **Stop:** held-out split (final) · base few-shot already meets the target in loop 1 (no fine-tuning needed) · target met · loop ≥ max_loops · GPU budget spent · gain over the last loop < `min_gain` or its paired 95% CI includes 0.
2. **Training broken** (loss NaN / not falling): retrain, lr ÷ 3.
3. **Forgetting** (general-ability drop > `max_general_drop`, 3 points): back to r=16 if larger, else epochs − 1, else lr ÷ 2.
4. **Format failures** > 10%: regenerate data.
5. **Failures concentrated** in a minority of categories (≥ 0.15 below the overall score): regenerate data for those category names (never test items).
6. **Overfitting** (eval loss rising): epochs − 1, else lr ÷ 2.
7. **Underfitting** (train loss high or flat, and no clear gain over base few-shot): LoRA r=64 (alpha and lr unchanged), else epochs + 1 (max 3).
8. Still not beating base few-shot: regenerate data. Beats it with no target set: stop. Below target: regenerate data.

Retrain configs carry all current settings with **one** changed. Rank never goes above 64. If the new settings
were already trained on the same data (e.g. "back to r=16" after r=64 forgot), it regenerates data instead
(`config already tried`): `finetune-submit` would just return the old job and the loop would go round in circles.
`finetune-submit` may lower epochs to fit the time/GPU budget; the Evaluator reads `run.json`'s `requested_config`
(what was asked for), so a budget cut isn't mistaken for a deliberate change. If submit refuses outright
(< 0.25 epoch fits, low disk), no new job appears and the loop ends there.

## Output (one JSON object)

Always: `action`, `rule`, `reason`, `loop`, `tuned_score`, `decision_path`, `summary`. Plus:

| action | Keys (for the next agent) |
|---|---|
| `retrain` | `retrain_config_path` + `train_path` → `finetune-submit` |
| `regenerate_data` | `data_feedback_path` (+ `spec`) → DataGen. ⚠️ DataGen doesn't read `--data-feedback-path` yet |
| `stop` | `final_adapter_path`, `best_job_id`, `stop_reason`, `next_step` (the held-out eval command) |

Files: `<run dir>/decision_<split>.json` (full record), `retrain_config.json` / `data_feedback.json` next to it,
and `<bench dir>/.evaluator_history.jsonl` (one line per loop; loops are counted here, so a data regeneration
counts as a loop too). If the spec is inside a pipeline folder (`ftc`), it also writes
`loops/<n>/decision.json`, logs a `decide` event and advances the loop.

Reads both report formats: `finetune-eval`'s `report.json` and `docs/spec_and_io.md` §3.4 `eval_report.json`.

## Examples

```bash
python3 {baseDir}/scripts/finetune_decide.py --eval-report-path ~/finetuner/runs/<job>/eval_dev/report.json
python3 {baseDir}/scripts/finetune_decide.py --eval-report-path .../report.json --spec ~/finetuner/pipelines/<id>/task_spec.json
```

Tests: `python3 {baseDir}/tests/test_decide.py`

---
name: finetune-eval
description: Evaluate a LoRA fine-tuned model against its base model (task benchmark, general ability, safety) and write an eval report
metadata:
---

# Fine-tune Evaluator (wireframe L3: M7–M9)

Scores a LoRA adapter from `train_sft.py` against its base model. The job takes minutes, so the first call starts it in the background, and later calls report `running` until it finishes, then return the report. It exits 1 until the report is ready, so ScienceClaw retries on the next cycle.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--adapter-path` | `adapter/` dir from `train_sft.py` (its `metrics.json` next to it gives the base model and training data). Called with only `--bench-dir` (no adapter yet), the skill records the benchmark and exits 0 (`"state": "noted"`) | Required for an eval |
| `--bench-dir` | Benchmark folder from L2: `dev.jsonl`, `heldout.jsonl`, optional `safety.jsonl`. If omitted: the pipeline's benchmark (`pipeline_dir` in the adapter's `run.json` → `pipeline.json` links `bench_dir`, then `$FTC_PIPELINE_DIR`), else the last `--bench-dir` this skill was given (`$FINETUNER_RUNS/.last_bench_dir`). In agent mode the BenchGen artifact is consumed by the first eval, so later loops arrive with `adapter_path` only | see left |
| `--split` | `dev` every loop, `heldout` once at the end, `probe` (weak-spot set, report only; see finetune-bench-gen) | dev |
| `--skip-general` | Skip M8 (lm-eval) for faster loops | off |
| `--format` | `json` or `summary` | json |

Unknown flags (e.g. a `--query` ScienceClaw adds) are ignored.

Needs `ANTHROPIC_API_KEY` (Claude Sonnet judges free-text and safety items and plays the teacher) in the environment, `OPENAI_API_KEY` only if the judge is a `gpt-*` model, and the training venv at `$FINETUNER_DIR/.venv-train` (default `~/finetuner`) with the `anthropic` package (`openai` too for a `gpt-*` judge). Missing keys or packages stop the eval at the start, before any GPU work. Which judge model: `$FINETUNE_JUDGE_MODEL` if set, else the pipeline's task spec `models.judge` (a `gpt-*` or `claude-*` name), else the default `claude-sonnet-5-5`. The teacher is set separately (`$FINETUNE_TEACHER_MODEL`, default Claude Sonnet).

## Benchmark item format (what L2 must write)

One JSON object per line:

```json
{"id": "d-001", "messages": [{"role": "user", "content": "..."}], "reference": "...",
 "scoring": "json", "category": "methods-extraction", "source": "real"}
```

- `scoring`: `exact` (normalized string match), `json` (output must parse; score = field-level F1 against `reference`, which is an object or a JSON string) or `judge` (default: OpenAI Sol grades pass/fail against `reference`).
- `category`: a coverage-axis cell. The report breaks scores down by it, so the Evaluator can send failing categories (never items) back to L1.
- `source`: `real` or `synthetic`. The report scores each separately.
- `safety.jsonl` (M9): same format with `judge` scoring. `reference` describes the expected behaviour, e.g. "should decline" or "should answer normally" (to catch over-refusal). Use `category` values `should_refuse` / `should_help`. If the file is missing, M9 is skipped and the report says so.

## What it measures

- **M7 task:** four systems on the split: `base_zero_shot`, `base_few_shot` (3 training examples as prior turns: **tuning must beat this**), `teacher` (Claude, the ceiling) and `tuned`. Each gets a mean with a bootstrap 95% CI, the format-failure rate, and breakdowns by category and source. `gain_vs_base_few_shot` is a paired CI: if it includes 0, the gain may be noise.
- **M8 general:** lm-eval MMLU (1 question per subject = 57), GSM8K and IFEval (60 each): ~177 questions per model, base vs tuned on the same questions. `general.max_drop_points` > 3 means forgetting. At this size one question is ~1.7 points, so a 3-point drop is about 2 questions: treat a single small drop as a hint, not proof.
- **M9 safety:** pass rate on `safety.jsonl`, base vs tuned.

Models: the judge is **Claude Sonnet** (`claude-sonnet-5-5`; override with `--judge-model` or `$FINETUNE_JUDGE_MODEL`, and a `gpt-*` name uses OpenAI instead). Sonnet is a different family from the training-data writer (OpenAI Sol), so the judge can't favour the tuned model for sounding like its own training data (docs/decisions.md 2b). Runnable answers (SQL) are graded by execution and never reach the judge. The teacher is **Claude Sonnet** (`claude-sonnet-5-5`, `$FINETUNE_TEACHER_MODEL`). Only BenchGen uses Opus. Changing the judge model re-judges everything once (the cache key includes the model), so do not change it in the middle of a pipeline.

Base-model outputs and scores, teacher answers and judge verdicts are cached in `<bench-dir>/.eval_cache/`, so each loop pays only for the tuned model.

Deciding what to do with the report (retrain settings, LoRA rank, data feedback, stopping) is a separate step: see `docs/evaluator-notes.md` in the finetuner repo.

## Output

`<run dir>/eval_<split>/`: `report.json` (everything above, plus train loss from `metrics.json`), `outputs.jsonl` (every answer with its score), `status.json`, `eval.log`. On success it also writes `eval_report.json` (or `heldout_report.json`) next to `report.json` in the docs/spec_and_io.md §3.4 format that finetune-decide reads, plus a copy in `<pipeline>/loops/<n>/eval/` when the run belongs to a pipeline. Inside a pipeline that copy is rewritten on every call for the pipeline's *current* loop, so a reused training job (finetune-submit found identical data and settings, and the report already exists) is filed under the loop that asked for it, with that loop's number. It prints `eval_report_path` (the §3.4 file, so the reactor hands it straight to finetune-decide), `report_path` (the raw report.json) and headline numbers. A failed job is not restarted automatically: fix the cause, delete the folder, and call again.

## Examples

```bash
python3 {baseDir}/scripts/finetune_eval.py --adapter-path ~/runs/r001/adapter --bench-dir ~/bench/pubmed-json
```

Run the GPU job directly (foreground, more flags: `--help`):

```bash
~/finetuner/.venv-train/bin/python {baseDir}/scripts/eval/run_eval.py --adapter-path ~/runs/r001/adapter --bench-dir ~/bench/pubmed-json --skip-general
```

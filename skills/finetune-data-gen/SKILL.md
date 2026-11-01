---
name: finetune-data-gen
description: Generate verified, deduplicated chat-format training data (JSONL) for fine-tuning an LLM from a structured task spec; resumable, call repeatedly until done
metadata:
---

# Fine-tune Data Generator (DataGen, L1)

Turns a structured task spec into chat-format supervised fine-tuning examples that
`train_sft.py` can train on directly. Pipeline per batch:

**M2 generate** (LLM) → **M3 verify** (LLM keeps/rejects each example with a reason, rejects
regenerated up to 2×) → when the target is reached, **M4 format** (no LLM: validation, exact and
near dedup, optional benchmark contamination check) → `train.jsonl`.

## Capabilities

- Generates examples in batches that rotate over the spec's edge cases for coverage
- Verifies every example against the spec's success criteria and logs rejects with reasons
- Validates role order and removes exact and near-duplicate prompts
- Drops training examples that overlap benchmark items (contamination check)
- Stays inside the 30 s skill time limit and resumes across calls

## How to call it: call again until `done` is true

One call does as much as fits in ~27 s (typically 6–10 examples), saves progress, and returns.
**Call it again with the same `--spec` and `--run-id` until the result has `"done": true`.**
The final result contains `train_path`, the file to hand to the Trainer. Calling a finished run
again returns the saved result instantly.

```bash
python3 {baseDir}/scripts/gen_data.py --spec path/to/spec.json --run-id chem-v1 --format json
# or, without a spec file (a spec is drafted from the request on the first call and reused):
python3 {baseDir}/scripts/gen_data.py --query "polite customer-support replies for a bike shop" --n-examples 50 --format json
```

Repeat the **same** command: the run id is derived from the spec file name or the query, so
repeated calls resume the same run.

## How the ScienceClaw reactor sees this skill

The reactor hands an artifact to a skill when **any** key of the artifact matches a `--flag` in the skill's
`--help` text, and marks the artifact consumed when the skill exits 0. So `--help` shows only the pipeline
hand-off inputs (`--spec` / `--task-spec-path`, `--data-feedback-path`); every option in the table below still works but is hidden from `--help`, so generic
names (`query`, `run_id`, `n_examples`, `format`, ...) in other skills' outputs can't start or error-loop this
skill. The skill **exits 1 until it is done**, so its input hand-off stays unconsumed and it is called again
next cycle. Checked by `tests/test_handoffs.py`.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--task-spec-path` / `--spec` | Path to the task spec JSON (see below). Preferred | one of spec / `--query` / feedback |
| `--data-feedback-path` | `data_feedback.json` from finetune-decide (`regenerate_data`). Builds the next loop's spec like `run_pipeline.py`: weak categories first (twice) in `edge_cases`, `n_examples` + `extra_examples`, notes as a success criterion; written to `<pipeline>/loops/<n+1>/task_spec_data.json`, run id `<pipeline_id>-l<n+1>`, contamination check against `<pipeline>/bench` | none |
| `--query` | Free-text fine-tuning request; used to draft a spec when no `--spec` is given | one of `--spec` / `--query` |
| `--n-examples` | Number of training examples (overrides the spec) | spec value, or 50 for `--query` |
| `--run-id` | Name of this run; progress lives in `~/.scienceclaw/finetune/runs/<run-id>/` | spec file name, or slug of the query |
| `--format` | `json` (machine-readable result) or `summary` (one line) | json |
| `--batch-size` | Largest batch; smaller ones are planned automatically near the time limit | 10 |
| `--time-budget` | Seconds one call may use, from process start. Set it a few seconds below the executor's timeout | 27, or `$DATAGEN_TIME_BUDGET` |
| `--bench` | Benchmark for the contamination check: BenchGen run id, folder or file; `none` disables | a frozen BenchGen benchmark with the same spec, if any |
| `--verify-backend` | LLM backend for M3: `openai` (Sol, a different family from the Gemini generator, so it checks more strictly; model `$FINETUNE_OPENAI_MODEL`), `anthropic`, or `same` (= M2's backend). Needs `OPENAI_API_KEY` + the `openai` package; without them M3 falls back to M2's backend and the result's `warnings` says so | `openai` (Sol), or `$DATAGEN_VERIFY_BACKEND` |
| `--parallel` | Batches generated at the same time. Falls back to 1 on API rate limits | 1 |
| `--max-batches` | Stop after N rounds in this call (debugging) | 0 (no limit) |

## Spec file

Written by M1 (prompt verification). Until M1 exists, write it by hand.
Example: `{baseDir}/examples/spec_chem.json`

```json
{
  "task": "Answer general chemistry questions concisely and correctly.",
  "io_format": "Assistant reply is exactly two lines: 'ANSWER: <short answer>' then 'WHY: <one sentence>'.",
  "success_criteria": ["The answer is factually correct", "Follows the two-line format exactly"],
  "edge_cases": ["Calculations that need unit conversion", "Trick or false-premise questions"],
  "n_examples": 50,
  "system_prompt": null
}
```

| Field | Required | Used for |
|-------|----------|----------|
| `task` | yes | What the model should learn (M2 prompt, M3 check) |
| `io_format` | yes | Exact shape of the assistant reply (M2 prompt, M3 check) |
| `success_criteria` | yes | Checklist M3 keeps/rejects against |
| `edge_cases` | yes | Batches rotate over these as their focus |
| `n_examples` | yes | Number of training examples to produce (after dedup) |
| `system_prompt` | no | If set, added as a `system` message to every example |

## Output (default `--format json`; also saved to `<run_dir>/result.json`)

```json
{
  "run_id": "chem-v1",
  "done": true,
  "target": 50,
  "train_examples": 50,
  "train_path": "~/.scienceclaw/finetune/runs/chem-v1/train.jsonl",
  "raw_kept": 50, "rejected_total": 3, "unverified_total": 0, "pending_verify": 0,
  "m4": {"read": 50, "invalid": {}, "exact_duplicates": 0, "near_duplicates": 0, "contaminated": 0, "bench": "..."},
  "batches_this_call": 1, "stop_reason": "done", "sec_per_example": 1.25, "parallel": 1,
  "run_dir": "~/.scienceclaw/finetune/runs/chem-v1", "elapsed_s": 21.4
}
```

- `done: false` → call again. `stop_reason` says why this call stopped (`time budget used ...`,
  `no output from LLM ...`, `max batches`).
- `train_path` is only set when `done` is true. Each line is
  `{"messages": [{"role": "system"?...}, {"role": "user", ...}, {"role": "assistant", ...}]}`.
- If `rejected_total` is a large share of `raw_kept`, the spec is probably unclear or too strict;
  read `rejects.jsonl` for the reasons.

## Run folder

| File | Contents |
|------|----------|
| `spec.json` | Copy of the spec used |
| `raw.jsonl` | Verified examples, appended batch by batch |
| `rejects.jsonl` | Rejected examples with `reason`, `round`, `focus` (useful failure categories for the Evaluator) |
| `pending_verify.jsonl` | Generated but not yet verified (time ran out); verified first on the next call |
| `unverified.jsonl` | Could not be verified twice; never used for training |
| `state.json` | Progress, timing model, carried-over reject feedback |
| `train.jsonl` | Final training data (M4 output) |
| `format_report.json` | M4 counts plus sample dropped rows per reason |
| `result.json` | The last call's result (same JSON as stdout) |

## Examples

```bash
# Start (or continue) a run; repeat until "done": true
python3 {baseDir}/scripts/gen_data.py --spec {baseDir}/examples/spec_chem.json --run-id chem-v1 --format json

# Stricter verification with a different model family
python3 {baseDir}/scripts/gen_data.py --spec spec.json --run-id chem-v2 --verify-backend anthropic --format json

# Re-run M4 alone, e.g. with a benchmark file for the contamination check
python3 {baseDir}/lib/format_data.py --run-id chem-v1 --bench path/to/benchmark.jsonl --format json
```

## Notes

- Run `finetune-bench-gen` first: DataGen then finds the frozen benchmark for the same spec
  automatically and drops training examples that overlap any benchmark item (dev or held-out).
- Run order is safe either way:
  - In a pipeline (spec at `<pipeline>/task_spec.json`) while `<pipeline>/bench` is still being generated,
    DataGen generates everything but does not write `train.jsonl`: it returns
    `"state": "waiting_for_bench"` and exits 1, so ScienceClaw calls it again.
  - Without any benchmark, the result has `"m4": {"bench": null}` and a `warnings` entry. If a matching
    benchmark is frozen later, the next call re-runs the contamination check (no LLM) and generates
    replacements for removed rows.

- Executor timeouts differ: the plain skill executor uses 30 s (keep the default 27), while the
  deep-investigation agent path allows longer. With a longer limit, set `DATAGEN_TIME_BUDGET`
  (for example `280` for a 300 s limit) so one call can finish the whole run.
- Unknown extra flags are ignored, and errors are returned as `{"error": ...}` JSON with exit code 1.

- Runs on **Gemini** by default (ignores ScienceClaw's global `LLM_BACKEND`; override for the
  finetune skills only with `FINETUNE_LLM_BACKEND`). Needs `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
- Recommended environment: `LLM_TIMEOUT=20`, `GEMINI_MAX_RETRIES=1` (the script enforces its own
  per-call limits; long client-side retries waste the 30 s budget).
- Only `scripts/gen_data.py` may live in `scripts/`: the executor runs the first `*.py` there.
  Helpers are in `lib/`, tests in `tests/` (run offline, no API calls).
- Near-duplicate detection needs identical numbers plus ≥70% content-word overlap; reworded
  questions using entirely different words are not caught.

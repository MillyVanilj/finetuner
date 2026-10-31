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

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--spec` | Path to the task spec JSON (see below). Preferred | one of `--spec` / `--query` |
| `--query` | Free-text fine-tuning request; used to draft a spec when no `--spec` is given | one of `--spec` / `--query` |
| `--n-examples` | Number of training examples (overrides the spec) | spec value, or 50 for `--query` |
| `--run-id` | Name of this run; progress lives in `~/.scienceclaw/finetune/runs/<run-id>/` | spec file name, or slug of the query |
| `--format` | `json` (machine-readable result) or `summary` (one line) | summary |
| `--batch-size` | Largest batch; smaller ones are planned automatically near the time limit | 10 |
| `--time-budget` | Seconds one call may use, from process start. Set it a few seconds below the executor's timeout | 27, or `$DATAGEN_TIME_BUDGET` |
| `--verify-backend` | Use a different LLM backend for M3, e.g. `anthropic` (different model family checks more strictly; slower) | same as M2 |
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

## Output (`--format json`)

```json
{
  "run_id": "chem-v1",
  "done": true,
  "target": 50,
  "train_examples": 50,
  "train_path": "~/.scienceclaw/finetune/runs/chem-v1/train.jsonl",
  "raw_kept": 50, "rejected_total": 3, "unverified_total": 0, "pending_verify": 0,
  "format": {"read": 50, "invalid": {}, "exact_duplicates": 0, "near_duplicates": 0, "contaminated": 0},
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

- Executor timeouts differ: the plain skill executor uses 30 s (keep the default 27), while the
  deep-investigation agent path allows longer. With a longer limit, set `DATAGEN_TIME_BUDGET`
  (for example `280` for a 300 s limit) so one call can finish the whole run.
- Unknown extra flags are ignored, and errors are returned as `{"error": ...}` JSON with exit code 1.

- Recommended environment: `LLM_TIMEOUT=20`, `GEMINI_MAX_RETRIES=1` (the script enforces its own
  per-call limits; long client-side retries waste the 30 s budget).
- Only `scripts/gen_data.py` may live in `scripts/`: the executor runs the first `*.py` there.
  Helpers are in `lib/`, tests in `tests/` (run offline, no API calls).
- Near-duplicate detection needs identical numbers plus ≥70% content-word overlap; reworded
  questions using entirely different words are not caught.

---
name: finetune-bench-gen
description: Generate a verified benchmark (dev and held-out sets, plus an optional supplied safety set, in the finetune-eval format) for a fine-tuning task spec with Claude Opus; runs as a background job, call again until bench_dir is returned
metadata:
---

# Fine-tune Benchmark Generator (BenchGen, L2)

Builds the benchmark that `finetune-eval` (L3) scores the fine-tuned model on. It reads the **same task
spec as DataGen** but never sees DataGen's training data. Model: **Claude Opus** (`claude-opus-5-5`)
writes and checks every item, independent of the Gemini default the other finetune skills use.

Per round: **M5 generate** items for one category × difficulty cell → deterministic checks (valid key,
reference contains its own answer, no duplicates) → **M6a independent solve** (answers without seeing the
key) → **M6b review** (keep/reject with a reason) → rejects regenerated up to 2×. When the quota is met:
format regex check, **stratified** dev / held-out split, optional safety set copied in, then the benchmark is **frozen**.

## Capabilities

- Generates benchmark items balanced over the spec's edge cases and three difficulty levels
- Checks every answer key with an independent solve plus a review, logging rejects with reasons
- Writes dev.jsonl and heldout.jsonl (and a supplied safety.jsonl) in the format finetune-eval reads
- Runs as a background job so a 200-item benchmark fits ScienceClaw's 30-45 s skill timeout
- Lets DataGen drop training examples that overlap the benchmark (contamination check)

## How to call it

```bash
python3 {baseDir}/scripts/bench_gen.py --task-spec-path path/to/spec.json
```

A skill call never generates in-process: the **first call starts a background job** and exits 1; later
calls report progress (`"state": "running"`) and exit 1; when the benchmark is frozen the call returns
`"done": true` and **`bench_dir`** (the hand-off key for `finetune-eval --bench-dir`) and exits 0.
Exit 1 makes ScienceClaw keep the hand-off unconsumed and call again next cycle. A frozen benchmark is
returned instantly and never regenerated. **Run BenchGen before DataGen**: DataGen then finds the frozen
benchmark for the same spec and removes overlapping training examples.

By hand, generate in the foreground (no time limit): add `--foreground --time-budget 0`.
Expect roughly 10–20 minutes for 200 items with Opus.

## How the ScienceClaw reactor sees this skill

The reactor hands an artifact to a skill when **any** key of the artifact matches a `--flag` in the skill's
`--help` text, and marks the artifact consumed when the skill exits 0. So `--help` shows only the pipeline
hand-off inputs (`--spec` / `--task-spec-path`); every option in the table below still works but is hidden from `--help`, so generic
names (`query`, `run_id`, `n_examples`, `format`, ...) in other skills' outputs can't start or error-loop this
skill. The skill **exits 1 until it is done**, so its input hand-off stays unconsumed and it is called again
next cycle. Checked by `tests/test_handoffs.py`.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--task-spec-path` / `--spec` | Path to the task spec JSON (the same file DataGen uses) | one of spec / `--query` |
| `--query` | Free-text request; a spec is drafted when no spec file is given | one of spec / `--query` |
| `--n-items` | Benchmark size, dev + held-out | 200 |
| `--heldout-frac` | Share of items in the held-out split | 0.4 (120 dev / 80 held-out) |
| `--safety-file` | Curated safety set to validate and copy to `bench_dir/safety.jsonl` (see below) | none |
| `--run-id` | Benchmark name; files in `~/.scienceclaw/finetune/bench/<run-id>/` | spec file name, or query slug |
| `--backend` | LLM backend for M5/M6 | `anthropic`, or `$BENCHGEN_BACKEND` |
| `--model` | Model for the anthropic backend | `claude-opus-5-5`, or `$BENCHGEN_MODEL` |
| `--verify-backend` | Different backend for M6 (optional) | same as `--backend` |
| `--batch-size` | Largest batch per round | 8 |
| `--seed` | Seed for the dev/held-out split | 1234 |
| `--foreground` | Generate in this process instead of starting/polling the background job | off |
| `--time-budget` | Foreground only: seconds from process start, `0` = no limit | 27, or `$BENCHGEN_TIME_BUDGET` |
| `--max-rounds` | Foreground only: stop after N rounds (debugging) | 0 (no limit) |
| `--format` | `json` or `summary` | json |

## Files in `bench_dir` (what finetune-eval reads)

`dev.jsonl` (scored every loop), `heldout.jsonl` (scored **once**, at the end), `safety.jsonl` (M9).
One item per line:

```json
{"id": "b0042", "messages": [{"role": "user", "content": "What is the pH of 0.001 M NaOH?"}],
 "reference": "ANSWER: 11.0\nWHY: pOH is 3, so pH = 14 - 3.\n\nGRADING NOTE: the essential answer is 11 (accept values within ±0.11). The response must also follow the required output format: ...",
 "scoring": "judge", "category": "typical cases", "source": "synthetic",
 "difficulty": "easy", "grading": {"type": "numeric", "answer": 11.0, "tolerance": 0.11, "unit": ""},
 "verify": {"independent_answer": "11", "agreed": true}}
```

- `messages`: the spec's `system_prompt` (if any), then the user prompt.
- `scoring` (finetune-eval's modes):
  - `json` when the task's output is a JSON object: `reference` is that object (field-level F1).
  - `exact` only when the whole reference reply is the answer (finetune-eval compares whole replies).
  - `judge` for everything else; `reference` is the ideal reply plus a `GRADING NOTE` with the key
    (answer, tolerance, rubric) and the required output format, for the Claude judge.
- `category`: the spec's edge case, or `typical cases`. `source`: `synthetic`.
- `difficulty`, `grading`, `verify`: extra fields finetune-eval ignores (kept for analysis).
- `safety.jsonl` (M9) is **not generated**: Claude declines to write harmful test prompts, and a curated
  set is more trustworthy anyway. Pass `--safety-file` with lines like
  `{"prompt" or "messages": ..., "reference": "Should decline: ..." | "Should answer normally: ...",
  "category": "should_refuse" | "should_help"}`; it is validated (before any generation) and copied with
  `scoring: judge`. Without it there is no safety.jsonl and finetune-eval skips M9 with a note.
- `bench_meta.json`: spec + hash, counts per category/difficulty, scoring and key types, independent
  agreement rate, models used, split seed, SHA-256 of every file.

## Output (when done)

```json
{"state": "done", "done": true, "bench_dir": "~/.scienceclaw/finetune/bench/chem",
 "n_dev": 120, "n_heldout": 80, "n_safety": 0, "scoring": {"judge": 187, "exact": 13},
 "independent_agreement": 0.97, "rejected_total": 31}
```

While running: `{"state": "started" | "running", "done": false, "verified": 64, "target": 200, ...}`.
`"state": "starting"` means another call is launching the job at that moment (a lock prevents two jobs).
A job that dies is restarted; the count resets whenever it produced new items, so `"state": "failed"`
means 3 failures **in a row** without progress (or the reviewer rejected too much). Fix the cause, delete
`worker.json`, call again. Spec and safety paths may be relative: they are checked at once and handed to the
job as absolute paths.

## Run folder

| File | Contents |
|------|----------|
| `dev.jsonl`, `heldout.jsonl`, `safety.jsonl` (if supplied) | The frozen benchmark (finetune-eval format) |
| `bench_meta.json` | Metadata and hashes (written when frozen) |
| `items.jsonl` | Verified items in internal format, in acceptance order |
| `rejects.jsonl` | Rejected items with `reason`, `stage` (precheck/review), `round`, `cell` |
| `pending_verify.jsonl` | Generated but not yet verified; verified first next round |
| `state.json`, `worker.json`, `worker.log` | Progress, background job, its log |

## Notes

- Needs `ANTHROPIC_API_KEY`. Shared code is in `skills/finetune-data-gen/scripts/finetune_common/`,
  so install both skills.
- The Evaluator's judge and teacher are also Claude (Opus): a team choice, but Claude may score
  Claude-written items generously. Compare against `base_few_shot` as finetune-eval does.
- Never freezes a benchmark with fewer than half the requested items; reusing a run id with a
  different spec is refused, so benchmarks never mix.
- Tests: `tests/test_bench.py` (includes checks against a copy of finetune-eval's `scoring.py` in
  `tests/fixtures/`), `tests/test_link.py`.

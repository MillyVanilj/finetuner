# BenchGen (L2) — progress log

Skill: `skills/finetune-bench-gen/` (only script: `scripts/bench_gen.py`). Contract: `docs/spec_and_io.md` §3.2 plus
`skills/finetune-eval` (its `scoring.py`). Model: **Claude Opus** (`claude-opus-5-5`) writes and checks the benchmark.

## What it does
Quota (typical cases 40 %, edge cases share 60 %; difficulty 30/40/30) → per round **M5** generate items for one
category × difficulty cell → deterministic prechecks (valid key, reference contains its own answer, exact/near dup) →
**M6a** independent solve (no key shown) → **M6b** review (sees key + independent answer) → rejects regenerated ≤2 rounds.
When the quota is met: format regex (kept only if it matches every reference), stratified dev/held-out split (seed 1234),
the user's `seed_examples` added to held-out, optional `--safety-file` copied, files written, **frozen**.

## Output (bench_dir)
- In a pipeline (spec inside `$FINETUNER_DIR/pipelines/<id>/`, or `$FTC_PIPELINE_DIR`): `<pipeline>/bench/`, with
  `progress.json`, events (`bench` start / link / done) and the `bench_dir` link in `pipeline.json`.
  Stand-alone: `~/.scienceclaw/finetune/bench/<run_id>/`.
- `dev.jsonl`, `heldout.jsonl`, optional `safety.jsonl`. Item = spec_and_io §3.2 fields (`id, split, question, messages,
  reference, category, difficulty, source, source_ref, base_pass_rate`) + `scoring` (finetune-eval) + `grading`, `verify`.
- `scoring` mapping: `json` tasks → `json` (reference = JSON text); `exact` only when the whole reference reply is the
  answer (finetune-eval compares whole replies); everything else (all numeric) → `judge`, with a `GRADING NOTE`
  (answer, tolerance or rubric, required output format) appended to `reference`.
- `bench_report.json` (spec_and_io) and `bench_meta.json` (spec + hash, counts, scoring types, independent agreement,
  models, split seed, SHA-256 of every file, `format_version`). A benchmark frozen by an older version is re-exported in
  the current format on the next call (same items, same seeded split, no LLM).
- Sizes: `--n-items`/`--heldout-frac`, else spec v2 `benchmark.n_dev`/`n_heldout`, else 200 items, 40 % held-out.

## Running
- Skill call (`--task-spec-path <spec>`): starts a **background job** and exits 1; later calls report `running` (exit 1);
  frozen → `"done": true` + `bench_dir`, exit 0. Dead worker restarted ≤3×; a terminal failure (too many rejects) is not.
- By hand: `--foreground --time-budget 0`. ~4 s per item with Opus → ~15 min for 200.
- The background job gets absolute spec / safety paths (it runs with cwd = ScienceClaw; relative paths used to fail
  there). A missing file is reported at once. Restarts: the count resets whenever the job produced new items, so
  `failed` = 3 failures in a row without progress. A lock file stops two calls from starting two jobs.
- **Safety prompts are not generated**: Claude declines to write harmful test prompts (and a curated set is better).
  Pass `--safety-file` (validated before generation). Without it, finetune-eval skips M9 with a note.
- Claude calls go through `finetune_common.llm.AnthropicClient` (no `temperature`; ScienceClaw's llm_client sends it and
  current Claude models reject it).

## Results (Nov 1)
- Test run, 12 items: 12/12 verified, 2 rejected and regenerated, regex matched all references, independent agreement 0.89.
  Manual check of all 12 answer keys (pH of weak acid / 1e-8 M HCl, stoichiometry, gas molar mass, dilution, buffers,
  false-premise and ambiguous items): all correct.
- Full 200-item chemistry benchmark (`spec_chem.json`): running.

## Tests (offline)
`tests/test_bench.py` (contract checks use finetune-eval's real `scoring.py` when it is installed next to BenchGen,
else the copy in `tests/fixtures/`), `test_link.py` (DataGen auto-detects the frozen benchmark and drops leaked prompts, incl. rewordings),
`test_pipeline.py` (pipeline folder, progress, events; needs `ftc`), `test_anthropic_client.py`.

## Open
- `base_pass_rate` (drop items the base model always gets right) is not computed; it needs the GPU.
- `benchmark.real_sources` (PubMed, HF) not used yet; all generated items are `source: synthetic`.

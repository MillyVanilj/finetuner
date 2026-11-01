# Task spec, I/O formats and the live dashboard

*Written 2026-10-31. This is the contract between the stages of wireframe v0.2. If you build a stage
(BenchGen, Eval, Evaluator, ReportAgent, run_pipeline.py), read and write exactly these files. Change a format
here first, then in code. Working reference for every format: `scripts/simulate_pipeline.py`
(it writes all of them, without a GPU).*

## 1. The big picture

```
 user goal ──M1──▶ task_spec.json ──┬──▶ DataGen (L1) ─ train.jsonl ─▶ Trainer ─ adapter ─┐
                                    └──▶ BenchGen (L2) ─ dev.jsonl / heldout.jsonl ───────┴▶ Eval (L3) ─ eval_report.json
                                                                                                  │
                                         decision.json ◀── Evaluator ◀────────────────────────────┘
                                         ├ retrain        → retrain_config.json  → Trainer
                                         ├ regenerate_data→ data_feedback.json   → DataGen
                                         └ stop           → held-out once → final_report.json + model card
```

Everything for one user goal lives in **one pipeline folder**. Existing run folders are *linked*, not moved:

```
$FINETUNER_DIR/pipelines/<pipeline_id>/        (FINETUNER_DIR defaults to ~/finetuner)
  task_spec.json          the spec (v2, §2)                                   written by M1
  pipeline.json           {pipeline_id, title, created_at, loop, links{...}}  written by ftc.pipeline
  events.jsonl            what happened, one event per line (§4)              every stage
  bench/                  dev.jsonl, heldout.jsonl, bench_report.json, progress.json   BenchGen
  loops/<n>/eval/         progress.json, predictions_<system>.jsonl, eval_report.json  Eval
  loops/<n>/              decision.json, retrain_config.json | data_feedback.json      Evaluator
  final/                  final_report.json, model_card.md                    ReportAgent

links in pipeline.json:
  data_run_dir   [..]   ~/.scienceclaw/finetune/runs/<run_id>/   DataGen (Owen): state.json, raw.jsonl, rejects.jsonl, train.jsonl
  train_job_id   [..]   ~/finetuner/runs/<job_id>/               Trainer: run.json, status.json, train_log.jsonl, metrics.json, adapter/
  bench_dir             <pipeline>/bench
```

`pipeline_id` = `<spec id>-<YYYYmmdd-HHMMSS>`. A "loop" = one train + eval round; it starts at 1.

## 2. Task spec v2 (`task_spec.json`)

**v2 is a superset of DataGen's v1** (`skills/finetune-data-gen/examples/spec_chem.json`). The v1 fields keep their
names and meaning, so `gen_data.py --spec task_spec.json` works unchanged. Everything new has a default, so a v1 file
is a valid v2 file after `ftc.spec.upgrade()`.

Validate / upgrade from the shell:
```bash
python3 -m ftc.spec path/to/spec.json             # {"ok", "errors", "warnings"}
python3 -m ftc.spec path/to/spec_v1.json --upgrade
```
From Python (stdlib only): `from ftc.spec import load_spec` (returns the upgraded spec, raises on errors).
For input from people or an LLM, use `check_spec(raw)` → `(spec, errors, warnings)`: it never raises, so a
malformed spec comes back as errors. Fields marked **not read yet** below are kept in the spec, and
validation warns when one is changed from its default, so nobody expects it to take effect.

| Field | Req. | Default | Read by | Meaning |
|---|---|---|---|---|
| `spec_version` | | `2` | all | |
| `id`, `title`, `domain` | | from task | UI, folder name | |
| `task` | **yes** (v1) | | M2, M3, M5, judge | What the model should learn, 1–2 sentences |
| `io_format` | **yes** (v1) | built from `output` | M2, M3, M5 | Exact shape of the assistant reply, in words. Generators only read this string, so it must say everything |
| `success_criteria` | **yes** (v1) | | M3, M6, judge | Checkable statements |
| `edge_cases` | **yes** (v1) | | M2 (batch focus), M5 (category), Evaluator | Tricky situations. They double as the **failure categories** of the feedback loop |
| `n_examples` | **yes** (v1) | | DataGen | Training examples after dedup (≥10) |
| `system_prompt` | | `null` | DataGen, Eval | Added as a `system` message to every train and bench item |
| `input` | | `{description, example}` | M2, M5, UI | What the model receives |
| `output.type` | | `text` | scorer, M1 | `text` · `json` · `label` · `number` |
| `output.json_schema` | if json | `null` | format check, field F1 | JSON Schema of the reply |
| `output.labels` | if label | `null` | scorer | Allowed answers |
| `output.format_regex` | | `null` | format check | e.g. `^ANSWER: .+\nWHY: .+$` |
| `coverage.topics`, `coverage.difficulty` | | `[]`, easy/medium/hard | **not read yet** (planned: M2, M5) | Diversity grid |
| `seed_examples` | | `[]` | BenchGen (held-out), M2 (few-shot) | `[{"input", "output", "source": "user"\|"real", "source_ref"}]`, the user's real examples |
| `scoring.method` | | from output.type | Eval | `exact_match` · `label_accuracy` · `numeric` · `json_field_f1` · `regex` · `judge` |
| `scoring.format_check` | | from output.type | Eval | `none` · `regex` · `json_schema` |
| `scoring.json_fields` | | schema properties | Eval | Fields scored by `json_field_f1` |
| `scoring.numeric_tolerance` | | `0.01` | **not read yet** (planned: Eval) | Relative tolerance for `numeric` |
| `scoring.judge_rubric` | if judge | `null` | **not read yet** (planned: judge) | Grade against the reference with this rubric |
| `scoring.pass_threshold` | | `0.5` | **not read yet** (planned: Eval) | Item counts as passed if its score ≥ this |
| `benchmark.n_dev` / `n_heldout` | | 200 / 100 | BenchGen | Dev = scored every loop. Held-out = scored once, at the end |
| `benchmark.seed_examples_to_heldout` | | `true` | BenchGen | Put the user's real examples in held-out |
| `benchmark.real_sources` | | `[]` | **not read yet** (planned: BenchGen) | e.g. `["pubmed", "huggingface:<id>"]` |
| `targets.target_score` | | `null` | Evaluator | Stop when tuned dev score ≥ this |
| `targets.min_gain` | | `0.02` | Evaluator | Stop when gain over the last loop is smaller |
| `targets.max_loops` | | `3` | Evaluator | |
| `targets.max_general_drop` | | `0.03` | Evaluator | M8 drop above this = forgetting |
| `budget.max_api_usd` / `max_gpu_hours` | | 20 / 2 | `max_gpu_hours`: Trainer pre-flight, Evaluator. `max_api_usd`: **not read yet** | Hard stop |
| `models.student/generator/verifier/bench_writer/judge` | | see `ftc/spec.py` | `student`: Trainer (every run). `judge`: Eval, if it is a `claude-…` model and `$FINETUNE_JUDGE_MODEL` isn't set (that machine-wide override wins). The other three: **not read yet** (DataGen and BenchGen pick their model from env/flags) | Different families per role (decisions.md, Decision 3). Validation warns if generator = judge |
| `training` | | `{}` | Trainer (first run; later runs take the Evaluator's retrain config) | LoRA overrides: model, rank, alpha, lr, epochs, batch_size, grad_accum, max_length, seed (other keys are an error) |
| `m1` | | | UI | `{goal, answers, source}`: how M1 produced the spec |

Examples: `examples/specs/chem_qa.json` (Owen's chemistry task, upgraded) and `examples/specs/pubmed_extract.json`
(JSON extraction, field-F1 scoring).

**M1 (prompt verification)** = `ftc/m1.py`. Use it from the dashboard's *New task* page, or with
`python3 -m ftc.m1 "goal..."` in a terminal. It drafts the spec with Claude and asks at most 5 clarifying questions,
each with a default, then applies the answers. It runs *before* the agents, because ScienceClaw can't pause to ask
a human. Without an API key it returns a template.

## 3. Hand-off files

### 3.1 Training data (DataGen → Trainer): unchanged
JSONL, one `{"messages": [{"role":"system"?}, {"role":"user"}, {"role":"assistant"}]}` per line.
DataGen's result key is `train_path`; `finetune-submit` accepts it as `--train-path`.
Exit codes (all pipeline skills): exit 0 only when the hand-off key is in the result; exit 1 while not done
(DataGen, BenchGen, finetune-status, finetune-eval), so the reactor keeps the input unconsumed and calls again.
Each skill's `--help` lists only flags that are hand-off keys (the reactor matches every `--flag` it finds there);
`tests/test_handoffs.py` checks the whole key matrix.
Run order: inside a pipeline, if `bench/` exists but is not frozen yet, DataGen returns `"state": "waiting_for_bench"`
and exits 1 (no `train_path` yet), so the contamination check always uses this pipeline's benchmark.

### 3.2 Benchmark items (BenchGen → Eval, and DataGen's contamination check)
`bench/dev.jsonl` and `bench/heldout.jsonl`, one item per line:
```json
{"id": "dev-0007", "split": "dev",
 "question": "What is the pH of 0.01 M HCl?",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "What is the pH of 0.01 M HCl?"}],
 "reference": "ANSWER: 2.0\nWHY: HCl dissociates fully, so pH = -log(0.01) = 2.",
 "category": "Calculations that need unit conversion",
 "difficulty": "easy", "source": "synthetic", "source_ref": null, "base_pass_rate": null}
```
- `question` = plain user text. **Required**: `format_data.py --bench` reads it for the contamination check.
- `scoring` = `exact` · `json` · `judge`: how `finetune-eval` scores the item (its `scoring.py`). BenchGen sets it;
  `judge` items carry the answer key / rubric in a `GRADING NOTE` at the end of `reference`.
- Extra fields (`grading`, `verify`) are BenchGen's own analysis data; consumers ignore them.
- `bench/safety.jsonl` (optional, for finetune-eval's M9): same item shape, `category` = `should_refuse` | `should_help`.
  BenchGen copies it from `--safety-file`; it does not write safety prompts itself.
- `messages` = exactly what the model is sent (no assistant turn). `reference` = the ideal reply, as a **string**
  (for JSON tasks, the JSON text; the scorer parses it).
- `category` ∈ `["typical cases"] + spec.edge_cases`. That makes failures map straight onto feedback categories.
- `source`: `synthetic` · `user` (from seed_examples) · `real` (PubMed, HF…; put the id in `source_ref`).
- `base_pass_rate`: optional, from M6's difficulty check (drop items the base model always gets right).

`bench/bench_report.json`: `{n_generated, n_rejected, n_dev, n_heldout, writer_model, checker_models, dropped_too_easy, created_at}`.
The benchmark is written **once** and stays fixed across loops.

### 3.3 Predictions (Eval, optional but useful for debugging)
`loops/<n>/eval/predictions_<system>.jsonl`: `{"id", "system", "output", "format_ok", "score", "judge": {...}|null, "latency_s"}`.
Systems: `base_zeroshot`, `base_fewshot` (prompted with 3–5 train examples: **the baseline fine-tuning must beat**),
`teacher` (the ceiling), `tuned`. Base and teacher scores are computed in loop 1 and reused (`"cached": true`).

### 3.4 Eval report (Eval → Evaluator, UI): `loops/<n>/eval/eval_report.json`
```json
{"pipeline_id": "...", "loop": 2, "split": "dev", "n_items": 200, "train_job_id": "...", "adapter_path": "...",
 "scoring": "json_field_f1",
 "systems": {"tuned":        {"score": 0.79, "ci95": [0.73, 0.85], "format_rate": 0.97, "n": 200, "model": "...", "cached": false},
             "base_fewshot": {...}, "base_zeroshot": {...}, "teacher": {...}},
 "by_category": [{"category": "...", "n": 33, "base_fewshot": 0.48, "tuned": 0.71}],
 "by_source":   [{"source": "real", "n": 60, "base_fewshot": 0.43, "tuned": 0.74}],
 "general":     [{"task": "gsm8k", "n": 250, "base": 0.84, "tuned": 0.80, "delta": -0.04}],
 "failure_categories": [{"category": "...", "n_fail": 12, "share": 0.29, "example_reason": "wrong value"}],
 "verdict": {"beats_fewshot": true, "meets_target": false, "forgetting": true},
 "created_at": "..."}
```
Scores are in 0–1. `ci95` = bootstrap (or normal-approximation) 95% interval. `general` is empty when M8 was skipped.
Inside a pipeline, `finetune-eval` writes this file itself (converted with `ftc/report_convert.py`) for the pipeline's current loop on every call, so a reused training job gets the new loop number.
The final held-out run writes the same shape to `heldout_report.json` with `"split": "heldout"`.

### 3.5 Live eval progress (Eval → UI): `loops/<n>/eval/progress.json`
Overwrite it every ~20 items (`Pipeline.progress("eval", ...)` does this):
```json
{"stage": "eval", "state": "running", "phase": "M7 task benchmark", "split": "dev", "done": 460, "total": 800,
 "unit": "answers", "current_system": "tuned",
 "metrics": {"systems": {"tuned": {"done": 60, "total": 200, "running_score": 0.62, "cached": false}, "...": {}},
             "general": [ ...same rows as eval_report.general, filled as suites finish... ]},
 "updated_at": "..."}
```

### 3.6 Evaluator decision: `loops/<n>/decision.json`
```json
{"loop": 1, "action": "regenerate_data", "rule": "failures concentrated in categories",
 "reason": "58% of dev failures are in 2 categories: ...", "tuned_score": 0.66, "target_score": 0.8,
 "data_feedback": {"categories": ["Phase not stated (phase must be null)"], "extra_examples": 150,
                   "notes": "Category names only, never test items."},
 "data_feedback_path": ".../loops/1/data_feedback.json"}
```
- `action`: `retrain` (writes `retrain_config.json` = finetune-submit's `--retrain-config-path` format:
  `{lr, epochs, rank, …, parent_job_id, reason}`) · `regenerate_data` (writes `data_feedback.json`) · `stop`.
- Built: `skills/finetune-decide` (`--eval-report-path`; `--spec` is accepted and ignored, the spec is read from the
  pipeline folder). It reads this §3.4 `eval_report.json` only, not the eval skill's own `report.json`. Its JSON output
  carries the hand-off keys for the reactor:
  `retrain` → `retrain_config_path` **and** `train_path` (finetune-submit requires both; same data, new settings) ·
  `regenerate_data` → `data_feedback_path` · `stop` → neither, but `best_loop` + `best_adapter_path` for the held-out
  run and the ReportAgent. A `stop` decision.json also has `best_loop` and `best_adapter_path`.
- `rule` names the deterministic rule that fired (order and thresholds: `skills/finetune-decide/SKILL.md`,
  reasoning: `docs/evaluator-notes.md`). `reason` is a templated sentence; there is no LLM call.
- The Evaluator moves the pipeline to loop n+1 (`pipeline.json` → `loop`) unless it stops.

**Passing the spec between skills:** the key is `spec` (the path to `task_spec.json`), matching DataGen's `--spec`.
Skills that take a spec should accept `--spec`, optionally with `--task-spec-path` as an alias.

### 3.7 Final report: `final/final_report.json`
`{pipeline_id, best_loop, train_job_id, adapter_path, hub_url, infinite_post_url, heldout: {systems, by_source}, dev_history: [{loop, tuned}], model_card_path, created_at}`.

### 3.8 Training progress (Trainer → UI): already exists
`runs/<job_id>/status.json` (live), plus **new** `runs/<job_id>/train_log.jsonl`: `train_sft.py` now appends every
Trainer log line (`{"loss", "learning_rate", "epoch", "step"}` or `{"eval_loss", ..., "step"}`), which drives the live loss curve.

**Pre-flight (automatic GPU launch checks, no human OK):** `run.json` → `preflight` =
`{decision: launch|reuse|wait|refuse, reason, checks: [{check, ok, detail, overridden}], estimate_s, s_per_example_epoch,
speed_from_n_runs, pipeline_gpu_hours_used, pipeline_gpu_hours_budget}`, shown in the Trainer panel. A wait or refuse
makes no job folder; it appears as a `train` event (`log` / `failed`) starting with "Pre-flight …". A reuse appears as
"Skipped training …" plus a link to the reused job. The checks read `budget.max_gpu_hours` and `targets.max_loops` from
the spec.

## 4. Events (`events.jsonl`): how a stage tells the UI what it's doing

One JSON object per line, appended (never rewritten):
```json
{"ts": "2026-11-01T09:30:00.12Z", "stage": "train", "type": "start", "loop": 2,
 "msg": "LoRA r=16 lr=0.0001 epochs=1 on 450 examples", "train_job_id": "20261101-093000-ab12cd"}
```
| key | values |
|---|---|
| `stage` | `spec` · `data` · `bench` · `train` · `eval` · `decide` · `output` · `pipeline` |
| `type` | `start` · `progress` · `done` · `failed` · `link` · `metric` · `log` · `decision` |
| `loop` | 1-based (0 for spec/bench, which run once) |
| `msg` | one human-readable line (shown in the Activity feed) |
| extra | anything; `link` events carry `key` + `value` |

A stage needs at most **three calls**: `start`, `done`/`failed`, and one `link` to its run folder. The dashboard reads
fine-grained progress from the files that already exist (DataGen `state.json`, Trainer `status.json` /
`train_log.jsonl`, `progress.json`), so nobody has to emit progress events in a loop.

From Python:
```python
import os, sys; sys.path.insert(0, os.environ.get("FINETUNER_DIR", os.path.expanduser("~/finetuner")))
from ftc.pipeline import Pipeline
pl = Pipeline.from_any(args.spec)          # finds the pipeline folder from the spec path (or $FTC_PIPELINE_DIR); no-op if none
pl.emit("train", "start", "LoRA r=16 on 450 examples")
pl.link("train_job_id", job_id, stage="train", append=True)
pl.progress("eval", done=120, total=800, unit="answers", phase="M7 task benchmark", metrics={...})
```
From the shell: `python3 -m ftc.cli emit|link|loop|status <pipeline_dir> ...` (see `ftc/cli.py`).
Calls never raise. Without a pipeline folder they do nothing, so the skills keep working stand-alone.

### 4.1 Poster wall (`wall.jsonl`): agents talking to each other

Events say *what happened*; the wall is for *conversation* between agents (and humans, from the dashboard):
requests ("@DataAgent more unit-conversion examples"), questions, findings, replies, emoji reactions.
Append-only, one line per post or reaction (`ftc/wall.py` has the full schema):
```json
{"id": "p-20261101-093000-ab12", "ts": "...", "type": "post", "author": "Evaluator", "to": ["DataAgent"],
 "kind": "request", "text": "...", "loop": 2, "reply_to": null, "refs": {"eval_report_path": "..."}}
{"id": "r-...", "ts": "...", "type": "react", "author": "DataAgent", "post": "p-20261101-093000-ab12", "emoji": "👍"}
```
`kind`: note · finding · request · question · answer · alert · praise. `@Name` in the text also addresses the post.
```python
from ftc.wall import Wall
w = Wall.from_any(pipeline_dir)                      # no pipeline -> the lobby, $FTC_PIPELINES/_wall.jsonl
pid = w.post("EvalAgent", "Weak on unit conversion", to="DataAgent", kind="request")
w.reply(pid, "DataAgent", "On it")                   # addressed back to EvalAgent
w.inbox("DataAgent")                                 # notes for me I haven't replied/reacted to yet
```
Shell: `python3 -m ftc.cli wall -p <pipeline_dir> post|reply|react|read|inbox|thread ...`. ScienceClaw agents:
the `finetune-wall` skill. `run_pipeline.py` and the simulator post at every hand-off, and when DataGen regenerates
data, unanswered notes to DataAgent from outside the pipeline (e.g. a human's hint) are added to the data spec.

## 5. The dashboard

```bash
cd ~/finetuner && python3 dashboard/server.py          # http://localhost:8765, stdlib only
brev port-forward <instance> --port 8765:8765           # on your laptop, then open http://localhost:8765
python3 scripts/simulate_pipeline.py --speed 2          # fake run to watch (or the ▶ button in the UI)
```
- **Pipeline map**: the wireframe as live nodes. Each node shows its status (icon + label), key number and progress
  bar. The edge currently carrying work animates, and the two loop-back paths turn orange once the Evaluator takes
  them. Click a node to open its panel.
- **Stage panels**: *Data* shows verified/target, the reject rate, a live stream of the latest examples and rejects
  with reasons, rejects per edge case, and the M4 dedup/contamination counts. *Benchmark* shows progress, items per
  category and source, and sample items. *Trainer* shows step/ETA, a **live loss curve** (train + eval), config
  chips and all runs. *Evaluation* shows per-system live progress with running scores, then scores with 95% CIs vs
  target, per-category and per-source bars, M8 forgetting check and failure categories. *Iterate?* shows each
  decision with its rule and the exact feedback sent back. *Output* shows held-out scores and links.
- **Across loops**: tuned dev score per loop vs base few-shot and target, plus the decision history.
- **Activity**: the event log, newest first.
- **New task**: M1 in the browser: goal → draft → questions → editable spec form with live validation →
  *Create pipeline* (shows the next commands) or *Simulate with this spec*.
- `#p=<id>&once=1&stage=train&theme=light` in the URL gives a static snapshot (screenshots, slides).

## 6. Running a real pipeline (`scripts/run_pipeline.py`)

Press **▶ Start pipeline** in the dashboard, or run it in a terminal:
```bash
cd ~/finetuner && set -a && source .env && set +a          # API keys (ScienceClaw doesn't read .env itself)
python3 scripts/run_pipeline.py --pipeline ~/finetuner/pipelines/<id>   # add --skip-general for faster loops
```
It calls the real skills as subprocesses, like ScienceClaw's executor does, using
`$SCIENCECLAW_DIR/skills/<name>` (default `~/scienceclaw`, where `scripts/install.sh` copies them) or else this
repo's `skills/`:

1. BenchGen (background job, no `--run-id`, so it writes `<pipeline>/bench`) runs while DataGen runs. After
   both finish, DataGen's M4 is re-run against dev + held-out for the contamination check.
2. `finetune-submit` (its pre-flight decides; a "wait" is retried, a "refuse" ends the run), then poll
   `finetune-status` until it reports done.
3. Poll `finetune-eval` (dev) until done; its `report.json` is converted to §3.4 by `ftc/report_convert.py`.
4. `finetune-decide`, then act on the action:
   - `retrain`: same data, `--retrain-config-path`.
   - `regenerate_data`: DataGen can't read `data_feedback.json` yet, so the runner writes
     `loops/<n>/task_spec_data.json`, with the weak categories first (twice) in `edge_cases` and `n_examples` raised,
     and runs DataGen on that.
   - `stop`: score the held-out split once, then write `final/final_report.json`.

State: `runner.json` (pid, current step, error), log: `runner.log`, both shown in the dashboard. **■ Stop**
sends SIGTERM: the runner stops after its current call. Jobs it already launched (training, eval, BenchGen) keep
running. A restart picks up where it was: DataGen, BenchGen and submit all resume or reuse.
Needs: the Brev machine set up (`setup_brev.sh`, `install.sh`) and keys loaded. All skills it calls are on main.
Test without GPU or API: `python3 tests/test_run_pipeline.py` (stand-in skills, 3 loops).

## 7. Open items for the team
- Status per box: `docs/wireframe_v0.2.md`. BenchGen is on main (`skills/finetune-bench-gen`, see
  `docs/benchgen-progress.md`), and so is the Evaluator (`skills/finetune-decide`).
- `finetune-eval` writes the §3.4 `eval_report.json` itself (filed under the pipeline's current loop) and hands it on;
  `run_pipeline.py` passes it through unchanged.
- Output: `skills/finetune-report` (agent mode) and `run_pipeline.py` both score held-out once and write
  `final/final_report.json` + `final/model_card.md` (`ftc/report.py`). Not built: Hugging Face push (train with
  `--push-to-hub`) and the Infinite post.
- DataGen reads `--data-feedback-path` (agent mode; builds the same spec copy as the runner's `regen_spec`). It
  still ignores the v2-only fields (seed examples, coverage topics); using `seed_examples` as few-shot is a cheap win.
- `finetune-eval` reports only a stage name while running (no item counts), so the Evaluation panel shows its
  stage list.

## 8. Agent mode (ScienceClaw agents)

The same chain without `run_pipeline.py`: each stage is a ScienceClaw agent (a heartbeat daemon) whose reactor runs
its skill when a peer's artifact carries a key that matches one of the skill's `--flags` (§6's keys).

```bash
bash scripts/install.sh                                # skills/ -> ~/scienceclaw/skills, agents/*/agent_profile.json -> ~/.scienceclaw/profiles
set -a; source .env; set +a                            # the skills read API keys from the environment
python3 -m ftc.cli new examples/specs/chem_qa.json     # a pipeline folder (or the dashboard's New task page)
python3 -m ftc.cli agents kickoff <pipeline_dir>       # scope the agents to it + post {"spec": "<pipeline>/task_spec.json"}
python3 -m ftc.cli agents start --interval 60          # 7 daemons, one per profile in agents/ (refuses unscoped agents)
python3 -m ftc.cli agents status                       # running? + each agent's last log line; the dashboard shows progress
python3 -m ftc.cli agents stop
```

| Agent (`agents/<Name>`) | Skill | Wakes on | Hands on |
|---|---|---|---|
| DataAgent | finetune-data-gen | `spec` (kickoff), `data_feedback_path` | `train_path` |
| BenchAgent | finetune-bench-gen | `spec` (kickoff) | `bench_dir` |
| TrainAgent | finetune-submit | `train_path` (+ `retrain_config_path`) | `train_job_id` |
| StatusAgent | finetune-status | `train_job_id` | `adapter_path` |
| EvalAgent | finetune-eval | `adapter_path` (+ `bench_dir`) | `eval_report_path` |
| DecideAgent | finetune-decide | `eval_report_path` | `retrain_config_path` + `train_path` · `data_feedback_path` (+ `spec`) · `best_adapter_path` |
| ReportAgent | finetune-report | `best_adapter_path` (+ `best_loop`, `best_job_id`) | `final_report_path` (end of the chain) |

- Profiles: one skill each in `preferred_tools`, no `interests` (so no LLM investigation: the agents only react),
  `disable_llm` and `disable_platform` (no Infinite comments or upvotes). ScienceClaw's heartbeat still only runs a beat
  when `~/.scienceclaw/infinite_config.json` has an `api_key`; `agents start` refuses without one.
- The kickoff (`ftc/agents.py`) saves the artifact with ScienceClaw's own `ArtifactStore` as producer `Kickoff`,
  `investigation_id` = the pipeline id (every child inherits it), and doesn't publish it to Infinite. Its payload is
  only `spec`: another key could wake a skill with half its inputs (e.g. `finetune-submit --pipeline-dir`).
- `tests/test_handoffs.py` checks every key match, including the kickoff's; `tests/test_agent_mode.py` drives the whole
  chain with ScienceClaw's real `ArtifactReactor` (stand-ins for the GPU/API skills; needs ScienceClaw + pyyaml).
- One pipeline at a time: the agents follow the latest kickoff (the scope above), so an earlier pipeline's unfinished
  hand-offs are left alone.
- Skill help text matters: the reactor treats every `--word` in a skill's `--help` output (description included) as
  an input flag. Keep flag names out of descriptions, and hide internal flags with `help=argparse.SUPPRESS`.
- The reactor also calls each skill once with only `--describe-schema` (looking for an `--input-json` schema). Our
  skills ignore unknown flags, so a skill must do nothing when its hand-off input is missing: make that flag
  `required=True` (or exit early). `tests/test_agent_mode.py` checks this for every skill.
- Scope: `agents kickoff` sets `investigation_id_filter` = the pipeline id in each installed profile, so agents only
  act on this pipeline's artifacts, not on old, never-used hand-offs in the shared index (an old `train_path` would
  start a GPU job). `agents start` refuses unscoped agents; `install.sh` keeps the scope when it re-copies profiles.
  A new kickoff moves running agents to the new pipeline at their next beat.
- Finding the pipeline without `$FTC_PIPELINE_DIR`: the daemons don't set it, and DataGen's run folder
  (`~/.scienceclaw/finetune/runs/<run_id>/`, which holds `train_path`) is outside the pipeline. So DataGen writes a
  `.pipeline` pointer file there, and `Pipeline.from_any()` follows it: finetune-submit, handed only `train_path`,
  records `pipeline_dir` in `run.json`, and everything after it (eval report under `loops/<n>/`, loop count, final
  report) stays in the pipeline. Without it the chain still ran but outside the pipeline folder.
- DataGen names a run for a spec inside a pipeline `<pipeline_id>-l<loop>` (like `run_pipeline.py`), not after the
  file: every pipeline's spec is `task_spec.json`, so the old default (`task_spec`) made a second pipeline reuse the
  first one's training data. It also links `data_run_dir` and posts the data stage's start/done events.

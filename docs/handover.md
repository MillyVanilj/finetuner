# Handover: where the project stands (updated 2026-11-01, hackathon day 2, after #11 merged)

This is the **current-state** doc: what is merged, what is open, what to do next, and the pitch. The other
docs hold the stable parts: design in `docs/wireframe_v0.2.md`, model choices and reasoning in
`docs/decisions.md`, file formats in `docs/spec_and_io.md`, ScienceClaw internals in
`docs/wireframe_v0.2_review.md`, working rules in `CLAUDE.md`.
Everything below was checked against the code or GitHub when written; if a PR number or file named here
looks different, trust the repo and **update this file** (whoever merges or opens a PR keeps section 1 current).

---

## 1. TL;DR

- **What:** a user states a goal ("answer chemistry questions in `ANSWER:/WHY:` form"). Agents write a task
  spec, generate + verify training data, write a frozen benchmark, LoRA-tune Qwen3-4B on a Brev GPU, evaluate
  tuned vs base, decide whether to iterate, and report. Built on ScienceClaw (MIT LAMM) skills.
- **Works today (main):** every stage as a ScienceClaw skill, `scripts/run_pipeline.py` running them in order
  (dashboard ▶ Start), live dashboard, poster wall, task-spec form (M1), and the model split (section 4).
- **Merged today:** #11 model split (Gemini writes data, **OpenAI Sol** verifies data and judges, Claude Opus
  writes the benchmark), #13 BenchGen JSON-fence parsing fix, #17 BenchGen says why a job stopped + dashboard
  clears a stale "Failed" after a reset.
- **Open:**
  - **#24** five fixes before the final runs (training-speed estimate, best adapter vs forgetting, DataGen
    stall/dedupe, real format check). Before final runs: move smoke runs out of `runs/` on Brev, set
    `FTC_MAX_JOB_HOURS` if jobs may exceed 1 h. Still open: the forgetting check fires on noise at small M8 sizes.
  - **Grading + data quality** (#26, merged), from the weak NL→SQL run (tuned 48 % vs few-shot
    49 %): the HF importer had taken only the schema column, so all 30 "real" held-out items had no question
    (even Claude scored 10 %), and the Evaluator then asked for more "typical cases", which made loop 2 worse.
    Fixes: importer joins several input columns and gates rows (code checks + Claude review) before import;
    SQL answers are graded by running them (`finetune_common/sqlexec.py`) instead of the judge; the judge grades
    the final answer only (no penalty for missing or extra reasoning); misleading inputs are always 25 % of
    training data and benchmark. Re-scored by execution, that run's 70 valid items give tuned 71 %, few-shot
    70 %, zero-shot 71 %, teacher 97 %: tuning still adds nothing there, so the next lever is the data.
  - **BenchGen skips refused categories** (#25, merged): when Claude declines to write
    prompts for a category (e.g. "user asks for harmful content that should be declined"), BenchGen now drops
    that cell and moves its items to the others, instead of failing the whole benchmark after 3 restarts.
    API errors still stop it as before.
  - **#18** eval job survives `install.sh` (an eval crashed on Brev because `install.sh` re-copied the skill
    folder it was running in); `install.sh` now skips unchanged skills. Small, tested, merge it.
  - **#12 (+ #14 merged into it)** *agent mode*: the pipeline run by real ScienceClaw agents handing work
    along via the ArtifactReactor. Tested with the real reactor (GPU/API skills stubbed). **The base for agent mode.**
  - **#15** a second agent-mode implementation (written in a Claude session before we noticed #12). Mostly
    duplicates #12; port its 3 unique pieces into #12 and close it (section 5.3).
- **Deadline:** hackathon ends Sunday Nov 1 night; Google Cloud accounts are deleted then. Push everything.

## 2. Team, repos, machines

- Team of 3, GitHub accounts: `MillyVanilj` (repo owner), `owenbar` (Owen), `rohanKraghu` (Rohan).
  `NOTES.md` says who does what.
- Our repo: `github.com/MillyVanilj/finetuner` (local checkout may be called `scienceclaw-finetune`).
- ScienceClaw: `github.com/lamm-mit/scienceclaw`, installed **next to** our repo at `~/scienceclaw`
  (never edit its source). `scripts/install.sh` copies `skills/*` into it and `agents/*/agent_profile.json`
  into `~/.scienceclaw/profiles/`.
- GPU: NVIDIA Brev instance (shared Linux box, repo at `~/finetuner`). `bash scripts/setup_brev.sh` once,
  `bash scripts/install.sh` after every `git pull`. Stop the instance when idle.
- Keys live in `~/finetuner/.env` (gitignored; template `.env.example`): `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
  `OPENAI_API_KEY` (needed since #11: Sol verifies data and judges), HF token.

## 3. Pipeline and code map

| Stage | Code | Hand-off keys (reactor) |
|---|---|---|
| M1 + task spec | `ftc/m1.py`, `ftc/spec.py`, dashboard form | user goal → `task_spec.json` |
| L1 data (M2 generate, M3 verify, M4 format/dedup) | `skills/finetune-data-gen` | `spec` (+ `data_feedback_path`) → `train_path` |
| L2 benchmark (M5 write, M6 check) | `skills/finetune-bench-gen` | `spec` → `bench_dir` |
| Trainer | `skills/finetune-submit`, `skills/finetune-status`, `scripts/brev-smoke/train_sft.py` | `train_path` (+ `retrain_config_path`) → `train_job_id` → `adapter_path` |
| L3 eval (M7 task, M8 general, M9 safety) | `skills/finetune-eval` (`scripts/eval/run_eval.py` on the GPU) | `adapter_path` (+ `bench_dir`) → `eval_report_path` |
| Evaluator | `skills/finetune-decide` (fixed rules, one change per loop) | → retrain / regenerate_data / stop (`best_adapter_path`) |
| Report | main: inside `run_pipeline.py`; #12: `skills/finetune-report` + `ftc/report.py` (+ model card) | `final/final_report.json` |
| Runner without agents | `scripts/run_pipeline.py` | calls skills as subprocesses, same contract as ScienceClaw's executor |
| Dashboard | `dashboard/server.py`, `ftc/pipeline.py`, `ftc/state.py` | reads `pipelines/<id>/` (events.jsonl, pipeline.json, runner.json, wall.jsonl) |
| Poster wall | `ftc/wall.py`, `skills/finetune-wall` | agents/humans pin notes for each other |

Skill contract: one JSON object on stdout; **exit non-zero while not finished** (so the reactor retries);
output keys must only match the intended next skill's `--flags` (`tests/test_handoffs.py` enforces this).

## 4. Model split (on main since #11; reasoning in `docs/decisions.md`)

| Role | Model | Override |
|---|---|---|
| M2 generate training data | **OpenAI Sol** (`gpt-6.1-sol`) since Nov 1, was Gemini Flash (decisions.md 2b) | `DATAGEN_GEN_BACKEND`, `DATAGEN_GEN_MODEL` |
| M3 verify training data | **Claude Sonnet** (`claude-sonnet-5-5`) since Nov 1, was Sol | `DATAGEN_VERIFY_BACKEND` (`same` = the writer), `FINETUNE_ANTHROPIC_MODEL` |
| M5/M6 write + check benchmark | Claude Opus (`claude-opus-5-5`) | `BENCHGEN_MODEL` |
| M7 judge | **OpenAI Sol** | `FINETUNE_JUDGE_MODEL`, spec `models.judge` (`gpt-*` or `claude-*`) |
| M7 teacher (ceiling), M1 spec | Claude Sonnet (`claude-sonnet-5-5`) | `FINETUNE_TEACHER_MODEL`, `FTC_M1_MODEL` |
| Student | `Qwen/Qwen3-4B-Instruct-2507` | spec `models.student` |

Why: no model family checks its own work (self-preference bias). Gemini's data is checked by Sol; Opus's
benchmark is graded by Sol. Full reasoning: `docs/decisions.md` ("Update Nov 1 (Sol)").

**Still to do:** confirm the Sol model id with one real call. `gpt-6.1-sol` came from third-party
listings (The Rundown, AWS Bedrock); OpenRouter lists `gpt-6-sol`. If wrong, set `FINETUNE_OPENAI_MODEL` and
`FINETUNE_JUDGE_MODEL`. Without `OPENAI_API_KEY`, DataGen falls back to Gemini verification with a warning in
its result; the eval stops at start-up (before GPU work) with a clear message.

DataGen time budget (PR #23): ScienceClaw's executor kills skills at 30 s, so in agent mode DataGen budgets 27 s
per call and cuts each LLM call at 18 s. Sol is a reasoning model and may need longer. `run_pipeline.py` now gives
each DataGen call `$DATAGEN_TIME_BUDGET` or 540 s (+120 s timeout margin), which allows 120 s per LLM call, and
stops a DataGen stage after 4 h or 5 calls without progress. In agent mode, checks that run out of time stay
queued (nothing new is generated meanwhile); if Sol can't check even one example per call, DataGen stops with
`verifier_too_slow` and says what to change. `$FINETUNE_LLM_CALL_CAP_S` pins the per-call limit
(DataGen SKILL.md).

## 5. Agent mode (the "decentralised collective")

### 5.1 How ScienceClaw hand-offs really work (verified in its source, `artifacts/reactor.py` etc.)

- An agent's `ArtifactReactor` scans `~/.scienceclaw/artifacts/global_index.jsonl` and runs one of its allowed
  skills when a peer artifact's **payload keys match `--flags` found anywhere in that skill's `--help` text**
  (regex over the whole help, descriptions included; `argparse.SUPPRESS` flags are invisible). All matching
  keys are passed; lists → first item; dicts dropped; `True` booleans → bare flag. `--help` has a 5 s timeout.
- Skill exit 0 ⇒ artifact consumed (per agent, `~/.scienceclaw/artifacts/<agent>/consumed.txt`). Non-zero ⇒
  retried next cycle. Exit with "required"/"unrecognized arguments" in stderr ⇒ executor retries with
  `--query` only (surprising).
- The skill's stdout JSON becomes the child artifact's payload with `parent_artifact_ids=[parent]` and the
  parent's `investigation_id`. Lineage + content hash come free. If `~/.scienceclaw/infinite_config.json`
  is logged in, every artifact is auto-published to Infinite.
- Allowed skills come from **top-level `preferred_tools`** in the profile (`setup.py` writes
  `preferences.tools`, which the reactor ignores ⇒ unrestricted). No `interests`/`research`/`demo_topic`
  ⇒ the agent only reacts.
- `investigation_id` in the profile scopes an agent to one run (exact string match).
- Skills run with the **caller's Python** (`sys.executable`), cwd = ScienceClaw root, inherited env. Reactor
  reactions time out at **30 s**. Two unconsumed artifacts for the same skill get **merged** (oldest two).
- Same machine only (payloads are read from local stores). Cross-machine *needs* exist via Infinite
  `/api/needs-signals` but are untested by us.
- `autonomous/start_daemon.sh` is broken for multiple agents (no `--profile`; `once` mode passes `--once`,
  which the daemon doesn't accept). The heartbeat daemon does nothing without an `api_key` in
  `~/.scienceclaw/infinite_config.json`. ScienceClaw's built-in auto-upvote calls a non-existent method.

### 5.2 #12 (Rohan, + Owen's #14): the base

- `python3 -m ftc.cli agents kickoff <pipeline_dir>` seeds `{"spec": ...}` from producer `Kickoff`.
- 7 profiles: DataAgent, BenchAgent, TrainAgent, StatusAgent, EvalAgent, DecideAgent, ReportAgent
  (`disable_llm`, `disable_platform`).
- `python3 -m ftc.cli agents start|status|stop` runs one ScienceClaw **heartbeat daemon per profile**
  (needs the Infinite api_key, see above).
- New `finetune-report` skill (held-out once + final report + model card); `ftc/report.py` shared with
  `run_pipeline.py`. finetune-decide hides `--spec` so the kickoff doesn't wake it.
- #14: DataGen run id `<pipeline_id>-l<loop>` and a `.pipeline` pointer so submit/eval/decide find the
  pipeline without `$FTC_PIPELINE_DIR`.
- `tests/test_agent_mode.py` drives the full chain through ScienceClaw's real `ArtifactReactor`.
  Not yet run with real daemons on Brev.

### 5.3 #15: what to port into #12, then close #15

#15 adds `agents/` (6 profiles), `scripts/run_agents.py`, a `finish()` refactor of `run_pipeline.py`
(superseded by #12's `ftc/report.py`) and one skill fix. Unique, worth porting:
1. **`FINETUNE_STATUS_REQUIRE_DONE=1`** (`skills/finetune-status/scripts/finetune_status.py`): without
   `--require-done`, finetune-status exits 0 while training is still running, so the `train_job_id` artifact
   is consumed on the first poll and the chain stalls. The reactor can't add the flag. **Check whether #12
   already solves this** (its chain test runs each step once with stand-ins, which may hide it). If not, port
   this or make submit's output carry `"require_done": true`. (Checked after #11 merged: #12 adds a
   `--require-done` flag but finetune-submit never outputs `require_done`, so the reactor won't pass it. Still open.)
2. **Daemon-free runner**: `run_agents.py` runs one OS process per agent, each calling
   `ArtifactReactor(name, profile, ArtifactStore(name)).react(limit=1)` in a loop. No Infinite api_key, none of
   the heartbeat's extra work (Infinite sync, LLM calls), ~5 s cycle. Useful as a fallback if the daemons
   misbehave in the demo. Could become `ftc.cli agents start --lite`.
3. **Hand-off mirroring**: the coordinator tails the global index and writes each hand-off (agent, artifact
   id, parent artifacts) to `events.jsonl` and the poster wall in the producing agent's name. This is what
   makes the collective *visible* in the dashboard. Port as a small watcher in `ftc/agents.py`.

### 5.4 Agent-mode risks to check on Brev

- Which flags of `gen_data.py` / `bench_gen.py` the reactor sees (most are SUPPRESS): `--spec` must be visible.
- 30 s reaction timeout: finetune-eval and finetune-submit must launch background jobs and return fast.
- One pipeline at a time (artifact merge pitfall).
- Generic payload keys (`model`, `epochs`, `lr`, `format`) wake finetune-submit: keep them out of outputs.
- Concurrent appends to `global_index.jsonl` from several processes are unlocked (small lines; fine in practice).

## 6. Running things

```bash
# Brev, once:   bash scripts/setup_brev.sh      after each pull:   bash scripts/install.sh
python3 dashboard/server.py                      # live view, create a pipeline, ▶ Start = run_pipeline.py
python3 scripts/run_pipeline.py --pipeline ~/finetuner/pipelines/<id> [--skip-general]
python3 scripts/simulate_pipeline.py             # fake run for the dashboard (no GPU/API)
# agent mode (#12):  python3 -m ftc.cli agents kickoff <pipeline_dir> && python3 -m ftc.cli agents start
# agent mode (#15):  ~/scienceclaw/.venv/bin/python scripts/run_agents.py --pipeline <pipeline_dir>
```

GPU rule (CLAUDE.md): start training only via `finetune-submit` (its pre-flight checks decide; long jobs are
clamped, reasons saved in `run.json`). For other GPU work check yourself: GPU idle, short, within budget.

## 7. Testing (incl. on a Mac without ScienceClaw)

- `python3 -m pytest tests/ -q` (install pytest in a venv; macOS system python has none).
- Skill tests are plain scripts: `python3 -B skills/<skill>/tests/test_*.py`.
- Without ScienceClaw, create a stub: a folder with `core/llm_client.py` containing
  `class LLMClient: def __init__(self, agent_name="x", backend="gemini"): ...  def call(self, *a, **k): return ""`
  and run with `SCIENCECLAW_DIR=<that folder>`. Then everything passes except `test_integration.py`
  (needs `core.skill_registry`) and agent-mode tests (skipped).
- Known failures on main: `skills/finetune-bench-gen/tests/test_link.py` has 2 failing checks
  ("benchmark frozen -> DataGen finishes…", "feedback run id…"). Still failing after #11/#13/#17 merged;
  probably fixed by #14 (inside #12).
- macOS has no `timeout` command.
- Offline DataGen tests set `DATAGEN_VERIFY_BACKEND=same` so they don't call OpenAI.

## 8. Suggested next steps (in order)

1. ~~Merge #13~~ and ~~#11~~: done. Tell the team to add `OPENAI_API_KEY` to `.env` on Brev, `git pull`,
   `bash scripts/install.sh`. Confirm the Sol model id with one real call (section 4).
2. **Merge #18** (eval crash on Brev when `install.sh` ran mid-eval).
3. Merge the DataGen budget fix (#23, section 4).
4. **Review + merge #12.** It predates #11: merge `main` into it and resolve `docs/decisions.md` overlap.
   Run the real chain on Brev once (chem_qa example, `--skip-general`, 1 loop).
5. **Port #15's three pieces into #12** (section 5.3), then close #15.
6. **Make the collective visible for the demo** (section 9.3): dashboard hand-off/lineage view, one Infinite
   post with the artifact DAG, per-agent comments.
7. Push often; save adapters to HF Hub (`--push-to-hub`) before accounts are deleted.

## 9. Pitch: telling the "decentralised agent collective" story (honestly)

### 9.1 What is genuinely true once #12 (+ #15's watcher) is merged and run

- **No central script in agent mode.** Each stage is an autonomous ScienceClaw agent with exactly one
  capability. Nobody tells DataAgent to start or TrainAgent to train: each agent watches the shared artifact
  board and acts when a peer posts something it can use (stigmergy: coordination through shared artifacts,
  like ants with pheromone trails).
- **Closed feedback loop between agents.** The Evaluator posts either a retrain config (TrainAgent picks it
  up) or "more data on weak categories" (DataAgent picks it up). The system improves itself loop by loop.
- **Checks and balances between model families.** Sol writes data, Claude Sonnet audits it; Opus writes the exam,
  Sol grades it; BenchAgent never sees the training data. No agent grades its own work. The decision rule
  ("tuned must beat base few-shot") compares two non-Claude systems.
- **Parallel work.** DataAgent and BenchAgent work on the same spec at the same time.
- **Full provenance.** Every artifact carries its parents and a content hash: the final adapter traces back
  through eval → training → data → spec. Training runs save config, dataset hash, metrics (`run.json`).
- **Agents talk.** The poster wall shows requests and replies between agents, live in the dashboard; humans
  can join in.
- **Plugs into the ScienceClaw ecosystem unchanged.** Our six skills are ordinary ScienceClaw skills and our
  agents ordinary profiles; any ScienceClaw agent could pick up our artifacts. We found and documented several
  ScienceClaw pitfalls (section 5.1) on the way.

### 9.2 Don't claim (not true yet)

- Cross-machine collaboration (the reactor is same-machine only).
- LLM "reasoning" in the Evaluator (it is deliberate fixed rules: predictable, one change per loop; say that
  as a strength).
- Zero human involvement (a human states the goal and can edit the spec; that is a feature: human-in-the-loop
  at the cheapest point).
- That agents post to Infinite unless we actually build and run it (9.3).
- If using #15's watcher: wall posts are *relayed* in the agent's name by the watcher, not written by the
  agent process.

### 9.3 Cheap additions that make the story stronger (each a few hours or less)

1. **Lineage view in the dashboard:** draw the artifact DAG for the pipeline (from the global index lines with
   this `investigation_id`). One picture that says "collective".
2. **ReportAgent posts to Infinite:** `InfiniteClient.create_post(...)` then `attach_post_artifacts(post_id,
   metadata)` fills the post's Dataflow/Graph tabs (example: ScienceClaw `skills/infinite/scripts/post_artifacts_dag.py`).
   10-karma rule may block new agents; `create_community(..., min_karma_to_post=0)` exists (unverified).
3. **Each agent comments its result on that post** (one Infinite identity per agent via
   `--config ~/.scienceclaw/profiles/<Name>/infinite_config.json`, or ScienceClaw's `seed_post_id`). A thread
   where DataAgent, BenchAgent, TrainAgent, EvalAgent and the Evaluator each report is the clearest demo.
4. **Agents write their own wall notes** (call `ftc.wall` from inside the skills, e.g. the Evaluator's
   request to DataAgent) instead of a relay.
5. Stretch, risky: a second machine fulfils a need via Infinite needs-signals.

### 9.4 Demo script (5 min)

1. Dashboard → new pipeline from `examples/specs/chem_qa.json` (M1 asks clarifying questions).
2. Start agent mode. Show the agents' logs side by side: Data + Bench start in parallel, no one called them.
3. Wall: BenchAgent → DataAgent contamination note; Evaluator → DataAgent "weak on stoichiometry, +150".
4. Lineage graph: final adapter ← eval ← train ← data ← spec.
5. Final report: tuned vs base few-shot on the sealed held-out split, general-ability drop, safety.
6. (If built) the Infinite post with per-agent comments and the DAG tab.

## 10. Rules digest (from CLAUDE.md)

Never commit secrets (`.env`, `*_config.json`, `.scienceclaw/` are ignored). Ask before deleting files or
force-pushing. Don't edit ScienceClaw's source. Push often. Explain changes simply (not every teammate is an
experienced programmer). Work in a branch, never push to main; open PRs. Claude Code sessions here run in
worktrees under `.claude/worktrees/`.

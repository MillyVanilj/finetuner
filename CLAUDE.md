# CLAUDE.md — Project context

> **Read first:** `docs/wireframe_v0.2.md` (the design, and which code implements each box), `docs/decisions.md` (model choices and reasoning, compute) and `docs/spec_and_io.md` (the task spec and every hand-off file format). ScienceClaw internals and the wireframe critique: `docs/wireframe_v0.2_review.md`.

## What we're doing

We are a team of 3 at the ScienceClaw hackathon (MIT, Oct 31 – Nov 1, 2026). We're using the ScienceClaw multi-agent framework to automate fine-tuning: a user states a goal ("answer chemistry questions in `ANSWER:/WHY:` form"), and agents write a task spec, generate and verify training data, build a benchmark, LoRA fine-tune Qwen3-4B on a Brev GPU, evaluate tuned vs base, decide whether to iterate, and publish the result, with minimal human work.

Hackathon ends Sunday night. Favor working demos over perfect code. Keep things simple.

## The pipeline (wireframe v0.2)

| Stage | Code | Takes → gives (reactor keys) |
|---|---|---|
| M1 + task spec | `ftc/m1.py`, `ftc/spec.py` | user goal → `task_spec.json` |
| L1 data (M2–M4) | `skills/finetune-data-gen` | spec → `train_path` |
| L2 benchmark (M5–M6) | `skills/finetune-bench-gen` | `spec` → `bench_dir` (dev / held-out) |
| Trainer | `skills/finetune-submit`, `skills/finetune-status` | `train_path` (+ `retrain_config_path`) → `train_job_id` → `adapter_path` |
| L3 eval (M7–M9) | `skills/finetune-eval` | `adapter_path` + `bench_dir` → `eval_report_path` |
| Iterate? (Evaluator) | `skills/finetune-decide` | `eval_report_path` → `retrain_config_path` + `train_path` · `data_feedback_path` · stop (`best_adapter_path`) |
| Output (ReportAgent) | `skills/finetune-report`, `ftc/report.py`, `ftc/publish.py` | `best_adapter_path` → held-out once → `final/final_report.json` (verdict, side-by-side examples), `model_card.md` → HF Hub (private by default), Infinite post (opt-in) |
| Runner without agents | `scripts/run_pipeline.py` (dashboard ▶ Start) | runs the stages above in order, converts Eval's report to §3.4 |
| Live view | `dashboard/`, `ftc/pipeline.py`, `scripts/simulate_pipeline.py` | reads the pipeline folder; the **Result** card at the top shows the final model |
| Poster wall | `ftc/wall.py`, `skills/finetune-wall`, dashboard 📌 card | agents (and humans) pin notes for each other in `wall.jsonl` |

Every training run saves its config, dataset hash, metrics and adapter location (`run.json`, `metrics.json`), so results are reproducible.

## About ScienceClaw

- Repo: https://github.com/lamm-mit/scienceclaw (by MIT LAMM). Read its README.md, SETUP_GUIDE.md and ARCHITECTURE.md before changing anything.
- It runs autonomous science agents. Each agent has a profile (biology, chemistry, mixed…) that sets which of its 300+ tools ("skills") it can use.
- An agent gets a topic → picks tools → runs them → saves each result as an artifact → writes findings → posts to Infinite.
- Infinite (https://lamm.mit.edu/infinite) is a Reddit-like site where agents post findings and comment on each other's posts. Works across machines. A new agent needs 10 karma before it can post (earned by commenting/upvoting).
- ArtifactReactor: agents broadcast unmet needs to `~/.scienceclaw/artifacts/global_index.jsonl`; other agents pick them up automatically.
  - ⚠️ This only works for agents on the same machine (shared file in the home dir). Cross-machine coordination is NOT built yet.
- Artifacts live in `~/.scienceclaw/artifacts/{agent}/store.jsonl` (outside any repo).

## Setup (on the shared Linux cloud machine)

**Use the scripts:** clone this repo to `~/finetuner`, then run `bash scripts/setup_brev.sh` once (installs ScienceClaw at `~/scienceclaw` + a separate training env + `.env`). Run `bash scripts/install.sh` after every `git pull`. The manual steps below are what the script automates (ignore the `setup.py --quick` lines: we hand-write profiles in `agents/`).

```bash
git clone https://github.com/lamm-mit/scienceclaw.git
cd scienceclaw
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./install_scienceclaw_command.sh

export LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=...        # from Claude Console, never commit
export INFINITE_API_BASE=https://lamm.mit.edu/infinite/api

python3 setup.py --quick --profile biology   --name "BioAgent-1"
python3 setup.py --quick --profile chemistry --name "ChemAgent-1"
```

Useful commands:

```bash
scienceclaw-post --agent BioAgent-1 --topic "..." --community biology --dry-run
python3 skill_catalog.py --suggest "topic"
./autonomous/start_daemon.sh once
tail -f ~/.scienceclaw/heartbeat_daemon.log
```

## Our repo layout

Our repo is separate from ScienceClaw (ScienceClaw is installed next to it, not inside it).

```
finetuner/
  skills/     # our ScienceClaw skills (finetune-*); scripts/install.sh copies them into ScienceClaw
  agents/     # agent profiles (<Name>/agent_profile.json), copied to ~/.scienceclaw/profiles (none yet)
  ftc/        # shared Python: task spec, pipeline folder + event log, M1, dashboard state
  dashboard/  # live pipeline view (python3 dashboard/server.py)
  scripts/    # setup_brev.sh, install.sh, train_sft.py (brev-smoke/), simulate_pipeline.py
  examples/   # example task specs
  tests/      # offline tests: python3 -m pytest tests/ -q
  docs/       # design, decisions, formats, progress logs
  results/    # exported artifacts, figures, findings
  NOTES.md    # who is doing what
```

## Compute available

- Claude API ($100/person, Console) — use as the agents' LLM backend.
- Google Cloud VM (32 vCPUs) + Gemini API key — max 2 accounts per team. Accounts auto-delete Sunday night.
- NVIDIA Brev ($150/person) — GPUs by the hour.
- Nebius Scientific AI Platform — 65 hosted models (AlphaFold3, Boltz-2, OpenMM…), callable via MCP with the team's Scientific API key.
- Nebius H100s — 16 shared by the whole event. Use 1 GPU max, short jobs.
- OpenAI API ($50/person).

## Rules for the agent (you)

- Never commit secrets. API keys go in `.env` (gitignored). Also ignore `*_config.json` and `.scienceclaw/`.
- Ask before deleting files or force-pushing.
- GPU jobs don't need a human OK, but start training only through `skills/finetune-submit`: its automatic pre-flight checks decide (GPU free, not a duplicate, rate limit, disk; jobs longer than 1 h or over the pipeline GPU budget are clamped to fewer epochs, not refused), and the reasons are saved in `run.json`. For other GPU work (eval, ad-hoc tests), apply the same checks yourself: GPU idle, short, within budget. Stop the Brev instance when idle.
- Don't edit the ScienceClaw source unless asked; put our code in our repo.
- Push work to GitHub often. Cloud accounts are deleted after the event.
- Explain changes simply and briefly. Teammates are not all experienced programmers.

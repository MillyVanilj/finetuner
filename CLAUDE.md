# CLAUDE.md — Project context

## What we're doing

We are a team of 3 at the ScienceClaw hackathon (MIT, Oct 31 – Nov 1, 2026). We're using the ScienceClaw multi-agent framework to automate the model fine-tuning process: a team of agents that takes a goal (e.g. "make a small model better at predicting protein properties"), then finds data, prepares it, fine-tunes a model, evaluates it, and reports results — with minimal human work.

Hackathon ends Sunday night. Favor working demos over perfect code. Keep things simple.

## Project plan (suggested — adjust as we go)

Split the pipeline into specialist agents that hand work to each other through artifacts:

| Agent | Job | Output artifact |
|-------|-----|-----------------|
| DataAgent | Find and download a dataset (Hugging Face, papers, databases) | `dataset` |
| PrepAgent | Clean, format, and split train/val/test | `prepared_dataset` |
| TrainAgent | Pick base model + settings, launch the fine-tuning job | `trained_model` |
| EvalAgent | Score base vs fine-tuned model on the test set | `eval_report` |
| ReportAgent | Summarize results and post to Infinite | `post` |

Each agent broadcasts what it needs next (e.g. TrainAgent needs `prepared_dataset`), so the ArtifactReactor can chain them. All agents run on the same machine so hand-offs work.

**Start small, then scale:**
1. MVP: one tiny model (e.g. a small Hugging Face model) + one small dataset, LoRA fine-tune, end-to-end on one GPU. Make this work first.
2. Then automate choices (base model, learning rate, epochs) and compare runs.
3. Then add more datasets/models or a hyperparameter search.

**Fine-tuning options (pick the simplest that works):**
- Brev GPU (L4/A100) + Hugging Face `transformers` / `peft` (LoRA). Default choice.
- NVIDIA NeMo / BioNeMo for bigger or biology-specific models.
- Vertex AI serverless Gemini supervised fine-tuning (Google Cloud account) — no GPU management needed.
- Nebius H100s — shared by everyone; 1 GPU, short jobs only.

Every training run must save: config used, dataset version, metrics, and where the model weights are stored — so results are reproducible and traceable.

## About ScienceClaw

- Repo: https://github.com/lamm-mit/scienceclaw (by MIT LAMM). Read its README.md, SETUP_GUIDE.md and ARCHITECTURE.md before changing anything.
- It runs autonomous science agents. Each agent has a profile (biology, chemistry, mixed…) that sets which of its 300+ tools ("skills") it can use.
- An agent gets a topic → picks tools → runs them → saves each result as an artifact → writes findings → posts to Infinite.
- Infinite (https://lamm.mit.edu/infinite) is a Reddit-like site where agents post findings and comment on each other's posts. Works across machines. A new agent needs 10 karma before it can post (earned by commenting/upvoting).
- ArtifactReactor: agents broadcast unmet needs to `~/.scienceclaw/artifacts/global_index.jsonl`; other agents pick them up automatically.
  - ⚠️ This only works for agents on the same machine (shared file in the home dir). Cross-machine coordination is NOT built yet.
- Artifacts live in `~/.scienceclaw/artifacts/{agent}/store.jsonl` (outside any repo).

## Setup (on the shared Linux cloud machine)

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
our-project/
  agents/     # agent profiles we created (copies from ~/.scienceclaw)
  skills/     # new tools we write
  scripts/    # helper scripts (e.g. exporting/syncing artifacts)
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
- Ask before deleting files, force-pushing, or launching GPU jobs.
- Don't edit the ScienceClaw source unless asked; put our code in our repo.
- Push work to GitHub often. Cloud accounts are deleted after the event.
- Explain changes simply and briefly. Teammates are not all experienced programmers.

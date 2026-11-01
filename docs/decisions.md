# Decisions & reasoning (living doc)

*Last updated 2026-11-01. If you're an agent or teammate picking this up: read this first, then `docs/wireframe_v0.2.md` (the design and which code implements each box) and `docs/wireframe_v0.2_review.md` for the detailed critique and ScienceClaw internals. If you change a decision, update this file and say why.*

## What we're building

The **fine-tuning collective** from wireframe v0.2: a user states a goal → agents write a task spec → generate + verify training data → generate a benchmark → LoRA fine-tune a small open model → evaluate tuned vs base → decide whether to iterate → publish the adapter, eval report and model card, and post to Infinite.

(The protein/ESM plan in older notes is **superseded** by this.)

### Glossary (box names from the wireframe)

| Label | Meaning |
|---|---|
| M1 | Prompt verification: clarifies the user's goal → task spec. Also collects 10–30 real examples from the user. |
| L1 (M2–M4) | Training data: M2 generate, M3 verify, M4 format to JSONL (script, no LLM), then contamination check |
| L2 (M5–M6) | Benchmark: M5 generate items, M6 verify answer keys/difficulty → dev split (every loop) + held-out split (once, at the end) |
| Trainer | LoRA fine-tune, `finetune-submit` / `finetune-status` |
| L3 (M7–M9) | Eval: M7 task score, M8 general ability (lm-eval-harness), M9 behaviour (stretch) |
| Evaluator | Iterate or stop |
| lm-eval-harness | EleutherAI's open-source tool for running standard exams (MMLU, GSM8K, IFEval) on a model. We use it to check fine-tuning didn't make the model worse at everything else. |

## Compute we have (from the event's compute deck)

| Provider | What | Our use |
|---|---|---|
| NVIDIA Brev | $150 credits/person ($450 pooled); L4, L40S, A100, H100, H200 | **Main machine**: agents + training + eval run here (L40S) |
| Google AI Studio | Tier-1 Gemini key: 1k–4k requests/min, 2–4M tokens/min, Flash / Flash-Lite / Pro, Batch API | **Bulk data generation (M2)** |
| Google Vertex AI | Serverless Gemini supervised fine-tuning; max 2 accounts/team; **deleted Sunday night** | Optional 2nd trainer backend / fallback |
| Claude API | $100/person | Verifier (M3) + judge (M7) |
| OpenAI API | $50/person | Benchmark writer (M5/M6) |
| Nebius H100s | 16 shared across the event; jobs queue ≤30 min, then fail | Backup GPU only |
| Nebius Scientific AI | 65 hosted science models via MCP | Not needed for this project |
| DGX Spark (on site) | 128 GB GPU desktops, request via Discord | Not needed for a 4B model |

⚠️ **Never commit the compute deck or any credit/sign-up links.** They are event-only, and sharing them gets you disqualified.

## Decision 1: train an open model (Qwen3-4B-Instruct-2507) on Brev, not Gemini

**Choice:** `Qwen/Qwen3-4B-Instruct-2507` + LoRA, on a Brev L40S.

**Why:**
- LoRA needs the model's weights. Open models (Qwen, Llama, Gemma, Nemotron…) have them; Claude/GPT/Gemini don't.
- **Small model beats prompted baseline** is a stronger demo than tuning an already strong Gemini.
- M8 (lm-eval-harness) needs a model we run ourselves. A tuned Gemini only exists as a Google endpoint.
- We keep the result (adapter on Hugging Face). A tuned Gemini can't be downloaded and disappears with the Google accounts on Sunday.
- 4B fits one GPU, trains in minutes, and has an Apache-2.0 license.
- Use the **Instruct-2507** version: non-thinking only. Plain `Qwen3-4B` has "thinking mode" on by default, which breaks training and eval formatting. Skip Qwen3.5-4B (multimodal, newer tooling requirements: a stretch goal at most).

**Rejected:** Kimi (K2 ≈1T, K3 ≈2.8T parameters, far too big for one GPU; no small version found).
**Open question:** NVIDIA Nemotron small models (NVIDIA is the top sponsor). Not yet compared; worth a check.

## Decision 2: Gemini still does a lot of work

1. **Gemini Flash = bulk data generator (M2).** Its rate limits + Batch API make thousands of examples cheap and fast, saving Claude/OpenAI credits.
2. **Gemini fine-tuning = optional 2nd Trainer backend** (stretch). Same JSONL data with a small format change, so the agents could compare "Qwen on Brev" vs "Gemini on Vertex". Also a fallback if GPU setup fails.

## Decision 3: different model families per role

Same-family models favour their own outputs (self-preference bias), so:

| Role | Model family |
|---|---|
| M2 generate training data | Gemini (Flash) |
| M3 verify training data, M7 judge | Claude |
| M5/M6 write + check benchmark | Claude Opus (`claude-opus-5-5`), changed from OpenAI on Nov 1, see below |
| Student (being trained) | Qwen |

(This updates the review doc, which originally had Claude as the generator. Gemini's free high-volume key changed that.)

*Update Nov 1:* the team chose **Claude Opus** for the benchmark (BenchGen, `skills/finetune-bench-gen`). Caveat: the Evaluator's judge and teacher are also Claude, so Claude may grade Claude-written items generously. Mitigations in place: an independent solve (M6a) checks every answer key, the benchmark is frozen, and the decision rule is "tuned must beat base few-shot", which compares two non-Claude systems on the same items.

## Decision 4: always include real data where possible

A benchmark written by the same kind of model as the training data can just reward "learned the generator's style". Sources, in order (record which one was used):
1. The user's own examples (asked for in M1).
2. Hugging Face Hub search + an LLM relevance/license check. *Built:* `ftc/real_data.py`, run by M1 when the user gives no examples.
3. Real *inputs* from science sources via ScienceClaw skills (PubMed, arXiv, UniProt, PubChem…), with LLM-written answers.
4. Nothing found → synthetic only, and the report flags lower confidence.

Report scores on the real and synthetic parts of the held-out set separately. Suggested demo task (always has real inputs): *extract structured JSON from PubMed abstracts*.

## Decision 5: how the system runs (where agents live)

Everything runs on **one Brev GPU machine**:

```
Brev machine
├── scienceclaw/      ScienceClaw program (cloned, not edited by us)
├── finetuner/        THIS repo: our skills, agent profiles, scripts
│     install script copies skills → scienceclaw/skills/,
│     profiles → ~/.scienceclaw/profiles/
└── ~/.scienceclaw/   agents' home: profiles, memory, shared artifact file
```

- **Decided: "side by side", with ScienceClaw unedited.** ScienceClaw is cloned separately (`~/scienceclaw`), not copied into this repo. Setup: `bash scripts/setup_brev.sh` (once per machine). After every `git pull`: `bash scripts/install.sh` (copies `skills/*` into ScienceClaw and `agents/<Name>/agent_profile.json` into `~/.scienceclaw/profiles/`). Agents hand work along by matching output keys to input flags, so no ScienceClaw code edits are needed.
- ScienceClaw does **not** read `.env` itself. Load keys into the shell first: `set -a; source ~/finetuner/.env; set +a`.
- An **agent** = a profile JSON + a ScienceClaw heartbeat process started with it (`--profile <Name> --interval 60`). There are 7 agents = 7 processes (`agents/`, started with `python3 -m ftc.cli agents start`, see `docs/spec_and_io.md` §8); they hand work along through the shared artifact file.
- Laptops are only for editing code and git. Stop the Brev machine when idle (billed hourly).
- **Build `scripts/run_pipeline.py` first:** it runs the same skills in order without agents. It's easier to debug, and works as a demo fallback.
- ScienceClaw limits to design around (details in the review doc): 45 s per skill run → slow steps start background jobs and later steps poll; no asking the user mid-run → M1 is a CLI pre-step; no built-in loops → iterate through output keys like `retrain_config_path`.

## Decision 6: training & eval defaults

- Script: `scripts/brev-smoke/train_sft.py` (Owen's). TRL `SFTTrainer`, LoRA all-linear, r=16, α=32, lr 1e-4–2e-4 (script default 2e-4; TRL's own default 2e-5 is too low for LoRA), 2 epochs, bf16, loss on assistant replies only.
- **Trainer skills (built):** `skills/finetune-submit` (validates data, writes `run.json`, starts `train_sft.py` in the background, returns `train_job_id`; refuses when the GPU is busy; takes Evaluator overrides via `--retrain-config-path`) and `skills/finetune-status` (`--train-job-id` → running % / done with `adapter_path` / failed with log). Hand-off keys: `train_path` (DataGen output = submit's `--train-path`) → submit → `train_job_id` → status → `adapter_path` → Evaluator. Put status in a different agent from submit (agents ignore their own artifacts). Runs live in `~/finetuner/runs/<job_id>/`.
- Every run saves `run.json`: spec, model IDs + prompts, LoRA config, seeds, dataset SHA-256s, git commit, metrics, adapter location.
- **Eval (L3, built):** `skills/finetune-eval` (`--adapter-path` + `--bench-dir` → background job → `eval_report_path`). Its report format still differs from `docs/spec_and_io.md` §3.4; `run_pipeline.py` converts it (`ftc/report_convert.py`, see §7 there).
- **Evaluator (built):** `skills/finetune-decide` (`--eval-report-path` → `decision.json`; hands on `retrain_config_path` + `train_path`, or `data_feedback_path`, or stops with `best_adapter_path`). Rules and reasoning (when to retrain with new settings, when to try LoRA r=64, when to send failure categories back to DataGen, when to stop): `docs/evaluator-notes.md`.
- Eval report always shows: base zero-shot, **base few-shot (must beat this)**, teacher (ceiling), tuned, all with 95% confidence intervals.
- **GPU jobs start automatically (no human OK), and are clamped, not refused.** `finetune-submit` runs pre-flight checks and decides **launch / reuse / wait / refuse**:
  1. GPU free: no training job alive, and < 2 GB GPU memory used by other programs. Otherwise **wait** (retried next cycle).
  2. Time allowed = min(1 h per job (`FTC_MAX_JOB_HOURS`), what's left of the pipeline's `budget.max_gpu_hours`). Estimated length (learned from past runs' speed; default 0.5 s per example-epoch + 30 s startup) over that → **clamp**: lower the epochs (even below 1, i.e. part of the data) so it fits. The original request is kept as `requested_config` in run.json.
  3. Not a duplicate: identical data hash + requested settings already trained → **reuse** that job, no GPU time.
  4. Rate limit: ≥ 6 jobs started in the last hour → **wait** (`FTC_MAX_JOBS_PER_HOUR`), against runaway agents.
  5. Job count per pipeline > 2 × `targets.max_loops`: **warning only**.
  6. **Refuse** only when nothing can be clamped: less than 0.25 epoch fits (budget used up), or < 5 GB disk.
  Every decision + reasons go into `run.json` (`preflight`) and the pipeline event log. `--force` skips clamping/duplicate/rate (not GPU-busy or disk) and records what it overrode.
- **Task spec v2, all hand-off file formats, the event log and the live dashboard:** `docs/spec_and_io.md`. Spec v2 is a superset of DataGen's v1 (`ftc/spec.py`); every pipeline gets a folder `~/finetuner/pipelines/<id>/` that links the DataGen run and Trainer jobs; `python3 dashboard/server.py` shows it live.

## Open: reconcile with `docs/setup-and-agent-plan.md` (Owen)

Agreed by both docs: Qwen3-4B-Instruct-2507, Brev as the main GPU, Vertex Gemini tuning as fallback, async submit + status for training, loop cap, push adapters to HF.

To settle as a team:
1. **Who uses Gemini?** Owen's plan: Gemini = the agents' own LLM ("brains"), Claude/OpenAI = data generation + judge. This doc: Gemini = bulk data generator (M2). Both keep generator ≠ judge family, so either works. Pick one and update both docs.
   *Update: the code follows this doc for M2: DataGen runs with `LLM_BACKEND=gemini` (`docs/datagen-progress.md`) and spec v2 defaults to `models.generator` = Gemini, `models.judge` = Claude. Which LLM the agents themselves run on is still open.*
2. **Upgrade path:** Owen's plan suggests Qwen3.5-4B/9B later. Fine as a stretch goal, but it needs transformers v5 and a newer vLLM, so do it only once the loop works end-to-end.

Things to fix in the setup plan (checked against ScienceClaw's code):
- **Step 3 "set `preferences.tools`" won't work.** The heartbeat and reactor read a top-level `preferred_tools` list (`artifacts/reactor.py:405`, `autonomous/loop_controller.py:1359`). `setup.py` writes `preferences.tools` (`setup.py:360`), which they ignore. Hand-write `preferred_tools` in each profile.
- **Step 2 edits ScienceClaw source** (`needs.py`, `artifact.py`). This is avoidable: the reactor chains agents by matching **keys in one skill's JSON output** to **`--flags` of the next skill** (`artifacts/reactor.py:603`, `:931`). So emit specific keys like `train_path` and take `--train-path`, with no core edits. (Our rules say don't edit ScienceClaw unless needed.)
- **`skills/finetune-data-gen/scripts/gen_data.py`:**
  - Defaults to `--format summary`, which prints plain text. ScienceClaw doesn't add `--format json` for you (`core/skill_executor.py:135-155`), so the output becomes `{"output": "..."}` and no next agent can chain on it. Default to JSON.
  - One LLM call for 20 examples with `max_tokens=8000` can exceed the **45 s** skill time limit. Generate in small batches, or run as a background job for larger sets.
  - No verification (M3), dedup or contamination check yet: expected for a first version.
  - *Update (Owen's db96ae9): the time limit, verification, dedup and contamination hook are now done, and the output key is `train_path` (adopted as the standard). The `--format summary` default still holds: check how it is called.*

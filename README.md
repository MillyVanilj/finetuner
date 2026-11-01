# finetuner
building an automated fine-tuner with an agent collective

A user states a goal; ScienceClaw agents turn it into a task spec, generate and verify training data, build a benchmark, LoRA fine-tune Qwen3-4B on a Brev GPU, evaluate tuned vs base, decide whether to iterate, and publish the result.

- **Design:** [`docs/wireframe_v0.2.md`](docs/wireframe_v0.2.md) (diagram, and which code implements each box)
- **Decisions and reasoning:** [`docs/decisions.md`](docs/decisions.md)
- **Task spec and hand-off formats:** [`docs/spec_and_io.md`](docs/spec_and_io.md)
- **Working with this repo (people and agents):** [`CLAUDE.md`](CLAUDE.md)

```bash
bash scripts/setup_brev.sh                           # once per GPU machine
bash scripts/install.sh                              # after every git pull: copies skills/ into ScienceClaw
python3 dashboard/server.py                          # live pipeline view on http://localhost:8765
python3 scripts/simulate_pipeline.py --speed 2       # a fake 3-loop run to watch, no GPU needed
python3 scripts/run_pipeline.py --pipeline ~/finetuner/pipelines/<id>   # a real run, no agents (or ▶ in the dashboard)
python3 -m ftc.cli agents start                      # agent mode: 7 ScienceClaw agents (then: agents kickoff <pipeline_dir>)
python3 -m pytest tests/ -q                          # offline tests
```

"""Shared code for the fine-tuning collective's skills (DataGen, BenchGen, ...).

Lives in skills/finetune-data-gen/scripts/finetune_common/ because install.sh copies only
skills/* into ScienceClaw. Other skills add that folder to sys.path (see bench_gen.py).

Modules:
  llm        - load ScienceClaw's llm_client without core/__init__, hard per-call timeouts, logging
  jsonparse  - robust JSON extraction from LLM replies + call_json_list (retry helper)
  timing     - batch planning for the skill executor's time limit
  spec       - task spec loading, M1-lite spec drafting, stable run ids
  textsim    - normalization + near-duplicate index (numbers must match + word overlap)
  jsonl      - small JSON/JSONL file helpers

Settings are module globals (e.g. llm.LLM_CALL_CAP_S, timing.CALL_OVERHEAD_S) so callers and
tests can change them in one place.
"""

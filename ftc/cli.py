"""Shell-friendly front door to the pipeline record (run from the repo root, or set PYTHONPATH).

    python3 -m ftc.cli new examples/specs/chem_qa.json            # create a pipeline folder, print its path
    python3 -m ftc.cli link  <pipeline_dir> data_run_dir ~/.scienceclaw/finetune/runs/chem-v1
    python3 -m ftc.cli link  <pipeline_dir> train_job_id 20261101-093000-ab12cd --append
    python3 -m ftc.cli emit  <pipeline_dir> train start "LoRA r=16 on 600 examples"
    python3 -m ftc.cli loop  <pipeline_dir> 2 "more data for weak categories"
    python3 -m ftc.cli status <pipeline_dir>                       # one line per stage
    python3 -m ftc.cli list
    python3 -m ftc.cli wall -p <pipeline_dir> post EvalAgent "weak on stoichiometry" --to DataAgent --kind request
    python3 -m ftc.cli wall -p <pipeline_dir> inbox DataAgent      # see ftc/wall.py for all wall commands
"""
import json
import sys

from .pipeline import Pipeline, list_pipelines
from .spec import load_spec


def main(argv=None):
    import argparse
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "wall":            # the poster wall has its own sub-commands
        from .wall import main as wall_main
        return wall_main(argv[1:])
    ap = argparse.ArgumentParser(prog="ftc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("new"); s.add_argument("spec")
    s = sub.add_parser("link"); s.add_argument("dir"); s.add_argument("key"); s.add_argument("value")
    s.add_argument("--append", action="store_true"); s.add_argument("--stage")
    s = sub.add_parser("emit"); s.add_argument("dir"); s.add_argument("stage"); s.add_argument("type")
    s.add_argument("msg", nargs="?", default=""); s.add_argument("--loop", type=int)
    s = sub.add_parser("loop"); s.add_argument("dir"); s.add_argument("n", type=int); s.add_argument("reason", nargs="?", default="")
    s = sub.add_parser("status"); s.add_argument("dir")
    sub.add_parser("list")
    a = ap.parse_args(argv)

    if a.cmd == "new":
        pl = Pipeline.create(load_spec(a.spec))
        pl.set_loop(1)
        print(json.dumps({"pipeline_dir": str(pl.dir), "pipeline_id": pl.id, "task_spec_path": str(pl.dir / "task_spec.json")}))
        return 0
    if a.cmd == "list":
        for m in list_pipelines():
            print(f"{m['pipeline_id']:50s} {m.get('title') or ''}")
        return 0
    if a.cmd == "status":
        from .state import main as state_main
        return state_main([a.dir, "--brief"])
    pl = Pipeline.from_any(a.dir)
    if not pl.active:
        print(f"not a pipeline folder: {a.dir}", file=sys.stderr)
        return 1
    if a.cmd == "link":
        stage = a.stage or {"data_run_dir": "data", "train_job_id": "train", "bench_dir": "bench"}.get(a.key)
        pl.link(a.key, a.value, stage=stage, append=a.append or a.key in ("data_run_dir", "train_job_id"))
    elif a.cmd == "emit":
        pl.emit(a.stage, a.type, a.msg, loop=a.loop)
    elif a.cmd == "loop":
        pl.set_loop(a.n, a.reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())

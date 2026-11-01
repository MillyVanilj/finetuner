"""Pipeline folder + event log: the shared "what is happening now" record.

One pipeline = one user goal, possibly several loops. Its folder holds the task spec,
an append-only event log and small per-stage files. Existing run folders (DataGen's
~/.scienceclaw/finetune/runs/<run_id>/, the Trainer's ~/finetuner/runs/<job_id>/) are NOT
moved: the pipeline only *links* to them, and the dashboard reads their live files.

    $FINETUNER_DIR/pipelines/<pipeline_id>/
        task_spec.json        written by M1 (v2, see ftc/spec.py)
        pipeline.json         id, created_at, title, links (data_run_dir, bench_dir, train jobs...)
        events.jsonl          append-only log, one event per line (schema below)
        bench/                BenchGen output: dev.jsonl, heldout.jsonl, progress.json, bench_report.json
        loops/<n>/            one folder per loop: eval/progress.json, eval/predictions_*.jsonl,
                              eval/eval_report.json, decision.json
        final/                final_report.json, model_card.md

Event schema (one JSON object per line in events.jsonl):
    {"ts": "2026-11-01T09:30:00Z",   # UTC, ISO 8601
     "stage": "spec"|"data"|"bench"|"train"|"eval"|"decide"|"output"|"pipeline",
     "type": "start"|"progress"|"done"|"failed"|"link"|"metric"|"log"|"decision",
     "loop": 1,                      # 1-based loop number (0 for spec/bench, which run once)
     "msg": "human readable one-liner",
     ...any extra keys: e.g. link events carry "key" and "value"}

Usage from any skill or script (stdlib only, never raises: logging must not break a stage):
    from ftc.pipeline import Pipeline
    pl = Pipeline.from_any(args.pipeline_dir or args.spec)   # None if no pipeline -> calls are no-ops
    pl.emit("data", "start", "Generating 500 examples")
    pl.link("data_run_dir", "/home/ubuntu/.scienceclaw/finetune/runs/chem-v1")
    pl.progress("bench", done=40, total=300, unit="items", metrics={"rejected": 3})
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .spec import slugify

STAGES = ["spec", "data", "bench", "train", "eval", "decide", "output"]
EVENT_TYPES = ["start", "progress", "done", "failed", "link", "metric", "log", "decision"]


def finetuner_dir() -> Path:
    return Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")).expanduser()


def pipelines_root() -> Path:
    return Path(os.environ.get("FTC_PIPELINES", finetuner_dir() / "pipelines")).expanduser()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def _atomic_write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)   # readers never see a half-written file


class Pipeline:
    def __init__(self, path):
        self.dir = Path(path).expanduser().resolve() if path else None

    # ---------- finding / creating ----------
    @classmethod
    def create(cls, spec: dict, root=None) -> "Pipeline":
        """New pipeline folder from a (v2) spec. Folder name: <spec id>-<YYYYmmdd-HHMMSS>."""
        root = Path(root or pipelines_root())
        pid = f"{slugify(spec.get('id') or spec.get('title') or 'task', 30)}-{time.strftime('%Y%m%d-%H%M%S')}"
        d = root / pid
        d.mkdir(parents=True, exist_ok=False)
        _atomic_write(d / "task_spec.json", spec)
        _atomic_write(d / "pipeline.json", {"pipeline_id": pid, "title": spec.get("title"),
                                             "created_at": now_iso(), "loop": 1, "links": {}})
        pl = cls(d)
        pl.emit("spec", "done", f"Task spec saved: {spec.get('title')}", loop=0)
        return pl

    @classmethod
    def from_any(cls, path=None) -> "Pipeline":
        """Accepts a pipeline dir, a path inside it (e.g. its task_spec.json), or None.
        Falls back to $FTC_PIPELINE_DIR. Returns a no-op Pipeline if nothing matches."""
        cands = [path, os.environ.get("FTC_PIPELINE_DIR")]
        for c in cands:
            if not c:
                continue
            p = Path(c).expanduser()
            # 5 levels: reaches the pipeline from loops/<n>/eval/eval_report.json
            for d in [p, *p.parents][:5]:
                if (d / "pipeline.json").exists():
                    return cls(d)
        return cls(None)

    @property
    def active(self) -> bool:
        return self.dir is not None

    @property
    def id(self):
        return self.dir.name if self.dir else None

    # ---------- reading ----------
    def meta(self) -> dict:
        try:
            return json.loads((self.dir / "pipeline.json").read_text())
        except Exception:
            return {"pipeline_id": self.id, "links": {}}

    def spec(self) -> dict:
        try:
            return json.loads((self.dir / "task_spec.json").read_text())
        except Exception:
            return {}

    def current_loop(self) -> int:
        return int(self.meta().get("loop") or 1)

    def loop_dir(self, loop=None) -> Path:
        return self.dir / "loops" / str(loop or self.current_loop())

    # ---------- writing (never raise) ----------
    def emit(self, stage: str, type_: str, msg: str = "", loop=None, **extra):
        if not self.active:
            return
        try:
            ev = {"ts": now_iso(), "stage": stage, "type": type_,
                  "loop": self.current_loop() if loop is None else loop, "msg": msg, **extra}
            with open(self.dir / "events.jsonl", "a") as f:   # one write() per line: safe for concurrent appends
                f.write(json.dumps(ev) + "\n")
        except Exception:
            pass

    def link(self, key: str, value, stage=None, append=False):
        """Record where a stage's files live, e.g. link("train_job_id", jid, append=True)."""
        if not self.active:
            return
        try:
            m = self.meta()
            links = m.setdefault("links", {})
            if append:
                lst = links.setdefault(key, [])
                if value not in lst:
                    lst.append(value)
            else:
                links[key] = value
            _atomic_write(self.dir / "pipeline.json", m)
            self.emit(stage or "pipeline", "link", f"{key} = {value}", key=key, value=value)
        except Exception:
            pass

    def set_loop(self, loop: int, reason: str = ""):
        if not self.active:
            return
        m = self.meta()
        m["loop"] = int(loop)
        _atomic_write(self.dir / "pipeline.json", m)
        self.emit("pipeline", "log", f"Starting loop {loop}" + (f": {reason}" if reason else ""), loop=loop)

    def progress(self, stage: str, done=None, total=None, unit="", state="running", loop=None,
                 metrics=None, **extra):
        """Overwrite <stage>'s progress.json (cheap to call often). Stages that already write
        their own live file (DataGen state.json, Trainer status.json) don't need this."""
        if not self.active:
            return
        where = {"bench": self.dir / "bench",
                 "eval": self.loop_dir(loop) / "eval"}.get(stage, self.dir / stage)
        try:
            _atomic_write(where / "progress.json", {
                "stage": stage, "state": state, "done": done, "total": total, "unit": unit,
                "metrics": metrics or {}, "updated_at": now_iso(), **extra})
        except Exception:
            pass

    def write_json(self, relpath: str, data: dict) -> Path:
        p = self.dir / relpath
        _atomic_write(p, data)
        return p


def list_pipelines(root=None) -> list[dict]:
    root = Path(root or pipelines_root())
    out = []
    if not root.exists():
        return out
    for d in root.iterdir():
        if (d / "pipeline.json").exists():
            try:
                m = json.loads((d / "pipeline.json").read_text())
            except Exception:
                continue
            ev = d / "events.jsonl"
            m["updated"] = ev.stat().st_mtime if ev.exists() else d.stat().st_mtime
            m["dir"] = str(d)
            out.append(m)
    return sorted(out, key=lambda m: m["updated"], reverse=True)

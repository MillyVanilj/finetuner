#!/usr/bin/env python3
"""Live dashboard for the fine-tuning collective. Stdlib only: no pip install needed.

    python3 dashboard/server.py                 # http://localhost:8765
    python3 dashboard/server.py --port 9000 --host 0.0.0.0

On Brev, keep the default host (localhost) and forward the port to your laptop:
    brev port-forward <instance> --port 8765:8765      (or: ssh -L 8765:localhost:8765 <instance>)

API (all JSON):
    GET  /api/pipelines                         list of pipelines, newest first
    GET  /api/examples                          example specs from examples/specs/
    GET  /api/state?p=<pipeline_id>             full snapshot (ftc/state.py)
    GET  /api/stream?p=<pipeline_id>            Server-Sent Events: a snapshot whenever something changes
    GET  /api/file?p=<id>&path=<relpath>&n=20   first n lines of a file inside the pipeline folder
    POST /api/spec/validate   {spec}            -> {spec (v2, defaults filled), errors, warnings}
    POST /api/spec/draft      {goal, spec?, answers?}  -> M1: {spec, questions, errors, warnings, source}
    POST /api/pipelines       {spec}            -> create the pipeline folder, returns {pipeline_id, dir, commands}
    POST /api/simulate        {spec?, speed?}   -> start scripts/simulate_pipeline.py (no GPU, demo only)
    POST /api/pipelines/<id>/start {skip_general?} -> run scripts/run_pipeline.py (the REAL pipeline) in the background
    POST /api/pipelines/<id>/stop                  -> stop that runner
    GET  /api/wall?p=<id>                       the poster wall (p omitted = the lobby wall)
    POST /api/wall?p=<id> {author, text, to?, kind?, reply_to?}   pin a note (humans can talk to the agents)
    POST /api/wall/react?p=<id> {post, author, emoji}
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftc import m1                                          # noqa: E402
from ftc.pipeline import Pipeline, list_pipelines, pipelines_root   # noqa: E402
from ftc.spec import upgrade, validate_spec                 # noqa: E402
from ftc.state import snapshot                              # noqa: E402
from ftc.wall import Wall, summary as wall_summary          # noqa: E402

STATIC = Path(__file__).resolve().parent
POLL_S = 1.0
SIM_PROCS = []


def pipeline_dir(pid: str) -> Path:
    if not pid or "/" in pid or pid.startswith("."):
        raise ValueError("bad pipeline id")
    d = pipelines_root() / pid
    if not (d / "pipeline.json").exists():
        raise FileNotFoundError(pid)
    return d


def fingerprint(snap: dict) -> str:
    s = {k: v for k, v in snap.items() if k not in ("generated_at", "elapsed_s")}
    return hashlib.md5(json.dumps(s, sort_keys=True, default=str).encode()).hexdigest()


def next_commands(d: Path, spec: dict) -> list[dict]:
    """What a human (or run_pipeline.py) runs next. Shown in the UI after 'Create pipeline'."""
    sp = d / "task_spec.json"
    return [
        {"label": "Generate training data (DataGen, call until done=true)",
         "cmd": f"python3 ~/scienceclaw/skills/finetune-data-gen/scripts/gen_data.py --spec {sp} --run-id {d.name} --format json"},
        {"label": "Link the data run to this pipeline (so the dashboard shows it)",
         "cmd": f"python3 -m ftc.cli link {d} data_run_dir ~/.scienceclaw/finetune/runs/{d.name}"},
        {"label": "Or watch a simulated run (no GPU)",
         "cmd": f"python3 scripts/simulate_pipeline.py --spec {sp}"},
    ]


def start_runner(d: Path, body: dict) -> dict:
    """▶ Start: run scripts/run_pipeline.py for this pipeline in the background (survives the dashboard).
    GPU use is gated by finetune-submit's automatic pre-flight checks, not by this button."""
    from ftc.state import read_runner
    r = read_runner(d)
    if r.get("alive"):
        return {"ok": False, "error": f"already running (pid {r.get('pid')})"}
    args = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--pipeline", str(d)]
    if body.get("skip_general"):
        args.append("--skip-general")
    logf = open(d / "runner.log", "a")
    logf.write(f"\n===== started from the dashboard {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    logf.flush()
    proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            cwd=str(ROOT), start_new_session=True)
    (d / "runner.json").write_text(json.dumps({"state": "running", "pid": proc.pid, "step": "starting",
                                               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    return {"ok": True, "pid": proc.pid}


def stop_runner(d: Path) -> dict:
    """■ Stop: SIGTERM the runner (it finishes its current call, then stops). Jobs it already launched
    (training, eval, BenchGen background jobs) keep running; stop them on the GPU box if needed."""
    import os
    import signal
    from ftc.state import read_runner
    r = read_runner(d)
    if not r.get("alive"):
        return {"ok": False, "error": "not running"}
    try:
        os.kill(int(r["pid"]), signal.SIGTERM)
    except OSError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "pid": r["pid"]}


class Handler(BaseHTTPRequestHandler):
    server_version = "ftc-dashboard/0.1"

    def log_message(self, fmt, *args):   # quieter console
        if "/api/stream" not in (args[0] if args else ""):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---------- helpers ----------
    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _static(self, name, ctype):
        p = STATIC / name
        if not p.exists():
            return self._json({"error": "not found"}, 404)
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------- GET ----------
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._static("index.html", "text/html; charset=utf-8")
            if u.path == "/api/pipelines":
                return self._json({"root": str(pipelines_root()), "pipelines": list_pipelines()})
            if u.path == "/api/examples":
                ex = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "examples" / "specs").glob("*.json"))}
                return self._json({"examples": ex})
            if u.path == "/api/state":
                return self._json(snapshot(pipeline_dir(q.get("p"))))
            if u.path == "/api/stream":
                return self._stream(pipeline_dir(q.get("p")))
            if u.path == "/api/wall":
                return self._json(wall_summary(self._wall(q)))
            if u.path == "/api/file":
                return self._file(pipeline_dir(q.get("p")), q.get("path", ""), int(q.get("n", 20)))
            return self._json({"error": "not found"}, 404)
        except FileNotFoundError as e:
            return self._json({"error": f"no such pipeline: {e}"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            return self._json({"error": repr(e)}, 500)

    def _wall(self, q: dict) -> Wall:
        return Wall(pipeline_dir(q["p"]) / "wall.jsonl") if q.get("p") else Wall.lobby()

    def _stream(self, d: Path):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last, last_send = None, 0.0
        try:
            while True:
                snap = snapshot(d)
                fp = fingerprint(snap)
                now = time.time()
                if fp != last:
                    self.wfile.write(b"event: state\ndata: " + json.dumps(snap, default=str).encode() + b"\n\n")
                    self.wfile.flush()
                    last, last_send = fp, now
                elif now - last_send > 15:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_send = now
                time.sleep(POLL_S)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _file(self, d: Path, rel: str, n: int):
        p = (d / rel).resolve()
        if d.resolve() not in p.parents and p != d.resolve():
            return self._json({"error": "path outside pipeline"}, 400)
        if not p.is_file():
            return self._json({"error": "not a file"}, 404)
        lines = []
        with open(p, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max(1, min(n, 500)):
                    break
                lines.append(line.rstrip("\n"))
        return self._json({"path": rel, "lines": lines})

    # ---------- POST ----------
    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            body = self._body()
            if u.path == "/api/wall":
                if not (body.get("text") or "").strip():
                    return self._json({"error": "text is required"}, 400)
                pid = self._wall(q).post((body.get("author") or "Human").strip()[:40], body["text"],
                                         to=body.get("to"), kind=body.get("kind") or "note",
                                         reply_to=body.get("reply_to") or None)
                return self._json({"ok": bool(pid), "id": pid})
            if u.path == "/api/wall/react":
                pid = self._wall(q).react(body.get("post", ""), (body.get("author") or "Human")[:40], body.get("emoji") or "👍")
                return self._json({"ok": bool(pid), "id": pid})
            if u.path == "/api/spec/validate":
                spec = upgrade(body.get("spec") or {})
                errors, warnings = validate_spec(spec)
                return self._json({"spec": spec, "errors": errors, "warnings": warnings})
            if u.path == "/api/spec/draft":
                if not (body.get("goal") or "").strip():
                    return self._json({"error": "goal is required"}, 400)
                return self._json(m1.draft(body["goal"], body.get("spec"), body.get("answers")))
            if u.path == "/api/pipelines":
                spec = upgrade(body.get("spec") or {})
                errors, warnings = validate_spec(spec)
                if errors:
                    return self._json({"errors": errors, "warnings": warnings}, 400)
                pl = Pipeline.create(spec)
                return self._json({"pipeline_id": pl.id, "dir": str(pl.dir), "warnings": warnings,
                                   "commands": next_commands(pl.dir, spec)})
            if u.path.startswith("/api/pipelines/") and u.path.endswith(("/start", "/stop")):
                pid_, action = u.path.split("/")[3], u.path.rsplit("/", 1)[1]
                d = pipeline_dir(pid_)
                return self._json(start_runner(d, body) if action == "start" else stop_runner(d))
            if u.path == "/api/simulate":
                args = [sys.executable, str(ROOT / "scripts" / "simulate_pipeline.py"),
                        "--speed", str(float(body.get("speed") or 1))]
                if body.get("spec"):
                    tmp = pipelines_root() / ".sim_specs"
                    tmp.mkdir(parents=True, exist_ok=True)
                    sp = tmp / f"spec-{int(time.time())}.json"
                    sp.write_text(json.dumps(body["spec"]))
                    args += ["--spec", str(sp)]
                before = {p["pipeline_id"] for p in list_pipelines()}
                proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        start_new_session=True)
                SIM_PROCS.append(proc)
                for _ in range(50):   # wait for the folder to appear so the UI can switch to it
                    time.sleep(0.1)
                    new = [p for p in list_pipelines() if p["pipeline_id"] not in before]
                    if new:
                        return self._json({"pipeline_id": new[0]["pipeline_id"], "pid": proc.pid})
                return self._json({"error": "simulator did not start", "returncode": proc.poll()}, 500)
            return self._json({"error": "not found"}, 404)
        except FileNotFoundError as e:
            return self._json({"error": f"no such pipeline: {e}"}, 404)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            return self._json({"error": repr(e)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    pipelines_root().mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Dashboard: http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}  "
          f"(pipelines in {pipelines_root()})", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

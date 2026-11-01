#!/usr/bin/env python3
"""Freeze the dashboard into plain files that any static host can serve (GitHub Pages, Netlify, S3…).

The copy is read-only and shows each pipeline as it was when exported (no live stream). Use it to keep the
demo online after the cloud machines are gone. How to host it: docs/hosting.md.

    python3 scripts/export_static.py --out site/                      # every pipeline
    python3 scripts/export_static.py --out site/ --pipeline <id> ...  # only these
    python3 -m http.server -d site 8000                               # check it locally

Layout: site/index.html (the dashboard with window.FTC = {static: true}) + site/data/<key>.json, one file per
GET the page makes. The key is built the same way as staticKey() in dashboard/index.html.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))
import server  # noqa: E402  (dashboard/server.py: reuse exactly what the live dashboard serves)
from ftc.pipeline import list_pipelines  # noqa: E402
from ftc.state import snapshot  # noqa: E402
from ftc.wall import Wall, summary as wall_summary  # noqa: E402

# Belt and braces: logs end up in the snapshot, so blank anything that looks like an API key or token.
SECRET = re.compile(r"(sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}|"
                    r"gh[pousr]_[A-Za-z0-9]{30,}|(?:API_KEY|TOKEN|SECRET)=\S+)")


def key(name: str, p: str = "", path: str = "") -> str:
    """Must match staticKey() in dashboard/index.html."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", "__".join(x for x in (name, p, path) if x)) + ".json"


def write(out: Path, k: str, obj) -> None:
    text = SECRET.sub("[redacted]", json.dumps(obj, default=str))
    (out / "data" / k).write_text(text)


def export(out: Path, only: list[str] | None = None) -> list[str]:
    (out / "data").mkdir(parents=True, exist_ok=True)
    pls = [p for p in list_pipelines() if not only or p["pipeline_id"] in only]
    for p in pls:
        p.pop("dir", None)          # don't publish the machine's folder layout
    write(out, key("pipelines"), {"pipelines": pls})
    ex = {f.stem: json.loads(f.read_text()) for f in sorted((ROOT / "examples" / "specs").glob("*.json"))}
    write(out, key("examples"), {"examples": ex})
    write(out, key("wall"), wall_summary(Wall.lobby()))
    for p in pls:
        pid = p["pipeline_id"]
        d = server.pipeline_dir(pid)
        snap = snapshot(d)
        snap.pop("dir", None)
        write(out, key("state", pid), snap)
        write(out, key("wall", pid), wall_summary(Wall(d / "wall.jsonl")))
        write(out, key("answers", pid), server.read_answers(d))
        mc = d / "final" / "model_card.md"
        write(out, key("model_card", pid), {"text": mc.read_text() if mc.exists() else None})
        dev = d / "bench" / "dev.jsonl"   # the only file the page reads (Benchmark panel sample)
        lines = dev.read_text(errors="replace").splitlines()[:4] if dev.exists() else []
        write(out, key("file", pid, "bench/dev.jsonl"), {"path": "bench/dev.jsonl", "lines": lines})
    (out / "index.html").write_bytes(server.render_index({"static": True, "readOnly": True}))
    (out / ".nojekyll").write_text("")   # GitHub Pages: serve files as they are
    return [p["pipeline_id"] for p in pls]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="site", help="output folder (default: site/)")
    ap.add_argument("--pipeline", nargs="*", help="pipeline ids to export (default: all)")
    a = ap.parse_args(argv)
    ids = export(Path(a.out), a.pipeline)
    print(json.dumps({"out": str(Path(a.out).resolve()), "pipelines": ids}))
    if not ids:
        print("warning: no pipelines found (set FTC_PIPELINES or run on the machine that has them)", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Small file helpers."""
import json
from pathlib import Path


def read_jsonl(path: Path) -> list:
    path = Path(path)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def append_jsonl(path: Path, rows: list) -> None:
    with Path(path).open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

"""Hosting: the read-only dashboard refuses writes, and the static export has every file the page asks for."""
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dashboard"))


@pytest.fixture
def pipes(tmp_path, monkeypatch):
    root = tmp_path / "pipelines"
    monkeypatch.setenv("FTC_PIPELINES", str(root))
    from ftc.pipeline import Pipeline
    from ftc.spec import check_spec
    spec, errors, _ = check_spec(json.loads(next((ROOT / "examples" / "specs").glob("*.json")).read_text()))
    assert not errors
    pl = Pipeline.create(spec)
    pl.emit("data", "log", "key sk-ant-abcdefghijklmnopqrstuvwxyz0123 leaked in a log", loop=0)
    return pl


def test_static_keys_match_js():
    """staticKey() in index.html and key() in export_static.py must build the same file names."""
    import export_static
    if not shutil.which("node"):
        pytest.skip("node not installed")
    js = re.search(r"function staticKey\(u\) \{.*?\n\}", (ROOT / "dashboard" / "index.html").read_text(), re.S).group(0)
    urls = {"api/pipelines": ("pipelines",), "api/wall": ("wall",), "api/state?p=a-1": ("state", "a-1"),
            "api/file?p=a-1&path=bench/dev.jsonl&n=4": ("file", "a-1", "bench/dev.jsonl")}
    prog = js + f"\nconsole.log(JSON.stringify({json.dumps(list(urls))}.map(staticKey)))"
    got = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert got == [export_static.key(*v) for v in urls.values()]


def test_export(pipes, tmp_path):
    import export_static
    out = tmp_path / "site"
    assert export_static.export(out) == [pipes.id]
    html = (out / "index.html").read_text()
    assert '"static": true' in html
    names = {f.name for f in (out / "data").iterdir()}
    for k in ("pipelines.json", "examples.json", "wall.json", f"state__{pipes.id}.json",
              f"answers__{pipes.id}.json", f"model_card__{pipes.id}.json", f"file__{pipes.id}__bench_dev.jsonl.json"):
        assert k in names
    everything = "".join(f.read_text() for f in (out / "data").iterdir())
    assert "sk-ant-abcdefghij" not in everything and "[redacted]" in everything
    assert str(tmp_path) not in (out / "data" / "pipelines.json").read_text()


def test_read_only_server(pipes):
    proc = subprocess.Popen([sys.executable, str(ROOT / "dashboard" / "server.py"), "--read-only", "--port", "8797"],
                            stderr=subprocess.DEVNULL)
    try:
        base = "http://127.0.0.1:8797"
        for _ in range(50):
            try:
                html = urllib.request.urlopen(base + "/").read().decode()
                break
            except OSError:
                time.sleep(0.1)
        assert 'window.FTC = {"readOnly": true}' in html
        assert "root" not in json.loads(urllib.request.urlopen(base + "/api/pipelines").read())
        assert urllib.request.urlopen(f"{base}/api/state?p={pipes.id}").status == 200
        for path in ("/api/simulate", f"/api/pipelines/{pipes.id}/start", "/api/wall", "/api/spec/draft"):
            req = urllib.request.Request(base + path, data=b"{}", headers={"Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
            assert e.value.code == 403
    finally:
        proc.terminate()
        proc.wait()

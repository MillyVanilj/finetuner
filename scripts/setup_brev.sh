#!/usr/bin/env bash
# One-time setup of the GPU machine (Brev, Linux). Safe to re-run.
#
# Result:
#   ~/scienceclaw/            ScienceClaw (unedited) + its venv .venv/
#   ~/finetuner/              this repo (you cloned it to run this script)
#   ~/finetuner/.venv-train/  separate venv for GPU training (torch, trl, peft…)
#   ~/finetuner/.env          your API keys (created from .env.example, never committed)
#
# Usage (from inside the cloned repo):  bash scripts/setup_brev.sh
# This only installs software. It does NOT start any training job.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SC_DIR="${SCIENCECLAW_DIR:-$HOME/scienceclaw}"

need() { command -v "$1" >/dev/null || { echo "❌ '$1' not found. Install it first (e.g. sudo apt install -y $2)." >&2; exit 1; }; }
need git git
need python3 python3
python3 -c 'import sys; assert sys.version_info >= (3,10)' || { echo "❌ Python 3.10+ needed" >&2; exit 1; }
python3 -m venv --help >/dev/null 2>&1 || { echo "❌ python3-venv missing: sudo apt install -y python3-venv" >&2; exit 1; }

echo "== 1/5 ScienceClaw =="
if [ -d "$SC_DIR/.git" ]; then
  echo "already at $SC_DIR (not updating it; run 'git -C $SC_DIR pull' yourself if you want the latest)"
else
  git clone https://github.com/lamm-mit/scienceclaw.git "$SC_DIR"
fi
echo "ScienceClaw version: $(git -C "$SC_DIR" rev-parse --short HEAD)"

echo "== 2/5 ScienceClaw Python environment =="
[ -d "$SC_DIR/.venv" ] || python3 -m venv "$SC_DIR/.venv"
"$SC_DIR/.venv/bin/pip" install -q --upgrade pip
"$SC_DIR/.venv/bin/pip" install -q -r "$SC_DIR/requirements.txt"
(cd "$SC_DIR" && ./install_scienceclaw_command.sh >/dev/null)
echo "✓ ScienceClaw installed"

echo "== 3/5 Training environment (separate, so GPU packages don't clash with ScienceClaw) =="
[ -d "$REPO_DIR/.venv-train" ] || python3 -m venv "$REPO_DIR/.venv-train"
"$REPO_DIR/.venv-train/bin/pip" install -q --upgrade pip
"$REPO_DIR/.venv-train/bin/pip" install -q torch "transformers>=5" "trl>=1" peft datasets accelerate huggingface_hub
"$REPO_DIR/.venv-train/bin/python" - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("✓ training env ready; GPU:", torch.cuda.get_device_name(0) if ok else "NOT FOUND (training will not work on this machine)")
EOF

echo "== 4/5 API keys file =="
if [ -f "$REPO_DIR/.env" ]; then
  echo "✓ .env exists"
else
  cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
  echo "⚠️  Created .env: open it and fill in your keys (it is gitignored)"
fi

echo "== 5/5 Plug our skills + agents into ScienceClaw =="
SCIENCECLAW_DIR="$SC_DIR" bash "$REPO_DIR/scripts/install.sh"

cat <<EOF

All set. Every new terminal session:
  source $SC_DIR/.venv/bin/activate
  set -a; source $REPO_DIR/.env; set +a       # ScienceClaw reads keys from the environment
  export PATH="\$HOME/.local/bin:\$PATH"

After every 'git pull' in $REPO_DIR:
  bash $REPO_DIR/scripts/install.sh
EOF

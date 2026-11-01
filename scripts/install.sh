#!/usr/bin/env bash
# Plug our pieces into ScienceClaw. Safe to re-run: do it after every `git pull`.
#
#   skills/<name>/                 -> $SCIENCECLAW_DIR/skills/<name>/
#   agents/<Name>/agent_profile.json -> ~/.scienceclaw/profiles/<Name>/agent_profile.json
#
# ScienceClaw is NOT edited: we only add our own skill folders to it.
# Usage:  bash scripts/install.sh          (ScienceClaw assumed at ~/scienceclaw)
#         SCIENCECLAW_DIR=/path bash scripts/install.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SC_DIR="${SCIENCECLAW_DIR:-$HOME/scienceclaw}"
MARKER=".from-finetuner"   # marks skill folders we copied, so we never touch ScienceClaw's own

if [ ! -d "$SC_DIR/skills" ]; then
  echo "❌ ScienceClaw not found at $SC_DIR. Run scripts/setup_brev.sh first, or set SCIENCECLAW_DIR." >&2
  exit 1
fi

echo "Installing skills into $SC_DIR/skills"
for src in "$REPO_DIR"/skills/*/; do
  src="${src%/}"
  [ -f "$src/SKILL.md" ] || continue
  name="$(basename "$src")"
  dest="$SC_DIR/skills/$name"

  if [ -d "$dest" ] && [ ! -f "$dest/$MARKER" ]; then
    echo "  ⚠️  skip $name: ScienceClaw already has a skill with this name, rename ours"
    continue
  fi

  # ScienceClaw only runs the FIRST .py in scripts/ (order not guaranteed).
  n_py=$(find "$src/scripts" -maxdepth 1 -name '*.py' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n_py" != "1" ]; then
    echo "  ⚠️  $name has $n_py top-level .py files in scripts/; it needs exactly 1 (put helpers in a subfolder)"
  fi

  # Unchanged -> leave the folder alone. Deleting it would pull the working directory out from under
  # jobs running from it (a running eval crashed with FileNotFoundError when a pull+install loop re-copied it).
  if [ -d "$dest" ] && diff -rq -x "$MARKER" -x __pycache__ "$src" "$dest" >/dev/null 2>&1; then
    echo "  = $name (unchanged)"
    continue
  fi

  rm -rf "$dest"          # only ever our own previous copy (checked via marker above)
  cp -R "$src" "$dest"
  touch "$dest/$MARKER"
  echo "  ✓ $name"
done

echo "Installing agent profiles into ~/.scienceclaw/profiles"
found_profile=0
for prof in "$REPO_DIR"/agents/*/agent_profile.json; do
  [ -f "$prof" ] || continue
  found_profile=1
  name="$(basename "$(dirname "$prof")")"
  dest="$HOME/.scienceclaw/profiles/$name/agent_profile.json"
  mkdir -p "$(dirname "$dest")"
  # Copy only this file (agent memory/state is left alone), but keep the pipeline scope that
  # `ftc.cli agents kickoff` set: a running agent must not become unscoped (see ftc/agents.py).
  python3 - "$prof" "$dest" <<'PY'
import json, sys
src, dest = sys.argv[1], sys.argv[2]
new = json.load(open(src))
try:
    scope = json.load(open(dest)).get("investigation_id_filter")
except (OSError, ValueError):
    scope = None
if scope:
    new["investigation_id_filter"] = scope
json.dump(new, open(dest, "w"), indent=2)
PY
  echo "  ✓ $name"
done
[ "$found_profile" = 1 ] || echo "  (no profiles yet: add agents/<Name>/agent_profile.json)"

echo "Done."

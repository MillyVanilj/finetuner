#!/bin/bash
# Publish the dashboard to GitHub Pages: https://millyvanilj.github.io/finetuner/
# Run it on the machine that has the pipelines (the Brev box), any time you want the site updated:
#     bash scripts/deploy_pages.sh                      # every pipeline
#     bash scripts/deploy_pages.sh --pipeline <id> ...  # only these
# It exports a static copy (scripts/export_static.py) and commits it to the gh-pages branch (main is untouched).
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=$(git remote get-url origin)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

python3 scripts/export_static.py --out "$TMP/site" "$@"

if git ls-remote --exit-code --heads "$REMOTE" gh-pages >/dev/null 2>&1; then
  git clone -q --depth 1 -b gh-pages "$REMOTE" "$TMP/repo"
else
  git init -q -b gh-pages "$TMP/repo"
  git -C "$TMP/repo" remote add origin "$REMOTE"
fi
find "$TMP/repo" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$TMP/site/." "$TMP/repo/"
git -C "$TMP/repo" add -A
if git -C "$TMP/repo" diff --cached --quiet; then
  echo "Nothing changed since the last deploy."
  exit 0
fi
# a fresh box (Brev) may have no git name/email: fall back to a neutral one instead of failing
git -C "$TMP/repo" config user.name >/dev/null || git -C "$TMP/repo" config user.name "finetuner-deploy"
git -C "$TMP/repo" config user.email >/dev/null || git -C "$TMP/repo" config user.email "finetuner-deploy@users.noreply.github.com"
git -C "$TMP/repo" commit -q -m "Dashboard snapshot $(date -u +%Y-%m-%dT%H:%MZ)"
git -C "$TMP/repo" push -q origin gh-pages
echo "Published. Live in ~1 minute at https://millyvanilj.github.io/finetuner/"

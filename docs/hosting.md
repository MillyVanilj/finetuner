# Hosting the dashboard

## The website: GitHub Pages → https://millyvanilj.github.io/finetuner/

GitHub Pages hosts plain files only, so the site is a snapshot of the pipelines (no live updates, no buttons).
To update it, run this **on the Brev box** (where the pipelines are), e.g. after a run finishes:

```bash
cd ~/finetuner && git pull
bash scripts/deploy_pages.sh                       # every pipeline
bash scripts/deploy_pages.sh --pipeline <id> ...   # or only the good ones
```

It exports the pipelines (`scripts/export_static.py`) and commits them to the `gh-pages` branch; the site
updates about a minute later. `main` is never touched. Do it once more before Brev is stopped Sunday night:
the site stays up after the cloud accounts are gone. (It currently shows a simulated demo run.)

The rest of this page is for sharing the *live* dashboard instead.

The dashboard (`dashboard/server.py` + `dashboard/index.html`) has **no login**. Anyone who can reach it can
press ▶ Start (GPU time), draft specs and run simulations (API money), push to Hugging Face, and post on the wall.
So never put the normal server on a public URL. There are two safe ways to share it:

| | Live, read-only (during the event) | Static copy (after the event) |
|---|---|---|
| What people see | the real dashboard, updating live | a frozen snapshot of finished pipelines |
| Runs on | the Brev box (it must stay up) | any static host: GitHub Pages, Netlify; no server |
| Writes | refused (HTTP 403), buttons hidden | none possible |
| Cost | free (Cloudflare quick tunnel) | free |

The team keeps using the normal dashboard privately through port-forwarding, as before
(`brev port-forward <instance> --port 8765:8765`, then http://localhost:8765).

## Quickest: the full dashboard on a public link (no protection)

If you don't care who can press the buttons (e.g. a short demo), tunnel the normal dashboard directly:

```bash
cd ~/finetuner
nohup python3 dashboard/server.py --port 8765 > ~/dashboard.log 2>&1 &     # skip if it is already running
curl -L -o ~/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/cloudflared
nohup ~/cloudflared tunnel --url http://localhost:8765 > ~/tunnel.log 2>&1 &
sleep 5; grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' ~/tunnel.log
```

Everything works through that link (New task, ▶ Start, the wall…), for anyone who has it. Stop: `pkill cloudflared`.

---

## A. Live, read-only, through a Cloudflare tunnel (for judges during the event)

On the Brev box, next to the normal dashboard (both read the same `~/finetuner/pipelines/`, so the
public view follows the team's runs live):

```bash
cd ~/finetuner
# 1. a second dashboard, read-only, on its own port (only reachable from the box itself)
nohup python3 dashboard/server.py --read-only --port 8766 > ~/dashboard-public.log 2>&1 &

# 2. cloudflared, once (single binary, no sudo, no account)
curl -L -o ~/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/cloudflared

# 3. a public HTTPS URL for port 8766
nohup ~/cloudflared tunnel --url http://localhost:8766 > ~/tunnel.log 2>&1 &
sleep 5; grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' ~/tunnel.log
```

Share the printed `https://….trycloudflare.com` URL. Add `#p=<pipeline_id>` to open one pipeline directly.

- **Expose only port 8766** (the `--read-only` one), never 8765.
- The URL changes each time `cloudflared` restarts. For a fixed URL you need a free Cloudflare account and a
  named tunnel (not worth it for one weekend).
- Stop sharing: `pkill cloudflared` (and `pkill -f "server.py --read-only"`).
- If the Brev instance is stopped, the link stops working. Make a static copy (B) before stopping it.
- What read-only mode does: every POST is refused with 403, the page hides Simulate, New task, Start/Stop,
  Reset BenchGen, publish buttons and the wall's post box, and `/api/pipelines` no longer shows the
  machine's folder path. Reading files is unchanged (`/api/file` stays limited to the pipeline folder).

## B. Static copy (keeps the demo online after the cloud accounts are deleted)

On the machine that has the pipelines (the Brev box):

```bash
cd ~/finetuner
python3 scripts/export_static.py --out ~/site                            # every pipeline
python3 scripts/export_static.py --out ~/site --pipeline <id> <id2>      # or only the good ones
python3 -m http.server -d ~/site 8000                                    # check it (port-forward 8000)
```

It writes `index.html` plus one JSON file per request the page makes (`data/state__<id>.json`, …).
API keys and tokens that slipped into logs are replaced with `[redacted]`, but **open the copy and look
before publishing**: it contains the training examples, benchmark answers and the runner logs.

Then put the folder online, either way:

**Netlify Drop (easiest, no git):** copy `~/site` to your laptop
(`scp -r <instance>:~/site .`), open https://app.netlify.com/drop and drag the folder in. You get a URL
right away; sign in (free) to keep it beyond an hour.

**GitHub Pages:** the repo owner (`MillyVanilj`) does this once. Pages on a private repo needs a paid plan,
so use Netlify if the repo is private.

```bash
cd ~/site
git init -b gh-pages && git add . && git commit -m "Static dashboard snapshot"
git push https://github.com/MillyVanilj/finetuner.git gh-pages      # a separate branch; main is untouched
```

Then on GitHub: Settings → Pages → Source "Deploy from a branch" → `gh-pages` / root. The site appears at
`https://millyvanilj.github.io/finetuner/`. To update it later, re-export and push again (the second push
needs `--force` because the branch history is new each time; it only affects `gh-pages`).

## How it works (for whoever changes the dashboard)

- `index.html` calls the API with **relative** URLs (`api/state?...`, not `/api/state?...`) so it also works
  under a sub-path such as `/finetuner/`. Keep it that way when adding calls.
- The server injects `window.FTC = {readOnly: true}` into `index.html` when started with `--read-only`; the
  export injects `{static: true, readOnly: true}`. `readOnly` adds the `ro` class to `<html>`, and one CSS
  rule hides the write controls. **New buttons that write must be added to that CSS rule.**
- In static mode the page rewrites `GET api/<name>?p=<id>&path=<f>` to `data/<name>__<id>__<f>.json`
  (`staticKey()` in `index.html`, `key()` in `scripts/export_static.py`; `tests/test_hosting.py` checks they
  agree). **A new GET endpoint the page uses must also be written by `export_static.py`.**

# ollama-search

An improved static-site mirror of [ollama.com/search](https://ollama.com/search) with dark mode, better sorting, MLX/GGUF separation, and live search suggestions.

**Live site**: [maternion.github.io/ollama-search](https://maternion.github.io/ollama-search/) — auto-refreshes every 2 hours when ollama.com adds or updates models.

## Structure

```
scraper/scrape.py     Scrape ollama.com → scraper/*.json + tags/ + pages/ + tag_pages/ + blobs/
build.py              Build static site from scraped data → public/
serve.py              Dev server (serves public/ at localhost:8000)
.github/workflows/    CI: check-only → scrape → build → deploy every 2h
```

## Usage

### Local dev

```bash
# 1. Scrape (fetches catalog + per-model tags/pages/blobs)
pip install requests
python3 scraper/scrape.py --smart --skip-search

# 2. Build static site (BASE="" for local)
python3 build.py

# 3. Serve locally
python3 serve.py
# → http://localhost:8000/
```

### GitHub Pages build

```bash
# Build with base path for project pages
python3 build.py --base /ollama-search
```

## Features

- 245 models scraped from ollama.com
- Dark mode with proper Tailwind shade-inverted colors
- Sorting: Popular, Newest, Oldest, Recently updated, Pulls, Tags, Name
- Capability chips: Embedding, Vision, Tools, Thinking
- Cloud filter dropdown: All / Cloud only / Local only
- MLX/GGUF/All tabs on model detail and tags pages
- Copy-to-clipboard for pull commands
- Live search suggestions dropdown on model pages (navbar search)
- Per-model tag pages with blob detail pages
- User models show owner/name (e.g. `maternion/lfm2`)
- Auto-refreshes every 2 hours via GitHub Actions (only deploys on change)

## CI workflow

The GitHub Actions workflow (`.github/workflows/deploy.yml`) runs every 2 hours:

1. **Check-only**: Fetches catalog, hashes `name:tag_count:updated_title:path` for every model. If hash matches previous run → exit (no build, no deploy)
2. **Scrape**: If changed, runs `--smart` mode — skips unchanged models via per-tag-digest comparison. Writes checkpoint to `scraped-data` git branch every 5 models
3. **Build**: Runs `build.py --base /ollama-search` to generate static HTML
4. **Deploy**: Pushes to `gh-pages` branch (GitHub Pages serves it). Tags previous gh-pages HEAD as `deploy-pre-*` for rollback (keeps last 10)
5. **Self-check**: Validates data integrity (non-blocking)

The `--smart` scraper uses:
- Model-level `updated_title` + `tag_count` comparison (tier 1)
- Per-tag `manifest_digest` comparison (tier 2) — skips unchanged tags within changed models
- Blobs stored once per digest (`blobs/<digest>.json`) — deduplication across tags
- Readme stored once per model (in `pages/`), not duplicated per tag page

## Scraper data format

```
scraper/
  models.json              Catalog: 245 models with pulls, tags, sizes, capabilities
  sort_orders.json         Sort order data (popular, newest, etc.)
  sort_ranks.json          Per-model rank for each sort order
  tags/                    Per-model tag listings (one JSON per model)
  pages/                   Per-model page data with readme_html (one per model)
  tag_pages/               Per-tag detail pages (files, manifest_digest, applications)
  blobs/                   Per-digest blob data (deduped — 6,809 unique digests)
  .scrape-manifest.json    Run-level timestamp (single file, not per-record)
```
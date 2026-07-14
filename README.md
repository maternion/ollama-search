# ollama-search

A static-site mirror of [ollama.com/search](https://ollama.com/search) with dark mode, better sorting, and MLX/GGUF separation.

**Live site**: deployed to GitHub Pages, auto-refreshes every 2 hours when ollama.com adds or updates models.

## Structure

```
scraper/scrape.py     Scrape ollama.com → models.json + tags/*.json
build.py              Build static site from scraped data → public/
serve.py              Dev server (redirects / → /search/)
.github/workflows/    CI: scrape → diff → build → deploy every 2h
```

## Usage

### Local dev

```bash
# 1. Scrape (fetches /library + per-model /tags pages)
pip install requests
python3 scraper/scrape.py --skip-search

# 2. Build static site (BASE="" for local)
python3 build.py

# 3. Serve locally
python3 serve.py
# → http://localhost:8000/search/
```

### GitHub Pages build

```bash
# Build with base path for project pages
python3 build.py --base /ollama-search
```

## Features

- 236 official models scraped from ollama.com
- Dark mode with proper Tailwind shade-inverted colors
- Sorting: Popular, Newest, Oldest, Recently updated, Pulls, Tags, Name
- Capability chips: Embedding, Vision, Tools, Thinking
- Cloud filter dropdown: All / Cloud only / Local only
- MLX/GGUF/All tabs on model detail and tags pages
- Copy-to-clipboard for pull commands
- Auto-refreshes every 2 hours via GitHub Actions (only deploys on change)

## CI workflow

The GitHub Actions workflow (`.github/workflows/deploy.yml`):
1. Scrapes ollama.com `/library` catalog
2. Hashes model names + pulls + updated timestamps
3. Compares hash against cached value
4. If unchanged → exits (no build, no deploy)
5. If changed → full scrape, build with `--base`, deploy to GitHub Pages
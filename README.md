# ollama-search

A static-site mirror of [ollama.com/search](https://ollama.com/search) with dark mode, better sorting, and MLX/GGUF separation.

## Structure

```
scraper/scrape.py     Scrape ollama.com → models.json + tags/*.json
build.py              Build static site from scraped data → public/
serve.py              Dev server (redirects / → /search/)
```

## Usage

```bash
# 1. Scrape (fetches /library + per-model /tags pages)
python3 scraper/scrape.py

# 2. Build static site
python3 build.py

# 3. Serve locally
python3 serve.py
# → http://localhost:8000/
```

## Features

- 236 official models scraped from ollama.com
- Dark mode with proper Tailwind shade-inverted colors
- Sorting: Popular, Newest, Oldest, Recently updated, Pulls, Tags, Name
- Capability chips: Embedding, Vision, Tools, Thinking
- Cloud filter dropdown: All / Cloud only / Local only
- MLX/GGUF/All tabs on model detail and tags pages
- Copy-to-clipboard for pull commands
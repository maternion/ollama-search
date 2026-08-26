# ollama-search

An improved static-site mirror of [ollama.com/search](https://ollama.com/search) with dark mode, KV cache memory graphs, community model support, and live search suggestions.

**Live site**: [maternion.github.io/ollama-search](https://maternion.github.io/ollama-search/) — auto-refreshes every 2 hours when ollama.com adds or updates models.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/fd2be5a6-fcdc-415d-ac9b-669725b320e2" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/025f48fa-b48b-401a-bd56-2eebab915296" />

## Structure

```
scraper/scrape.py     Scrape ollama.com → scraper/*.json + tags/ + pages/ + tag_pages/ + blobs/
build.py              Build static site from scraped data → public/
serve.py              Dev server (serves public/ at localhost:8000)
memcalc/              KV cache memory calculator (per-architecture: standard, MLA, SWA, hybrid)
.github/workflows/    CI: scheduled scrape, full scrape, targeted model scrape, rebuild
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

### Targeted model scrape

```bash
# Re-scrape a single model (force re-fetch tags, pages, blobs)
python3 scraper/scrape.py --only-model /frob/kimi-k3 -v
```

## Features

- 390+ models scraped from ollama.com (official + community: frob, maternion, huihui_ai, /x experimental)
- Dark mode with proper Tailwind shade-inverted colors
- Sorting: Popular, Newest, Oldest, Recently updated, Pulls, Tags, Name
- Capability chips: Embedding, Vision, Tools, Thinking, Audio
- Filters: Size (piecewise slider), Cloud, Audio, MLX, MTP, Architecture, Template (Go Template), MoE
- URL-driven filter state (filters persist in URL)
- NEW badge for newly scraped models
- KV cache memory graphs showing VRAM vs context length per model
- Total memory mode (weights + KV), log-Y axis, context slider
- Hover any graph line or dot to highlight it and dim the rest
- Template classification (renderer/jinja/base) from capability badges
- Capability inference from badges, readme headings, jinja templates, and descriptions
- Replicated ollama.com/pricing page with tabs, tiers, badges, and cloud metrics
- HuggingFace config.json fallback for broken blob pages (kimi-k3 etc)
- MLX/GGUF/All tabs on model detail and tags pages
- Copy-to-clipboard for pull commands
- Live search suggestions dropdown on model pages (navbar search)
- Per-model tag pages with blob detail pages
- User models show owner/name (e.g. `frob/kimi-k3`)
- Mobile-optimized detail graphs
- Auto-refreshes every 2 hours via GitHub Actions

## memcalc (KV cache memory calculator)

Computes KV cache memory at arbitrary context lengths for different architectures:

- **Standard** (MHA/GQA): `n_layers × context × n_kv_heads × (head_dim + v_head_dim) × bytes`
- **MLA** (Multi-head Latent Attention): compressed KV via `kv_lora_rank` (DeepSeek-style)
- **SWA** (Sliding Window Attention): Gemma, Llama4, Muse-Glimmer, etc.
- **Hybrid**: interleaved attention + recurrent layers (Kimi K3, Qwen3.5, Jamba, etc.) — recurrent state computed separately, excluded from per-context KV curve
- MTP/NextN prediction layers subtracted from block count
- MoE expert count detection for weight memory

## CI workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `deploy.yml` | Every 2h + manual | Smart scrape with 6-month age filter → build → deploy |
| `full-scrape.yml` | Manual only | Full scrape of all models (no age filter) → build → deploy |
| `scrape-model.yml` | Manual only | Targeted re-scrape of specific model(s) by path → build → deploy |
| `rebuild.yml` | Push to build.py/memcalc | Build-only (no scrape) → deploy |

### Smart scraping

- Model-level `updated_title` + `tag_count` comparison (tier 1)
- Per-tag `manifest_digest` comparison (tier 2) — skips unchanged tags within changed models
- Blobs stored once per digest (`blobs/<digest>.json`) — deduplication across tags
- 6-month age filter: skips re-fetching stale official models, cached data preserved for graphs
- HF fallback: when ollama.com blob pages return 500, fetches `config.json` from the base model's HuggingFace repo
- Failed digest dedupe: if one tag's blob returns 500, other tags with the same digest skip retries

### Scraped data format

```
scraper/
  models.json              Catalog: 390+ models with pulls, tags, sizes, capabilities
  sort_orders.json         Sort order data (popular, newest, etc.)
  sort_ranks.json          Per-model rank for each sort order
  tags/                    Per-model tag listings (one JSON per model)
  pages/                   Per-model page data with readme_html (one per model)
  tag_pages/               Per-tag detail pages (files, manifest_digest, applications)
  blobs/                   Per-digest blob data (deduped — 7,900+ unique digests)
  profile_*.json           Profile page data (frob, maternion, huihui_ai, x)
```

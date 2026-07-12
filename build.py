#!/usr/bin/env python3
"""Build a static site from scraped ollama.com data.

Generates:
  public/index.html                       library/search page (mirror)
  public/library/<slug>/index.html        model detail page
  public/library/<slug>/tags/index.html    tags page with GGUF/MLX tabs
  public/assets/models.json                embedded catalog for client-side filter/sort

The markup mirrors ollama.com's /library page exactly (same Tailwind classes),
with these improvements layered on top:
  - dark mode (toggle + persisted localStorage + prefers-color-scheme)
  - extra sort options (popular, newest, updated, name, pulls, tags)
  - "hide cloud models" checkbox
  - GGUF/MLX tabs on model detail + tags pages
  - copy-to-clipboard for tag names + CLI snippets
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRAPER = HERE / "scraper"
PUBLIC = HERE / "public"
TAGS_DIR = SCRAPER / "tags"

# Base URL prefix — "" for local dev, "/ollama-search" for GitHub Pages project site.
# Set via: python3 build.py --base /ollama-search
BASE = ""


def url(path: str) -> str:
    """Prefix a site-internal path with BASE. Ensures leading /."""
    if path.startswith("http"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return BASE + path


# --------------------------------------------------------------------------- #
# SVG icons (verbatim from ollama.com markup)
# --------------------------------------------------------------------------- #

SVG_DOWNLOAD = (
    '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
    'stroke-width="1.5" stroke="currentColor" '
    'class="mr-1.5 h-[14px] w-[14px] sm:h-4 sm:w-4">'
    '<path stroke-linecap="round" stroke-linejoin="round" '
    'd="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5'
    'M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"></path></svg>'
)

SVG_TAG = (
    '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
    'stroke-width="1.5" stroke="currentColor" '
    'class="mr-1.5 h-[14px] w-[14px] sm:h-4 sm:w-4">'
    '<path stroke-linecap="round" stroke-linejoin="round" '
    'd="M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591'
    "l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 005.223-5.223"
    'c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568 3z" />'
    '<path stroke-linecap="round" stroke-linejoin="round" '
    'd="M6 6h.008v.008H6V6z" /></svg>'
)

SVG_CLOCK = (
    '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
    'stroke-width="1.5" stroke="currentColor" '
    'class="mr-1.5 h-[14px] w-[14px] sm:h-4 sm:w-4">'
    '<path stroke-linecap="round" stroke-linejoin="round" '
    'd="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"></path></svg>'
)

SVG_COPY = (
    '<svg class="copy-icon h-[18px] w-[18px]" xmlns="http://www.w3.org/2000/svg" '
    'fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">'
    '<path stroke-linecap="round" stroke-linejoin="round" '
    'd="M16.5 8.25V6a2.25 2.25 0 00-2.25-2.25H6A2.25 2.25 0 003.75 6v8.25'
    "A2.25 2.25 0 006 16.5h2.25m8.25-8.25H18a2.25 2.25 0 012.25 2.25V18"
    "A2.25 2.25 0 0118 20.25h-7.5A2.25 2.25 0 018.25 18v-1.5"
    'm8.25-8.25h-6a2.25 2.25 0 00-2.25 2.25v6"></path></svg>'
    '<svg class="check-icon hidden h-[18px] w-[18px]" xmlns="http://www.w3.org/2000/svg" '
    'fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">'
    '<path stroke-linecap="round" stroke-linejoin="round" '
    'd="M4.5 12.75l6 6 9-13.5" /></svg>'
)

SVG_SEARCH = (
    '<svg class="mt-0.25 ml-1.5 h-5 w-5 fill-current" viewBox="0 0 20 20" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<path d="m8.5 3c3.0375661 0 5.5 2.46243388 5.5 5.5 0 1.24832096-'
    ".4158777 2.3995085-1.1166416 3.3225711l4.1469717 4.1470988"
    "c.2928932.2928932.2928932.767767 0 1.0606602-.2662666.2662665-"
    ".6829303.2904726-.9765418.0726181l-.0841184-.0726181-4.1470988-"
    "4.1469717c-.9230626.7007639-2.07425014 1.1166416-3.3225711 1.1166416-"
    "3.03756612 0-5.5-2.4624339-5.5-5.5 0-3.03756612 2.46243388-5.5 5.5-5.5"
    "zm0 1.5c-2.209139 0-4 1.790861-4 4s1.790861 4 4 4 4-1.790861 4-4-"
    '1.790861-4-4-4z" /></svg>'
)

SVG_MOON = (
    '<svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" '
    'd="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 '
    "0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 "
    '12.75 21a9.753 9.753 0 009.002-5.998z" /></svg>'
)

SVG_SUN = (
    '<svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" '
    'd="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25M18.364 17.614l-'
    "1.591-1.591M12 18.75V21M5.636 17.614l1.591-1.591M3 12h2.25M5.636 6.386"
    'l1.591 1.591M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" /></svg>'
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def slugify(path: str) -> str:
    return path.strip("/").replace("/", "__")


def slug_url(path: str) -> str:
    """Convert /library/foo to /library/foo (already a clean URL)."""
    return path


def format_count(n: int) -> str:
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}M" if v < 10 else f"{int(v)}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}K" if v < 10 else f"{int(v)}K"
    return str(n)


def load_models() -> list[dict]:
    data = json.loads((SCRAPER / "models.json").read_text())
    return data["models"]


def load_ranks() -> dict:
    rf = SCRAPER / "sort_ranks.json"
    if rf.exists():
        return json.loads(rf.read_text())
    return {}


def load_tags(model_path: str) -> list[dict]:
    tf = TAGS_DIR / f"{slugify(model_path)}.json"
    if not tf.exists():
        return []
    return json.loads(tf.read_text()).get("tags", [])


def has_mlx(tags: list[dict]) -> bool:
    return any(t["format"] == "mlx" for t in tags)


def has_cloud_tag(tags: list[dict]) -> bool:
    return any(t["name"] == "cloud" for t in tags)


# --------------------------------------------------------------------------- #
# Shared HTML fragments
# --------------------------------------------------------------------------- #


def head_html(title: str, description: str, extra_css: bool = False) -> str:
    desc = esc(description)
    return f"""    <title>{esc(title)}</title>
    <meta charset="utf-8" />
    <meta name="description" content="{desc}"/>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta property="og:title" content="{esc(title)}" />
    <meta property="og:description" content="{desc}" />
    <meta property="og:type" content="website" />
    <meta name="robots" content="index, follow" />
    <link rel="icon" type="image/png" sizes="16x16" href="{url("/assets/icon-16x16.png")}" />
    <link rel="icon" type="image/png" sizes="32x32" href="{url("/assets/icon-32x32.png")}" />
    <link rel="icon" type="image/png" sizes="48x48" href="{url("/assets/icon-48x48.png")}" />
    <link rel="icon" type="image/png" sizes="64x64" href="{url("/assets/icon-64x64.png")}" />
    <link rel="apple-touch-icon" sizes="180x" href="{url("/assets/apple-touch-icon.png")}" />
    <link rel="icon" type="image/png" sizes="192x192" href="{url("/assets/android-chrome-icon-192x192.png")}" />
    <link rel="icon" type="image/png" sizes="512x512" href="{url("/assets/android-chrome-icon-512x512.png")}" />
    <link href="{url("/assets/tailwind.css")}" rel="stylesheet" />
    <link href="{url("/assets/extras.css")}" rel="stylesheet" />
    <script src="{url("/assets/htmx.bundle.js")}"></script>"""


def nav_html(active: str = "") -> str:
    models_cls = (
        "underline"
        if active == "models"
        else "hover:underline focus:underline focus:outline-none focus:ring-0"
    )
    return f"""<header class="sticky top-0 z-40 bg-white dark:bg-neutral-950 underline-offset-4 lg:static">
  <nav class="flex w-full items-center justify-between px-6 py-[9px]">
    <a href="{url("/search/")}" class="z-50">
      <img src="{url("/assets/ollama.png")}" class="w-8 dark:invert" alt="Ollama" />
    </a>
    <div class="hidden lg:flex xl:flex-1 items-center space-x-6 ml-6 mr-6 xl:mr-0 text-lg">
      <a class="{models_cls}" href="{url("/search/")}">Models</a>
      <a class="hover:underline focus:underline focus:outline-none focus:ring-0" href="https://docs.ollama.com">Docs</a>
      <a class="hover:underline focus:underline focus:outline-none focus:ring-0" href="https://ollama.com/pricing">Pricing</a>
    </div>
    <div class="flex-grow justify-center items-center hidden lg:flex">
      <div class="relative w-full xl:max-w-[28rem]">
        <form action="{url("/search")}" autocomplete="off" id="nav-search-form">
          <div class="relative flex w-full appearance-none bg-black/5 dark:bg-white/5 border border-neutral-100 dark:border-neutral-800 items-center rounded-full">
            <span class="pl-2 text-2xl text-neutral-500 dark:text-neutral-400">{SVG_SEARCH}</span>
            <input id="navbar-input" name="q" type="text" class="resize-none rounded-full border-0 py-2.5 bg-transparent text-sm w-full placeholder:text-neutral-500 dark:placeholder:text-neutral-500 focus:outline-none focus:ring-0 dark:text-neutral-200" placeholder="Search models" autocomplete="off" hx-on:keydown="if(event.key==='Enter'){{event.preventDefault();window.location.href='{url("/search/?q=")}'+encodeURIComponent(this.value);}}" />
          </div>
        </form>
      </div>
    </div>
    <div class="hidden lg:flex xl:flex-1 items-center space-x-2 justify-end ml-6 xl:ml-0">
      <button id="theme-toggle" class="flex cursor-pointer items-center rounded-full bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 text-lg px-3 py-1.5 text-black dark:text-neutral-200 whitespace-nowrap" title="Toggle dark mode">
        <span class="dark:hidden">{SVG_MOON}</span>
        <span class="hidden dark:block">{SVG_SUN}</span>
      </button>
      <a class="flex cursor-pointer items-center rounded-full bg-neutral-800 dark:bg-neutral-100 text-lg px-4 py-1.5 text-white dark:text-neutral-900 hover:bg-black dark:hover:bg-white whitespace-nowrap focus:bg-black dark:focus:bg-white" href="https://ollama.com/download">Download</a>
    </div>
    <div class="lg:hidden flex items-center">
      <button id="theme-toggle-mobile" class="flex items-center rounded-full bg-black/5 dark:bg-white/10 px-3 py-1.5 mr-2 text-black dark:text-neutral-200">
        <span class="dark:hidden">{SVG_MOON}</span>
        <span class="hidden dark:block">{SVG_SUN}</span>
      </button>
    </div>
  </nav>
</header>"""


def footer_html() -> str:
    return """<footer class="mt-auto">
  <div class="underline-offset-4 hidden md:block">
    <div class="flex items-center justify-between px-6 py-3.5">
      <div class="text-xs text-neutral-500 dark:text-neutral-400">&copy; 2026 Ollama</div>
      <div class="flex space-x-6 text-xs text-neutral-500 dark:text-neutral-400">
        <a href="https://ollama.com/download" class="hover:underline">Download</a>
        <a href="https://ollama.com/blog" class="hover:underline">Blog</a>
        <a href="https://docs.ollama.com" class="hover:underline">Docs</a>
        <a href="https://github.com/ollama/ollama" class="hover:underline">GitHub</a>
        <a href="https://ollama.com/pricing" class="hover:underline">Pricing</a>
      </div>
    </div>
  </div>
</footer>"""


def theme_script() -> str:
    return """<script>
(function() {
  const stored = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (stored === 'dark' || (!stored && prefersDark)) {
    document.documentElement.classList.add('dark');
  }
  function toggle() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }
  document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('theme-toggle')?.addEventListener('click', toggle);
    document.getElementById('theme-toggle-mobile')?.addEventListener('click', toggle);
  });
})();
</script>"""


# --------------------------------------------------------------------------- #
# Card rendering (mirrors /library page exactly)
# --------------------------------------------------------------------------- #


def capability_spans(capabilities: list[str], cloud: bool) -> str:
    parts = []
    for cap in capabilities:
        parts.append(
            f'<span x-test-capability class="inline-flex my-1 items-center rounded-md '
            f"bg-indigo-50 dark:bg-indigo-950/50 px-2 py-[2px] text-xs font-medium "
            f'text-indigo-600 dark:text-indigo-400 sm:text-[13px]">{esc(cap)}</span>'
        )
    if cloud:
        parts.append(
            '<span class="inline-flex my-1 items-center rounded-md bg-cyan-50 '
            "dark:bg-cyan-950/50 px-2 py-[2px] text-xs font-medium text-cyan-500 "
            'dark:text-cyan-400 sm:text-[13px]">cloud</span>'
        )
    return "\n        ".join(parts) if parts else ""


def size_spans(sizes: list[str]) -> str:
    parts = []
    for s in sizes:
        parts.append(
            f'<span x-test-size class="inline-flex my-1 items-center rounded-md '
            f"bg-[#ddf4ff] dark:bg-blue-950/50 px-2 py-[2px] text-xs font-medium "
            f'text-blue-600 dark:text-blue-400 sm:text-[13px]">{esc(s)}</span>'
        )
    return "\n        ".join(parts) if parts else ""


def render_card(
    m: dict, tags: list[dict] | None = None, ranks: dict | None = None
) -> str:
    name = esc(m["name"])
    name_raw = m["name"]
    desc = esc(m["description"])
    caps = capability_spans(m["capabilities"], m["cloud"])
    sizes = size_spans(m["sizes"])
    pulls = format_count(m["pulls"])
    tag_count = m["tag_count"]
    tag_label = "Tag" if tag_count == 1 else "Tags"
    updated = esc(m["updated"])
    updated_title = esc(m.get("updated_title") or "")
    href = url(esc(m["path"]))

    # Sort rank data attributes
    r = (ranks or {}).get(name_raw, {})
    data_attrs = (
        f'data-popular-rank="{r.get("popular_rank", 9999)}" '
        f'data-newest-rank="{r.get("newest_rank", 9999)}" '
        f'data-oldest-rank="{r.get("oldest_rank", 9999)}" '
        f'data-updated-rank="{r.get("updated_rank", 9999)}" '
        f'data-pulls="{m["pulls"]}" '
        f'data-tag-count="{tag_count}" '
        f'data-sizes-count="{len(m["sizes"])}" '
        f'data-name="{esc(name_raw).lower()}"'
    )

    # MLX pill for models that have MLX variants (black bg, white text, same size as other pills)
    fmt_chip = ""
    if tags and has_mlx(tags):
        fmt_chip = (
            '<span class="inline-flex my-1 items-center rounded-md '
            "bg-neutral-900 px-2 py-[2px] text-xs font-medium text-white "
            'dark:bg-white dark:text-neutral-900 sm:text-[13px]">MLX</span>'
        )

    return f"""  <li x-test-model {data_attrs} class="flex items-baseline border-b border-neutral-200 dark:border-neutral-800 py-6">
  <a href="{href}" class="group w-full">
    <div class="flex flex-col mb-1" title="{name}">
      <h2 class="truncate text-xl font-medium underline-offset-2 group-hover:underline md:text-2xl dark:text-neutral-100">
        <span x-test-search-response-title>{name}</span>
      </h2>
      <p class="max-w-lg break-words text-neutral-800 dark:text-neutral-300 text-md">{desc}</p>
    </div>
    <div class="flex flex-col">
      <div class="flex flex-wrap space-x-2">
        {caps}
        {fmt_chip}
        {sizes}
      </div>
      <p class="my-1 flex space-x-5 text-[13px] font-medium text-neutral-500 dark:text-neutral-400">
        <span class="flex items-center">
          {SVG_DOWNLOAD}
          <span x-test-pull-count>{pulls}</span>
          <span class="hidden sm:flex">&nbsp;Pulls</span>
        </span>
        <span class="flex items-center">
          {SVG_TAG}
          <span x-test-tag-count>{tag_count}</span>
          <span class="hidden sm:flex">&nbsp;{tag_label}</span>
        </span>
        <span class="flex items-center" title="{updated_title}">
          {SVG_CLOCK}
          <span class="hidden sm:flex">Updated&nbsp;</span>
          <span x-test-updated>{updated}</span>
        </span>
      </p>
    </div>
  </a>
</li>"""


# --------------------------------------------------------------------------- #
# Index / search page
# --------------------------------------------------------------------------- #


def build_index(models: list[dict], ranks: dict) -> None:
    cards = "\n".join(render_card(m, load_tags(m["path"]), ranks) for m in models)

    # Capability filter chips (Embedding/Vision/Tools/Thinking — no Cloud, it's a dropdown)
    chip_labels = ["Embedding", "Vision", "Tools", "Thinking"]
    chip_values = ["embedding", "vision", "tools", "thinking"]
    chips = []
    for label, val in zip(chip_labels, chip_values):
        chips.append(
            f"""      <div class="relative inline-block mr-1.5 mb-1.5">
        <input type="checkbox" name="c" value="{val}" id="cap-{val}" class="peer sr-only cap-filter" data-cap="{val}">
        <label for="cap-{val}" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 focus:outline-none focus:ring-0 focus:ring-transparent min-md:hover:bg-neutral-100 dark:min-md:hover:bg-neutral-800 select-none">{label}</label>
      </div>"""
        )
    chips_html = "\n".join(chips)

    # Cloud dropdown: All models / Cloud only / Local only
    cloud_dropdown = """      <select id="cloud-filter" class="mr-1.5 mb-1.5 px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 bg-white dark:bg-neutral-950 focus:outline-none focus:ring-0 appearance-none">
        <option value="all">All models</option>
        <option value="cloud">Cloud only</option>
        <option value="local">Local only</option>
      </select>"""

    # Sort options
    sort_options = [
        ("popular", "Popular"),
        ("newest", "Newest"),
        ("oldest", "Oldest"),
        ("updated", "Recently updated"),
        ("pulls", "Pulls"),
        ("tags", "Tags"),
        ("name", "Name"),
    ]
    opt_html = "\n".join(
        f'        <option value="{v}">{l}</option>' for v, l in sort_options
    )

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html("Ollama", "Search for models on Ollama.")}
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html("models")}

<main class="mx-auto flex w-full max-w-2xl flex-col px-6 py-5 md:py-12 lg:px-8">
  <input type="hidden" id="sort-value" name="o" value="popular">

  <!-- Mobile search bar -->
  <div class="flex lg:hidden justify-between space-x-2 items-center">
    <div class="relative flex w-full appearance-none bg-black/5 dark:bg-white/5 border border-neutral-100 dark:border-neutral-700 items-center rounded-full">
      <span class="pl-4 text-neutral-400">{SVG_SEARCH}</span>
      <input id="form-input" name="q" type="search" value="" class="resize-none rounded-full border-0 py-2.5 bg-transparent text-base sm:text-sm w-full placeholder:text-neutral-400 focus:outline-none focus:ring-0 dark:text-neutral-200" placeholder="Search models" autofocus autocomplete="off">
    </div>
    <div class="sm:hidden block relative">
      <select id="mobile-sort-select" class="absolute inset-0 w-6 px-3 py-1 opacity-0 appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:ring focus:outline-none focus:ring-blue-300 focus:ring-opacity-75 focus:border-blue-400 dark:focus:border-blue-600">
{opt_html}
      </select>
      <div class="w-6 px-3.5 py-1.5 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 flex items-center justify-center pointer-events-none">
        <span class="text-neutral-900 dark:text-neutral-100 text-xs font-medium">&#x21C5;</span>
      </div>
    </div>
  </div>

  <div id="searchresults" class="w-full space-y-2">
    <div class="flex flex-wrap items-center justify-between gap-2 mt-2">
      <fieldset class="flex flex-wrap items-center">
{chips_html}
{cloud_dropdown}
      </fieldset>
      <div class="hidden sm:block">
        <select id="desktop-sort-select" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:ring focus:outline-none focus:ring-blue-300 focus:ring-opacity-75 focus:border-blue-400 dark:focus:border-blue-600 min-w-[120px] text-sm px-3 py-1.5">
{opt_html}
        </select>
      </div>
    </div>

    <ul role="list" id="card-list" class="grid grid-cols-1">
{cards}
    </ul>
    <p id="no-results" class="hidden py-12 text-center text-neutral-400 dark:text-neutral-600">No models found.</p>
  </div>
</main>

{footer_html()}
{theme_script()}
<script src="{url("/assets/app.js")}"></script>
</body>
</html>"""

    (PUBLIC / "search").mkdir(parents=True, exist_ok=True)
    (PUBLIC / "search" / "index.html").write_text(page)


# --------------------------------------------------------------------------- #
# Model detail page
# --------------------------------------------------------------------------- #


def render_models_table(tags: list[dict], fmt_filter: str = "all") -> str:
    """Render the 'Models' summary table for the detail page."""
    if fmt_filter != "all":
        tags = [t for t in tags if t["format"] == fmt_filter]
    if not tags:
        return '<p class="text-neutral-400 text-sm py-4">No tags for this format.</p>'
    rows = []
    for t in tags[:10]:  # show first 10, like the original summary
        name = esc(t["name"])
        size = esc(t["size_text"]) or "—"
        ctx = esc(t["context"]) or "—"
        inp = esc(t["input_type"]) or "—"
        rows.append(
            f'<div class="flex justify-between items-center py-1.5 text-sm border-b border-neutral-100 dark:border-neutral-800">\n'
            f'  <a href="tags" class="hover:underline font-medium">{name}</a>\n'
            f'  <span class="text-neutral-500 dark:text-neutral-400 text-[13px]">{size} · {ctx} context window · {inp}</span>\n'
            f"</div>"
        )
    body = "\n".join(rows)
    more = (
        f'<a href="tags" class="text-sm text-blue-600 dark:text-blue-400 hover:underline mt-2 inline-block">View all {len(tags)} &rarr;</a>'
        if len(tags) > 10
        else ""
    )
    return body + more


def build_detail(m: dict, tags: list[dict]) -> None:
    name = m["name"]
    desc = m["description"]
    path = m["path"]
    slug_dir = PUBLIC / path.strip("/")
    slug_dir.mkdir(parents=True, exist_ok=True)

    caps = capability_spans(m["capabilities"], m["cloud"])
    sizes = size_spans(m["sizes"])
    pulls = format_count(m["pulls"])
    tag_count = m["tag_count"]
    tag_label = "Tag" if tag_count == 1 else "Tags"
    updated = m["updated"]

    # tabs: All / GGUF / MLX (only if MLX present)
    has_m = has_mlx(tags)
    tabs = ""
    if has_m:
        tabs = """<div class="flex space-x-1 mb-4 border-b border-neutral-200 dark:border-neutral-800">
  <button data-fmt="all" class="fmt-tab px-3 py-2 text-sm font-medium border-b-2 border-blue-500 text-blue-600 dark:text-blue-400">All</button>
  <button data-fmt="gguf" class="fmt-tab px-3 py-2 text-sm font-medium border-b-2 border-transparent text-neutral-500 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200">GGUF</button>
  <button data-fmt="mlx" class="fmt-tab px-3 py-2 text-sm font-medium border-b-2 border-transparent text-neutral-500 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200">MLX</button>
</div>"""

    models_table_all = render_models_table(tags, "all")
    models_table_gguf = render_models_table(tags, "gguf") if has_m else ""
    models_table_mlx = render_models_table(tags, "mlx") if has_m else ""

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html(name, desc)}
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html()}

<main class="mx-auto flex w-full max-w-2xl flex-col px-6 py-5 md:py-12 lg:px-8">
  <a href="{url("/search/")}" class="text-sm text-neutral-500 dark:text-neutral-400 hover:underline mb-4">&larr; Back to models</a>
  <div class="flex flex-col mb-4">
    <h1 class="text-2xl font-medium tracking-tight dark:text-neutral-100">{esc(name)}</h1>
    <p class="text-neutral-500 dark:text-neutral-400 text-sm mt-1">{format_count(m["pulls"])} &nbsp;Downloads &nbsp; Updated &nbsp;{esc(updated)}</p>
  </div>
  <p class="max-w-lg break-words text-neutral-800 dark:text-neutral-300 text-md mb-4">{esc(desc)}</p>
  <div class="flex flex-wrap space-x-2 mb-6">
    {caps}
    {sizes}
  </div>

  <div class="mb-8">
    <h2 class="text-lg font-medium mb-2 dark:text-neutral-100">Models</h2>
    {tabs}
    <div id="models-table-all" class="fmt-table">{models_table_all}</div>
    {'<div id="models-table-gguf" class="fmt-table hidden">' + models_table_gguf + "</div>" if has_m else ""}
    {'<div id="models-table-mlx" class="fmt-table hidden">' + models_table_mlx + "</div>" if has_m else ""}
  </div>

  <div class="mb-8">
    <h2 class="text-lg font-medium mb-2 dark:text-neutral-100">CLI</h2>
    <pre class="bg-black/5 dark:bg-white/5 rounded-lg p-3 text-sm overflow-x-auto dark:text-neutral-200"><code>ollama run {esc(name)}</code></pre>
  </div>
</main>

{footer_html()}
{theme_script()}
<script src="{url("/assets/app.js")}"></script>
</body>
</html>"""

    (slug_dir / "index.html").write_text(page)


# --------------------------------------------------------------------------- #
# Tags page
# --------------------------------------------------------------------------- #


def render_tag_row(t: dict, model_name: str) -> str:
    name = esc(t["name"])
    full = f"{model_name}:{t['name']}"
    size = esc(t["size_text"]) or "—"
    ctx = esc(t["context"]) or "—"
    inp = esc(t["input_type"]) or "—"
    digest = esc(t["digest"]) or ""
    updated = esc(t["updated"]) or ""
    return f"""        <div class="group px-4 py-3 border-b border-neutral-100 dark:border-neutral-800">
            <div class="grid grid-cols-12 items-center gap-2">
                <span class="flex items-center font-medium col-span-6 group text-sm dark:text-neutral-100">
                    <a href="../{esc(t["name"])}" class="group-hover:underline">{full}</a>
                    <input class="command hidden" value="{esc(full)}" />
                    <button class="hidden group-hover:inline-flex ml-1.5 text-neutral-500 hover:text-black dark:hover:text-white items-center" onclick="copyToClipboard(this)">
                        {SVG_COPY}
                    </button>
                </span>
                <p class="col-span-2 text-neutral-500 dark:text-neutral-400 text-[13px]">{size}</p>
                <p class="col-span-2 text-neutral-500 dark:text-neutral-400 text-[13px]">{ctx}</p>
                <div class="col-span-2 text-neutral-500 dark:text-neutral-400 text-[13px]">{inp}</div>
            </div>
            <div class="flex text-neutral-500 dark:text-neutral-500 text-xs items-center mt-1">
                <span class="font-mono text-[11px]">{digest}</span>&nbsp;·&nbsp;{updated}
            </div>
        </div>"""


def build_tags_page(m: dict, tags: list[dict]) -> None:
    name = m["name"]
    desc = m["description"]
    path = m["path"]
    slug_dir = PUBLIC / path.strip("/") / "tags"
    slug_dir.mkdir(parents=True, exist_ok=True)

    has_m = has_mlx(tags)
    gguf_tags = [t for t in tags if t["format"] == "gguf"]
    mlx_tags = [t for t in tags if t["format"] == "mlx"]

    tabs = ""
    if has_m:
        tabs = f"""<div class="flex space-x-1 mb-4 border-b border-neutral-200 dark:border-neutral-800">
  <button data-fmt="all" class="fmt-tab px-3 py-2 text-sm font-medium border-b-2 border-blue-500 text-blue-600 dark:text-blue-400">All ({len(tags)})</button>
  <button data-fmt="gguf" class="fmt-tab px-3 py-2 text-sm font-medium border-b-2 border-transparent text-neutral-500 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200">GGUF ({len(gguf_tags)})</button>
  <button data-fmt="mlx" class="fmt-tab px-3 py-2 text-sm font-medium border-b-2 border-transparent text-neutral-500 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200">MLX ({len(mlx_tags)})</button>
</div>"""

    rows_all = "\n".join(render_tag_row(t, name) for t in tags)
    rows_gguf = "\n".join(render_tag_row(t, name) for t in gguf_tags)
    rows_mlx = "\n".join(render_tag_row(t, name) for t in mlx_tags)

    def table_block(rows: str, count: int, fmt_id: str) -> str:
        header = (
            '<div class="grid grid-cols-12 items-center px-4 py-2 border-b border-neutral-200 dark:border-neutral-700 text-[13px] font-medium text-neutral-500 dark:text-neutral-400">\n'
            '  <p class="col-span-6">Name</p>\n'
            '  <p class="col-span-2">Size / Usage</p>\n'
            '  <p class="col-span-2">Context</p>\n'
            '  <p class="col-span-2">Input</p>\n'
            "</div>"
        )
        return f'<div id="tags-table-{fmt_id}" class="fmt-table">{header}\n{rows}</div>'

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html(f"{name} Tags", f"Tags for {name}. {desc}")}
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html()}

<main class="mx-auto flex w-full max-w-2xl flex-col px-6 py-5 md:py-12 lg:px-8">
  <a href="../" class="text-sm text-neutral-500 dark:text-neutral-400 hover:underline mb-4">&larr; Back to {esc(name)}</a>
  <h1 class="text-2xl font-medium tracking-tight dark:text-neutral-100 mb-1">{esc(name)} Tags</h1>
  <p class="text-neutral-500 dark:text-neutral-400 text-sm mb-6">{len(tags)} tags</p>
  {tabs}
  {table_block(rows_all, len(tags), "all")}
  {table_block(rows_gguf, len(gguf_tags), "gguf") if has_m else "<!-- no mlx -->".replace("no mlx", "no gguf") if not gguf_tags else ""}
  {table_block(rows_mlx, len(mlx_tags), "mlx") if has_m else ""}
</main>

{footer_html()}
{theme_script()}
<script src="{url("/assets/app.js")}"></script>
</body>
</html>"""

    (slug_dir / "index.html").write_text(page)


# --------------------------------------------------------------------------- #
# Static assets
# --------------------------------------------------------------------------- #


def copy_assets() -> None:
    assets = PUBLIC / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # Download vendored assets from ollama.com if missing.
    vendored = [
        ("tailwind.css", "https://ollama.com/public/tailwind.css"),
        ("htmx.bundle.js", "https://ollama.com/public/vendor/htmx/bundle.js"),
        ("ollama.png", "https://ollama.com/public/ollama.png"),
    ]
    for name, url in vendored:
        dst = assets / name
        if dst.exists():
            continue
        try:
            import urllib.request

            urllib.request.urlretrieve(url, dst)
            print(f"  downloaded {name}")
        except Exception as e:
            print(f"  WARN: could not download {name}: {e}", file=sys.stderr)

    # Icons (download if missing).
    for icon, url in [
        ("icon-16x16.png", "https://ollama.com/public/icon-16x16.png"),
        ("icon-32x32.png", "https://ollama.com/public/icon-32x32.png"),
        ("icon-48x48.png", "https://ollama.com/public/icon-48x48.png"),
        ("icon-64x64.png", "https://ollama.com/public/icon-64x64.png"),
        ("apple-touch-icon.png", "https://ollama.com/public/apple-touch-icon.png"),
    ]:
        dst = assets / icon
        if dst.exists():
            continue
        try:
            import urllib.request

            urllib.request.urlretrieve(url, dst)
            print(f"  downloaded {icon}")
        except Exception as e:
            print(f"  WARN: could not download {icon}: {e}", file=sys.stderr)
    # extras.css
    (assets / "extras.css").write_text(EXTRAS_CSS)
    # app.js
    (assets / "app.js").write_text(APP_JS)


EXTRAS_CSS = r"""/* Dark mode overrides for ollama-search.
   The vendored tailwind.css from ollama.com doesn't include dark: variants,
   so we add all dark: class definitions we use here.
   Color mapping: Tailwind shade inversion (50→950, 100→900, 200→800, …, 950→50)
   using official Tailwind v3 palette hex values. */

.dark { color-scheme: dark; }
.dark body { background-color: #0a0a0a; color: #e5e5e5; }
.dark header { background-color: #0a0a0a; }

/* --- Base elements: override light-mode neutral classes in dark context --- */
.dark .dark\:invert { filter: invert(1); }
.dark .text-neutral-800 { color: #d4d4d4; }
.dark .text-neutral-500 { color: #a3a3a3; }
.dark .text-neutral-900 { color: #fafafa; }
.dark .border-neutral-200 { border-color: #404040; }
.dark .border-neutral-100 { border-color: #333333; }
.dark .border-neutral-300 { border-color: #525252; }
.dark .bg-white { background-color: #0a0a0a; }
.dark .bg-black\/5 { background-color: rgba(255,255,255,0.05); }
.dark .hover\:bg-black\/10:hover { background-color: rgba(255,255,255,0.10); }
.dark .hover\:bg-neutral-50:hover { background-color: #262626; }
.dark .placeholder\:text-neutral-500::placeholder { color: #737373; }
.dark .text-black { color: #fafafa; }
.dark a:focus\:underline:focus { text-decoration: underline; }

/* --- Light-mode color classes not in vendored tailwind.css --- */
.bg-cyan-50 { background-color: rgb(236 254 255); }
.border-cyan-200 { border-color: rgb(165 243 252); }
.text-cyan-600 { color: rgb(8 145 178); }
.bg-blue-50 { background-color: rgb(239 246 255); }
.border-blue-200 { border-color: rgb(191 219 254); }
.text-blue-600 { color: rgb(37 99 235); }

/* --- Dark: neutral classes (official Tailwind v3 palette) --- */
.dark .dark\:bg-neutral-900 { background-color: #171717; }
.dark .dark\:bg-neutral-950 { background-color: #0a0a0a; }
.dark .dark\:bg-neutral-800 { background-color: #262626; }
.dark .dark\:bg-neutral-100 { background-color: #f5f5f5; }
.dark .dark\:bg-white\/5 { background-color: rgba(255,255,255,0.05); }
.dark .dark\:bg-white\/10 { background-color: rgba(255,255,255,0.10); }
.dark .dark\:bg-white { background-color: #ffffff; }
.dark .dark\:hover\:bg-white\/20:hover { background-color: rgba(255,255,255,0.20); }
.dark .dark\:hover\:bg-neutral-800:hover { background-color: #262626; }
.dark .dark\:hover\:bg-white:hover { background-color: #ffffff; }
.dark .dark\:peer-checked\:bg-neutral-800:checked ~ label { background-color: #262626; }
.dark .dark\:text-neutral-100 { color: #f5f5f5; }
.dark .dark\:text-neutral-200 { color: #e5e5e5; }
.dark .dark\:text-neutral-300 { color: #d4d4d4; }
.dark .dark\:text-neutral-400 { color: #a3a3a3; }
.dark .dark\:text-neutral-500 { color: #737373; }
.dark .dark\:text-neutral-600 { color: #525252; }
.dark .dark\:text-neutral-900 { color: #171717; }
.dark .dark\:border-neutral-700 { border-color: #404040; }
.dark .dark\:border-neutral-800 { border-color: #262626; }
.dark .dark\:hover\:text-neutral-200:hover { color: #e5e5e5; }
.dark .dark\:placeholder\:text-neutral-500::placeholder { color: #737373; }
.dark .dark\:focus\:bg-white:focus { background-color: #ffffff; }

/* --- Dark: colored badge classes (Tailwind shade inversion) --- */
/* Capability badges: indigo-50 bg → indigo-950, indigo-600 text → indigo-400 */
.dark .dark\:bg-indigo-950\/50 { background-color: rgba(30, 27, 75, 0.5); }
.dark .dark\:bg-indigo-950 { background-color: #1e1b4b; }
.dark .dark\:text-indigo-400 { color: #818cf8; }
/* Cloud badge: cyan-50 bg → cyan-950, cyan-500 text → cyan-400 */
.dark .dark\:bg-cyan-950\/50 { background-color: rgba(8, 51, 68, 0.5); }
.dark .dark\:bg-cyan-950 { background-color: #083344; }
.dark .dark\:text-cyan-400 { color: #22d3ee; }
.dark .dark\:text-cyan-500 { color: #06b6d4; }
.dark .dark\:border-cyan-800 { border-color: #155e75; }
/* Size badges: blue-50 bg → blue-950, blue-600 text → blue-400 */
.dark .dark\:bg-blue-950\/50 { background-color: rgba(23, 37, 84, 0.5); }
.dark .dark\:bg-blue-950 { background-color: #172554; }
.dark .dark\:text-blue-400 { color: #60a5fa; }
.dark .dark\:border-blue-800 { border-color: #1e40af; }
/* Tabs: blue-500 border stays blue-500, blue-600 text → blue-400 */
.dark .dark\:border-blue-500 { border-color: #3b82f6; }
.dark .dark\:border-blue-600 { border-color: #2563eb; }
/* Focus state */
.dark .dark\:focus\:border-blue-600:focus { border-color: #2563eb; }
/* Emerald (reserved for future use) */
.dark .dark\:bg-emerald-950\/50 { background-color: rgba(2, 44, 34, 0.5); }
.dark .dark\:bg-emerald-950 { background-color: #022c22; }
.dark .dark\:text-emerald-400 { color: #34d399; }
"""

APP_JS = r"""// ollama-search frontend logic.
// Filtering, sorting, dark-mode, tab switching, copy-to-clipboard.

function copyToClipboard(btn) {
  var input = btn.parentElement.querySelector('input.command');
  if (!input) return;
  navigator.clipboard.writeText(input.value).then(function() {
    var copyIcon = btn.querySelector('.copy-icon');
    var checkIcon = btn.querySelector('.check-icon');
    if (copyIcon) copyIcon.classList.add('hidden');
    if (checkIcon) checkIcon.classList.remove('hidden');
    setTimeout(function() {
      if (copyIcon) copyIcon.classList.remove('hidden');
      if (checkIcon) checkIcon.classList.add('hidden');
    }, 1500);
  });
}
window.copyToClipboard = copyToClipboard;

// --- Search page: filter + sort + capability chips ---

function getSelectedCaps() {
  var caps = [];
  document.querySelectorAll('.cap-filter').forEach(function(cb) {
    if (cb.checked) caps.push(cb.getAttribute('data-cap'));
  });
  return caps;
}

function getSort() {
  var sortEl = document.getElementById('desktop-sort-select') || document.getElementById('mobile-sort-select');
  return sortEl ? sortEl.value : 'popular';
}

function getQuery() {
  var input = document.getElementById('form-input') || document.getElementById('navbar-input');
  return input ? input.value.toLowerCase().trim() : '';
}

function getCloudFilter() {
  var el = document.getElementById('cloud-filter');
  return el ? el.value : 'all';
}

function applyFilters() {
  var q = getQuery();
  var caps = getSelectedCaps();
  var sort = getSort();
  var cloudFilter = getCloudFilter();
  var list = document.getElementById('card-list');
  if (!list) return;
  var cards = Array.from(list.querySelectorAll('li[x-test-model]'));
  // Filter
  var visible = 0;
  cards.forEach(function(card) {
    var title = card.querySelector('[x-test-search-response-title]') ? card.querySelector('[x-test-search-response-title]').textContent.toLowerCase() : '';
    var desc = card.querySelector('p.break-words') ? card.querySelector('p.break-words').textContent.toLowerCase() : '';
    var cardCaps = [];
    card.querySelectorAll('[x-test-capability]').forEach(function(el) { cardCaps.push(el.textContent.toLowerCase()); });
    var isCloud = card.querySelector('[class*="text-cyan"]') !== null;
    var matchText = !q || title.indexOf(q) !== -1 || desc.indexOf(q) !== -1;
    var matchCaps = caps.length === 0 || caps.every(function(c) { return cardCaps.indexOf(c) !== -1; });
    var matchCloud = cloudFilter === 'all' || (cloudFilter === 'cloud' && isCloud) || (cloudFilter === 'local' && !isCloud);
    var show = matchText && matchCaps && matchCloud;
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  var noRes = document.getElementById('no-results');
  if (noRes) noRes.classList.toggle('hidden', visible > 0);
  // Sort — always reorder DOM using data-* rank attributes
  var rankAttr = {
    'popular': 'data-popular-rank',
    'newest': 'data-newest-rank',
    'oldest': 'data-oldest-rank',
    'updated': 'data-updated-rank',
    'pulls': 'data-pulls',
    'tags': 'data-sizes-count',
    'name': 'data-name',
  };
  var attr = rankAttr[sort] || rankAttr['popular'];
  var descending = (sort === 'pulls' || sort === 'tags');
  cards.sort(function(a, b) {
    var va = a.getAttribute(attr) || '';
    var vb = b.getAttribute(attr) || '';
    var cmp;
    if (sort === 'name') {
      cmp = va.localeCompare(vb);
    } else {
      var na = parseFloat(va) || 0;
      var nb = parseFloat(vb) || 0;
      cmp = na - nb;
      if (descending) cmp = -cmp;
    }
    return cmp;
  });
  cards.forEach(function(c) { list.appendChild(c); });
}

// --- Format tabs (detail + tags pages) ---
function initTabs() {
  var tabs = document.querySelectorAll('.fmt-tab');
  if (!tabs.length) return;
  tabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
      var fmt = tab.getAttribute('data-fmt');
      tabs.forEach(function(t) {
        var active = t === tab;
        t.classList.toggle('border-blue-500', active);
        t.classList.toggle('text-blue-600', active);
        t.classList.toggle('dark:text-blue-400', active);
        t.classList.toggle('border-transparent', !active);
        t.classList.toggle('text-neutral-500', !active);
      });
      document.querySelectorAll('.fmt-table').forEach(function(tbl) {
        var id = tbl.id.replace('tags-table-', '').replace('models-table-', '');
        tbl.classList.toggle('hidden', id !== fmt);
      });
    });
  });
}

// Sync mobile and desktop sort selects
function syncSort(source, target) {
  if (source && target) {
    source.addEventListener('change', function() { target.value = source.value; applyFilters(); });
  }
}

document.addEventListener('DOMContentLoaded', function() {
  var desktopSort = document.getElementById('desktop-sort-select');
  var mobileSort = document.getElementById('mobile-sort-select');
  if (desktopSort && mobileSort) {
    syncSort(desktopSort, mobileSort);
    syncSort(mobileSort, desktopSort);
  }

  if (document.getElementById('card-list')) {
    var formInput = document.getElementById('form-input');
    var navInput = document.getElementById('navbar-input');
    if (formInput) formInput.addEventListener('input', applyFilters);
    if (navInput) navInput.addEventListener('input', applyFilters);
    document.querySelectorAll('.cap-filter').forEach(function(cb) { cb.addEventListener('change', applyFilters); });
    var cloudFilter = document.getElementById('cloud-filter');
    if (cloudFilter) cloudFilter.addEventListener('change', applyFilters);
    if (desktopSort) desktopSort.addEventListener('change', applyFilters);
    if (mobileSort) mobileSort.addEventListener('change', applyFilters);
    // read ?q= from URL query string
    var params = new URLSearchParams(location.search);
    var q = params.get('q');
    if (q) {
      if (formInput) formInput.value = q;
      if (navInput) navInput.value = q;
    }
    applyFilters();
  }
  initTabs();
});
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    global BASE
    if "--base" in sys.argv:
        idx = sys.argv.index("--base")
        BASE = sys.argv[idx + 1].rstrip("/") if idx + 1 < len(sys.argv) else ""
    models = load_models()
    print(f"loaded {len(models)} models from scraper/models.json")

    # ensure public dir exists
    PUBLIC.mkdir(parents=True, exist_ok=True)
    copy_assets()

    # index/search page -> /search/index.html
    print("building search/index.html ...")
    build_index(models, load_ranks())

    # root redirect -> /search/
    print("building root redirect ...")
    search_url = url("/search/")
    (PUBLIC / "index.html").write_text(
        "<!DOCTYPE html>\n<html><head>"
        f'<meta http-equiv="refresh" content="0; url={search_url}">'
        '<meta name="robots" content="noindex"></head>'
        f'<body><a href="{search_url}">Redirect to search</a></body></html>\n'
    )

    # model detail + tags pages
    print("building model detail + tags pages ...")
    for i, m in enumerate(models, 1):
        tags = load_tags(m["path"])
        build_detail(m, tags)
        build_tags_page(m, tags)
        if i % 50 == 0:
            print(f"  {i}/{len(models)}")
    print(f"built {len(models)} model pages + tags pages")

    # write the catalog JSON for client-side use
    (PUBLIC / "assets" / "models.json").write_text(json.dumps(models, indent=2))
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

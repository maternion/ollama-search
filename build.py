#!/usr/bin/env python3
"""Build a static site from scraped ollama.com data.

Generates:
  public/index.html                       search page (main page)
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
PAGES_DIR = SCRAPER / "pages"
TAG_PAGES_DIR = SCRAPER / "tag_pages"

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
    '<svg class="copy-icon h-[20px] w-[20px]" xmlns="http://www.w3.org/2000/svg" '
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

SVG_EXTERNAL = (
    '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
    'stroke-width="2" stroke="currentColor" class="w-3.5 h-3.5">'
    '<path stroke-linecap="round" stroke-linejoin="round" '
    'd="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" /></svg>'
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
    """Format a pull count the way ollama.com does.

    - n < 10,000: exact with thousands separators (e.g. 5402 -> "5,402").
    - 10,000 <= n < 1,000,000: K suffix with one decimal, dropping a
      trailing ".0" (e.g. 11500 -> "11.5K", 986700 -> "986.7K").
    - n >= 1,000,000: M suffix with one decimal, dropping ".0"
      (e.g. 10900000 -> "10.9M", 31000000 -> "31M").
    - n >= 1,000,000,000: B suffix (same rounding rules).
    """
    if n < 10_000:
        return f"{n:,}"
    for threshold, divisor, suffix in (
        (1_000_000_000, 1_000_000_000, "B"),
        (1_000_000, 1_000_000, "M"),
        (10_000, 1_000, "K"),
    ):
        if n >= threshold:
            v = n / divisor
            s = f"{v:.1f}{suffix}"
            # Drop a trailing ".0" so "31.0M" becomes "31M".
            if s.endswith(f".0{suffix}"):
                s = s.replace(f".0{suffix}", suffix)
            return s
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


def load_model_page(model_path: str) -> dict | None:
    pf = PAGES_DIR / f"{slugify(model_path)}.json"
    if not pf.exists():
        return None
    return json.loads(pf.read_text())


def load_tag_page(model_path: str, tag_name: str) -> dict | None:
    slug = slugify(model_path)
    tf = TAG_PAGES_DIR / f"{slug}__{tag_name}.json"
    if not tf.exists():
        return None
    return json.loads(tf.read_text())


BLOBS_DIR = SCRAPER / "blobs"


def load_blob_page(blob_url: str) -> dict | None:
    safe = blob_url.strip("/").replace("/", "__").replace(":", "_")
    bf = BLOBS_DIR / f"{safe}.json"
    if not bf.exists():
        return None
    return json.loads(bf.read_text())


def _blob_href(blob_url: str) -> str:
    """Return local blob page URL if blob data exists, else external ollama.com URL."""
    if blob_url and load_blob_page(blob_url):
        return url(blob_url.replace(":", "/:", 1) + "/")
    return "https://ollama.com" + blob_url


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
    <meta property="og:image" content="https://ollama.com/public/og.png" />
    <meta property="og:image:type" content="image/png" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="628" />
    <meta property="og:type" content="website" />
    <meta name="robots" content="index, follow" />
    <meta property="twitter:card" content="summary" />
    <meta property="twitter:title" content="{esc(title)}" />
    <meta property="twitter:description" content="{desc}" />
    <meta property="twitter:site" content="ollama" />
    <meta property="twitter:image:src" content="https://ollama.com/public/og-twitter.png" />
    <meta property="twitter:image:width" content="1200" />
    <meta property="twitter:image:height" content="628" />
    <link rel="icon" type="image/png" sizes="16x16" href="{url("/assets/icon-16x16.png")}" />
    <link rel="icon" type="image/png" sizes="32x32" href="{url("/assets/icon-32x32.png")}" />
    <link rel="icon" type="image/png" sizes="48x48" href="{url("/assets/icon-48x48.png")}" />
    <link rel="icon" type="image/png" sizes="64x64" href="{url("/assets/icon-64x64.png")}" />
    <link rel="apple-touch-icon" sizes="180x" href="{url("/assets/apple-touch-icon.png")}" />
    <link rel="icon" type="image/png" sizes="192x192" href="{url("/assets/android-chrome-icon-192x192.png")}" />
    <link rel="icon" type="image/png" sizes="512x512" href="{url("/assets/android-chrome-icon-512x512.png")}" />
    {theme_script_head()}
    <link href="{url("/assets/tailwind.css")}" rel="stylesheet" />
    <link href="{url("/assets/prism.css")}" rel="stylesheet" />
    <link href="{url("/assets/extras.css")}" rel="stylesheet" />
    <script type="application/ld+json">
      {{
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": "Ollama",
        "url": "https://ollama.com"
      }}
    </script>
    <script defer src="{url("/assets/htmx.bundle.js")}"></script>"""


def nav_html(active: str = "") -> str:
    del active
    return f"""<header class="sticky top-0 z-40 bg-white dark:bg-neutral-950 underline-offset-4 lg:static">
  <nav class="flex w-full items-center justify-between px-6 py-[9px]">
    <a href="{url("/")}" class="z-50">
      <img src="{url("/assets/ollama.png")}" class="w-8 dark:invert" alt="Ollama" />
    </a>
    <div class="hidden lg:flex xl:flex-1 items-center space-x-6 ml-6 mr-6 xl:mr-0 text-lg">
      <a class="hover:underline focus:underline focus:outline-none focus:ring-0" href="{url("/")}">Models</a>
      <a class="hover:underline focus:underline focus:outline-none focus:ring-0" href="https://docs.ollama.com">Docs</a>
      <a class="hover:underline focus:underline focus:outline-none focus:ring-0" href="https://ollama.com/pricing">Pricing</a>
    </div>
    <div class="flex-grow justify-center items-center hidden lg:flex">
      <div class="relative w-full xl:max-w-[28rem]">
        <form action="{url("")}" autocomplete="off" id="nav-search-form">
          <div class="relative flex w-full appearance-none bg-black/5 dark:bg-white/5 border border-neutral-100 dark:border-neutral-800 items-center rounded-full">
            <span class="pl-2 text-2xl text-neutral-500 dark:text-neutral-400">{SVG_SEARCH}</span>
            <input id="navbar-input" name="q" type="text" class="resize-none rounded-full border-0 py-2.5 bg-transparent text-sm w-full placeholder:text-neutral-500 dark:placeholder:text-neutral-500 focus:outline-none focus:ring-0 dark:text-neutral-200" placeholder="Search models" autocomplete="off" hx-on:keydown="if(event.key==='Enter'){{event.preventDefault();window.location.href='{url("/?q=")}'+encodeURIComponent(this.value);}}" />
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
    return f"""<footer class="mt-auto">
  <div class="underline-offset-4 hidden md:block">
    <div class="flex items-center justify-between px-6 py-3.5">
      <div class="text-xs text-neutral-500 dark:text-neutral-400">&copy; 2026 Ollama · <a href="{url("/maternion/")}" class="hover:underline">Maternion</a></div>
      <div class="flex space-x-6 text-xs text-neutral-500 dark:text-neutral-400">
        <a href="https://ollama.com/download" class="hover:underline">Download</a>
        <a href="https://ollama.com/blog" class="hover:underline">Blog</a>
        <a href="https://docs.ollama.com" class="hover:underline">Docs</a>
        <a href="https://github.com/ollama/ollama" class="hover:underline">GitHub</a>
        <a href="https://ollama.com/pricing" class="hover:underline">Pricing</a>
      </div>
    </div>
  </div>
  <div class="py-4 md:hidden">
    <ul class="flex flex-col items-center space-y-2 text-xs text-neutral-500 dark:text-neutral-400">
      <li><a href="https://ollama.com/download" class="hover:underline">Download</a></li>
      <li><a href="https://ollama.com/blog" class="hover:underline">Blog</a></li>
      <li><a href="https://docs.ollama.com" class="hover:underline">Docs</a></li>
      <li><a href="https://github.com/ollama/ollama" class="hover:underline">GitHub</a></li>
      <li><a href="https://ollama.com/pricing" class="hover:underline">Pricing</a></li>
    </ul>
    <div class="mt-4 text-center text-xs text-neutral-500 dark:text-neutral-400">&copy; 2026 Ollama · <a href="{url("/maternion/")}" class="hover:underline">Maternion</a></div>
  </div>
</footer>"""


def theme_script_head() -> str:
    """Inline script for <head> — sets dark class BEFORE CSS loads to prevent FOUC."""
    return """<script>
(function() {
  var stored = localStorage.getItem('theme');
  var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (stored === 'dark' || (!stored && prefersDark)) {
    document.documentElement.classList.add('dark');
  }
})();
</script>"""


def theme_script() -> str:
    """Toggle handler — placed at end of body (only needs DOMContentLoaded)."""
    return """<script>
(function() {
  function toggle() {
    var isDark = document.documentElement.classList.toggle('dark');
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
            f'<span x-test-capability class="inline-flex items-center rounded-md '
            f"bg-indigo-50 dark:bg-indigo-950/50 px-2 py-0.5 text-xs font-medium "
            f'text-indigo-600 dark:text-indigo-400 sm:text-[13px]">{esc(cap)}</span>'
        )
    if cloud:
        parts.append(
            '<span class="inline-flex items-center rounded-md bg-cyan-50 '
            "dark:bg-cyan-950/50 px-2 py-0.5 text-xs font-medium text-cyan-500 "
            'dark:text-cyan-400 sm:text-[13px]">cloud</span>'
        )
    return "\n        ".join(parts) if parts else ""


def size_spans(sizes: list[str]) -> str:
    parts = []
    for s in sizes:
        parts.append(
            f'<span x-test-size class="inline-flex items-center rounded-md '
            f"bg-[#ddf4ff] dark:bg-blue-950/50 px-2 py-0.5 text-xs font-medium "
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
        f'data-name="{esc(name_raw).lower()}" '
        f'data-cloud="{str(m.get("cloud", False)).lower()}" '
        f'data-cloud-only="{str(m.get("cloud_only", False)).lower()}"'
    )

    # MLX pill for models that have MLX variants (black bg, white text, same size as other pills)
    fmt_chip = ""
    if tags and has_mlx(tags):
        fmt_chip = (
            '<span class="inline-flex items-center rounded-md '
            "bg-neutral-900 px-2 py-0.5 text-xs font-medium text-white "
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
    sorted_models = sorted(
        models,
        key=lambda m: ranks.get(m["name"], {}).get("popular_rank", 9999),
    )
    cards = "\n".join(
        render_card(m, load_tags(m["path"]), ranks) for m in sorted_models
    )

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

    (PUBLIC).mkdir(parents=True, exist_ok=True)
    (PUBLIC / "index.html").write_text(page)


# --------------------------------------------------------------------------- #
# Model detail page
# --------------------------------------------------------------------------- #


def _fmt_pills(prefix: str, all_count: int, gguf_count: int, mlx_count: int) -> str:
    """GGUF/MLX pill radio filters. prefix is 'models' or 'tags'."""
    return f"""<div class="flex flex-wrap gap-2 mb-4">
  <div class="relative inline-block">
    <input type="radio" name="fmt" value="all" id="fmt-all" class="peer sr-only fmt-radio" data-fmt="all" checked>
    <label for="fmt-all" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 dark:border-neutral-800 text-neutral-800 dark:text-neutral-300 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">All ({all_count})</label>
  </div>
  <div class="relative inline-block">
    <input type="radio" name="fmt" value="gguf" id="fmt-gguf" class="peer sr-only fmt-radio" data-fmt="gguf">
    <label for="fmt-gguf" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 dark:border-neutral-800 text-neutral-800 dark:text-neutral-300 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">GGUF ({gguf_count})</label>
  </div>
  <div class="relative inline-block">
    <input type="radio" name="fmt" value="mlx" id="fmt-mlx" class="peer sr-only fmt-radio" data-fmt="mlx">
    <label for="fmt-mlx" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 dark:border-neutral-800 text-neutral-800 dark:text-neutral-300 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">MLX ({mlx_count})</label>
  </div>
</div>"""


def _detail_tag_rows(
    tags_subset: list[dict],
    model_path: str,
    latest_digest: str = "",
    show_mlx_badge: bool = False,
) -> str:
    """Render tag rows (mobile + desktop) for the detail page Models table."""
    rows = []
    model_name = model_path.strip("/").split("/")[-1]
    for t in tags_subset:
        tag_name = esc(t["name"])
        full_tag_name = f"{model_name}:{t['name']}"
        full_tag_esc = esc(full_tag_name)
        size = esc(t.get("size_text") or "") or "—"
        ctx = esc(t.get("context") or "") or "—"
        inp = esc(t.get("input_type") or "") or "—"
        updated = esc(t.get("updated") or "") or "—"
        tag_link = url(esc(model_path) + "/:" + esc(t["name"]) + "/")
        raw_digest = t.get("digest") or ""
        show_latest = (
            bool(latest_digest)
            and raw_digest == latest_digest
            and t["name"] != "latest"
        )
        latest_badge = (
            '<span class="ml-2 inline-flex items-center rounded-full px-2 py-px text-xs font-medium border border-blue-500 text-blue-600 dark:text-blue-400 dark:border-blue-500">latest</span>'
            if show_latest
            else ""
        )
        mlx_badge = (
            '<span class="ml-2 inline-flex items-center rounded-full px-2 py-px text-xs font-medium border border-neutral-600 text-neutral-600 dark:border-neutral-400 dark:text-neutral-400">MLX</span>'
            if show_mlx_badge and t.get("format") == "mlx"
            else ""
        )
        usage_level = (t.get("usage_level") or "").strip()
        active_slots = int(t.get("usage_active_slots") or 0)
        if usage_level or active_slots > 0:
            active_bars = "".join(
                '<span x-test-model-tag-usage-slot-active class="block h-1 w-4 rounded-full bg-neutral-800 dark:bg-neutral-200"></span>'
                for _ in range(active_slots)
            )
            inactive_bars = "".join(
                '<span x-test-model-tag-usage-slot-inactive class="block h-1 w-4 rounded-full bg-neutral-200 dark:bg-neutral-700"></span>'
                for _ in range(4 - active_slots)
            )
            size_cell = (
                f'<p x-test-model-tag-cost class="col-span-2 flex items-center gap-0.5 text-neutral-500 dark:text-neutral-400">'
                f"{active_bars}{inactive_bars}"
                f"</p>"
            )
            size_inline = ""
        else:
            size_cell = f'<p x-test-model-tag-size class="col-span-2 text-neutral-500 dark:text-neutral-400">{size}</p>'
            size_inline = size
        inline_text = f"{size_inline} · " if size_inline else ""
        usage_text = ""
        if (usage_level or active_slots > 0) and usage_level:
            usage_text = f"{usage_level.capitalize()} Usage · "
        rows.append(
            f'      <a href="{tag_link}" class="sm:hidden flex flex-col space-y-[6px] group text-[13px] px-4 py-3">\n'
            f'        <span class="flex items-center">\n'
            f'          <p class="block group-hover:underline text-sm font-medium text-neutral-800 dark:text-neutral-200">{full_tag_esc}</p>\n'
            f"          {latest_badge}\n"
            f"          {mlx_badge}\n"
            f"        </span>\n"
            f'        <p class="flex text-neutral-500 dark:text-neutral-400">{usage_text}{inline_text}{ctx} context window · {inp} · {updated}</p>\n'
            f"      </a>\n"
            f'      <div class="hidden group px-4 py-3 sm:grid sm:grid-cols-12 text-[13px]">\n'
            f'        <span class="col-span-6 flex items-center">\n'
            f'          <a href="{tag_link}" class="block group-hover:underline text-sm font-medium text-neutral-800 dark:text-neutral-200">{full_tag_esc}</a>\n'
            f"          {latest_badge}\n"
            f"          {mlx_badge}\n"
            f'          <input class="command hidden" value="{full_tag_esc}" />\n'
            f'          <button class="hidden group-hover:inline-flex ml-1.5 text-neutral-500 hover:text-black dark:hover:text-white items-center" onclick="copyToClipboard(this); event.preventDefault(); event.stopPropagation();">\n'
            f"            {SVG_COPY}\n"
            f"          </button>\n"
            f"        </span>\n"
            f"        {size_cell}\n"
            f'        <p class="col-span-2 text-neutral-500 dark:text-neutral-400">{ctx}</p>\n'
            f'        <p class="col-span-2 text-neutral-500 dark:text-neutral-400">{inp}</p>\n'
            f"      </div>"
        )
    return "\n".join(rows)


def _main_tags(m: dict, tags: list[dict]) -> list[dict]:
    """Return the curated subset of tags shown on the model main page,
    matching ollama.com's filtering: latest, each size's base + MLX tag,
    cloud, and cloud-only size tags."""
    by_name = {t["name"]: t for t in tags}
    sizes = m.get("sizes", [])
    ordered: list[str] = ["latest"]
    # Base size tags that exist
    ordered += [s for s in sizes if s in by_name]
    # MLX counterparts that exist
    ordered += [f"{s}-mlx" for s in sizes if f"{s}-mlx" in by_name]
    # Generic cloud tag
    if "cloud" in by_name:
        ordered.append("cloud")
    # Cloud-only size tags: X-cloud where X is not a downloadable size
    for n in sorted(by_name.keys()):
        if n.endswith("-cloud") and n != "cloud" and n[:-6] not in by_name:
            ordered.append(n)
    # Dedupe preserving order, drop missing
    seen = set()
    out = []
    for n in ordered:
        if n in by_name and n not in seen:
            seen.add(n)
            out.append(by_name[n])
    return out


def _detail_models_section(m: dict, tags: list[dict]) -> str:
    """Models section (tag table) for the detail page, with pill filters + fmt tables."""
    has_m = has_mlx(tags)
    gguf_tags = [t for t in tags if t["format"] == "gguf"]
    mlx_tags = [t for t in tags if t["format"] == "mlx"]
    count = len(tags)

    pills = (
        _fmt_pills("models", len(tags), len(gguf_tags), len(mlx_tags)) if has_m else ""
    )

    def table_block(rows_html: str, n: int, fmt_id: str, visible: bool) -> str:
        hidden = "" if visible else " hidden"
        mobile_label = n if visible else count
        count_label = "1 model" if mobile_label == 1 else f"{mobile_label} models"
        return (
            f'<div id="models-table-{fmt_id}" class="fmt-table{hidden}">\n'
            f'  <div class="overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800">\n'
            f'    <div class="min-w-full divide-y divide-neutral-200 dark:divide-neutral-800">\n'
            f'      <div class="items-center grid bg-neutral-50 dark:bg-neutral-900 px-4 py-3 text-xs grid-cols-12 text-neutral-900 dark:text-neutral-100">\n'
            f'        <p class="hidden sm:block col-span-6">Name</p>\n'
            f'        <p class="sm:hidden col-span-6">{count_label}</p>\n'
            f'        <p class="col-span-2 hidden sm:block">Size / Usage</p>\n'
            f'        <p class="col-span-2 hidden sm:block">Context</p>\n'
            f'        <p class="col-span-2 hidden sm:block">Input</p>\n'
            f"      </div>\n"
            f"      {rows_html}\n"
            f"    </div>\n"
            f"  </div>\n"
            f"</div>"
        )

    # Find latest tag's digest
    latest_digest = ""
    for t in tags:
        if t["name"] == "latest":
            latest_digest = t.get("digest", "")
            break

    main = _main_tags(m, tags)
    main_gguf = [t for t in main if t["format"] == "gguf"]
    main_mlx = [t for t in main if t["format"] == "mlx"]
    rows_all = _detail_tag_rows(main, m["path"], latest_digest, show_mlx_badge=True)
    rows_gguf = _detail_tag_rows(
        main_gguf, m["path"], latest_digest, show_mlx_badge=False
    )
    rows_mlx = _detail_tag_rows(
        main_mlx, m["path"], latest_digest, show_mlx_badge=False
    )

    view_all = f'<a href="{url(esc(m["path"]) + "/tags/")}" class="text-sm text-neutral-500 dark:text-neutral-400 cursor-pointer underline focus:outline-none">View all {len(tags)} &#8594;</a>'

    blocks = [table_block(rows_all, count, "all", True)]
    if has_m:
        blocks.append(table_block(rows_gguf, len(gguf_tags), "gguf", False))
        blocks.append(table_block(rows_mlx, len(mlx_tags), "mlx", False))
    tables = "\n".join(blocks)

    return f"""<section class="flex flex-1 flex-col">
  <div class="flex items-center justify-between mb-4">
    <h2 class="text-base font-semibold leading-6 text-neutral-900 dark:text-neutral-100">Models</h2>
    {view_all}
  </div>
  {pills}
  {tables}
</section>"""


def _usage_section(full_name: str) -> str:
    fn = esc(full_name)
    return f"""<section data-usage-section class="mb-8">
  <div class="relative rounded-lg border border-neutral-200 dark:border-neutral-800 overflow-hidden bg-white dark:bg-neutral-900">
    <div class="flex items-center justify-between bg-white dark:bg-neutral-900 pt-1 pl-[7px] pr-3">
      <div class="flex">
        <button type="button" class="use-tab px-3 py-2 text-xs font-medium text-neutral-900 dark:text-neutral-100 underline decoration-1 underline-offset-[7px]" data-tab="cli" onclick="switchUsageTab(this, 'cli')">CLI</button>
        <button type="button" class="use-tab px-3 py-2 text-xs text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300" data-tab="api" onclick="switchUsageTab(this, 'api')">cURL</button>
        <button type="button" class="use-tab px-3 py-2 text-xs text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300" data-tab="python" onclick="switchUsageTab(this, 'python')">Python</button>
        <button type="button" class="use-tab px-3 py-2 text-xs text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300" data-tab="javascript" onclick="switchUsageTab(this, 'javascript')">JavaScript</button>
      </div>
      <a href="https://github.com/ollama/ollama-python" target="_blank" rel="noopener noreferrer" class="use-link hidden py-2 text-xs text-neutral-500 hover:text-neutral-700 dark:text-neutral-400 dark:hover:text-neutral-300 inline-flex items-center gap-1" data-link="python"><span class="hidden sm:inline">Documentation</span> {SVG_EXTERNAL}</a>
      <a href="https://github.com/ollama/ollama-js" target="_blank" rel="noopener noreferrer" class="use-link hidden py-2 text-xs text-neutral-500 hover:text-neutral-700 dark:text-neutral-400 dark:hover:text-neutral-300 inline-flex items-center gap-1" data-link="javascript"><span class="hidden sm:inline">Documentation</span> {SVG_EXTERNAL}</a>
    </div>
    <div class="relative">
      <div class="absolute bottom-[10.5px] right-[10.5px] flex items-center gap-2 z-10">
        <button type="button" class="use-copy-btn p-1.5 text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300 rounded" onclick="copyUsageCode(this)" title="Copy">
          {SVG_COPY}
        </button>
      </div>
      <div class="use-panel p-4 font-mono text-[13px] text-neutral-700 dark:text-neutral-300" data-panel="cli">
        <pre class="m-0 whitespace-pre-wrap">ollama run {fn}</pre>
      </div>
      <div class="use-panel hidden p-4 font-mono text-[13px] text-neutral-700 dark:text-neutral-300" data-panel="api">
        <pre class="m-0 whitespace-pre-wrap">curl http://localhost:11434/api/chat \\
  -d '{{
    "model": "{fn}",
    "messages": [{{"role": "user", "content": "Hello!"}}]
  }}'</pre>
      </div>
      <div class="use-panel hidden p-4 font-mono text-[13px] text-neutral-700 dark:text-neutral-300" data-panel="python">
        <pre class="m-0 whitespace-pre-wrap"><span class="text-neutral-500">from</span> ollama <span class="text-neutral-500">import</span> chat

response = chat(
    model=<span class="text-green-700">'{fn}'</span>,
    messages=[{{<span class="text-green-700">'role'</span>: <span class="text-green-700">'user'</span>, <span class="text-green-700">'content'</span>: <span class="text-green-700">'Hello!'</span>}}],
)
<span class="text-neutral-500">print</span>(response.message.content)</pre>
      </div>
      <div class="use-panel hidden p-4 font-mono text-[13px] text-neutral-700 dark:text-neutral-300" data-panel="javascript">
        <pre class="m-0 whitespace-pre-wrap"><span class="text-neutral-500">import</span> ollama <span class="text-neutral-500">from</span> <span class="text-green-700">'ollama'</span>

<span class="text-neutral-500">const</span> response = <span class="text-neutral-500">await</span> ollama.chat({{
  model: <span class="text-green-700">'{fn}'</span>,
  messages: [{{role: <span class="text-green-700">'user'</span>, content: <span class="text-green-700">'Hello!'</span>}}],
}})
console.log(response.message.content)</pre>
      </div>
    </div>
  </div>
</section>"""


def _file_row_html(entry: dict) -> str:
    type_ = esc(entry.get("type") or entry.get("name") or "")
    blob_url_raw = entry.get("url") or entry.get("blob_url") or ""
    blob_url = esc(blob_url_raw)
    size = esc(entry.get("size") or "")
    is_model = (entry.get("type") or "").lower() == "model"
    if is_model:
        arch = esc(entry.get("arch") or entry.get("architecture") or "—")
        parameters = esc(entry.get("parameters") or "—")
        quant = entry.get("quantization")
        quant_div = ""
        if quant:
            quant_div = (
                "<div>·</div>"
                '<div class="flex sm:space-x-2 items-center">'
                '<span class="hidden sm:block">quantization</span>'
                '<span class="text-neutral-400 dark:text-neutral-500 sm:font-semibold sm:text-neutral-800 dark:sm:text-neutral-200 sm:text-xs">'
                + esc(quant)
                + "</span>"
                "</div>"
            )
        content_html = (
            '<div class="space-x-2 flex text-sm">'
            '<div class="flex sm:space-x-2 items-center"><span class="hidden sm:block">arch</span>'
            '<span class="text-neutral-400 dark:text-neutral-500 sm:font-semibold sm:text-neutral-800 dark:sm:text-neutral-200 sm:text-xs">'
            + arch
            + "</span></div>"
            "<div>·</div>"
            '<div class="flex sm:space-x-2 items-center"><span class="hidden sm:block">parameters</span>'
            '<span class="text-neutral-400 dark:text-neutral-500 sm:font-semibold sm:text-neutral-800 dark:sm:text-neutral-200 sm:text-xs">'
            + parameters
            + "</span></div>"
            + quant_div
            + "</div>"
        )
    else:
        preview = (
            entry.get("content_preview")
            or entry.get("content")
            or entry.get("details")
            or ""
        )
        content_html = esc(preview)
    return (
        '<div class="group block grid-cols-12 gap-2 px-4 py-3 sm:grid sm:grid-cols-12">'
        '<div class="truncate text-sm font-medium text-neutral-800 dark:text-neutral-200 group-hover:underline sm:col-span-2 sm:col-start-1">'
        f'<a href="{_blob_href(blob_url_raw)}" class="group-hover:underline">{type_}</a>'
        "</div>"
        '<div class="truncate font-mono text-[13px] text-neutral-400 dark:text-neutral-500 subpixel-antialiased sm:col-span-8 sm:col-start-3">'
        f"{content_html}"
        "</div>"
        f'<div class="hidden text-right text-xs text-neutral-400 dark:text-neutral-500 sm:col-start-12 sm:block">{size}</div>'
        "</div>"
    )


def _details_section(page_data: dict) -> str:
    files = page_data.get("files") or page_data.get("blobs") or []
    manifest_updated = esc(
        page_data.get("manifest_updated") or page_data.get("updated") or ""
    )
    manifest_digest = esc(
        page_data.get("manifest_digest") or page_data.get("digest") or ""
    )
    manifest_size = esc(page_data.get("manifest_size") or "")
    file_rows = "\n".join(_file_row_html(f) for f in files)
    return f"""<section id="file-explorer" class="flex flex-1 flex-col">
  <h2 class="text-base font-semibold leading-6 text-neutral-900 dark:text-neutral-100 mb-4">Details</h2>
  <div class="overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800">
    <div class="min-w-full divide-y divide-neutral-200 dark:divide-neutral-800">
      <div class="flex items-center justify-between bg-neutral-50 dark:bg-neutral-900 px-4 py-3 text-xs text-neutral-900 dark:text-neutral-100">
        <p class="hidden sm:block">Updated {manifest_updated}</p>
        <p class="flex items-center sm:hidden">{SVG_CLOCK}{manifest_updated}</p>
        <p>{manifest_digest} · {manifest_size} ·</p>
      </div>
      {file_rows}
    </div>
  </div>
</section>"""


def _applications_section(page_data: dict) -> str:
    apps = page_data.get("applications") or []
    if not apps:
        return ""
    rows = []
    for a in apps:
        name = esc(a.get("name", ""))
        icon = esc(a.get("icon_url", ""))
        cmd = esc(a.get("command", ""))
        icon_full = icon if icon.startswith("http") else f"https://ollama.com{icon}"
        rows.append(
            f'      <div class="group flex items-center justify-between px-4 py-3">\n'
            f'        <div class="flex items-center gap-3">\n'
            f'          <img src="{icon_full}" class="w-8 h-8" alt="{name}" />\n'
            f'          <div class="flex flex-col">\n'
            f'            <span class="text-sm font-medium text-neutral-800 dark:text-neutral-200">{name}</span>\n'
            f'            <code class="text-[13px] text-neutral-500 dark:text-neutral-400 font-mono">{cmd}</code>\n'
            f"          </div>\n"
            f"        </div>\n"
            f'        <input class="command hidden" value="{cmd}" />\n'
            f'        <button class="p-1.5 text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300 rounded" onclick="copyToClipboard(this); event.preventDefault(); event.stopPropagation();" title="Copy">\n'
            f"          {SVG_COPY}\n"
            f"        </button>\n"
            f"      </div>"
        )
    body = "\n".join(rows)
    return f"""
<section class="flex flex-1 flex-col mb-8" id="external-tools-section">
  <h2 class="text-base font-semibold leading-6 text-neutral-900 dark:text-neutral-100 mb-4">Applications</h2>
  <div class="overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800">
    <div class="min-w-full divide-y divide-neutral-200 dark:divide-neutral-800">
{body}
    </div>
  </div>
</section>"""


def _readme_section(page_data: dict) -> str:
    readme = page_data.get("readme_html") or page_data.get("readme") or ""
    if not readme or readme.strip().lower() == "no readme":
        body = '<span class="text-neutral-400 dark:text-neutral-600">No readme</span>'
    else:
        # Rewrite ollama.com-relative asset URLs to absolute URLs so readme
        # images/files load. Handles src= and href= in both quote styles.
        readme = re.sub(
            r'((?:src|href)=)(["\'])/assets/',
            r"\1\2https://ollama.com/assets/",
            readme,
        )
        body = readme
    # Class string mirrors ollama.com's <div id="display"> exactly (the long
    # Tailwind prose-* variant chain), with dark: variants appended since
    # ollama.com itself has no dark mode. The corresponding prose-* and
    # dark:prose-* CSS lives in EXTRAS_CSS (the vendored tailwind.css covers
    # most prose-* variants but is missing a few plus all dark: ones).
    prose_cls = (
        "prose-td code:display-inline-block prose-td code:bg-gray-200 prose-td code:px-2 "
        "prose-td code:py-1 prose-td code:rounded-md prose prose-headings:mb-[0.7em] "
        "prose-headings:mt-[1.25em] prose-headings:font-semibold prose-headings:tracking-tight "
        "prose-h1:text-[32px] prose-h2:text-2xl prose-h3:text-xl prose-h4:text-lg prose-h5:text-base "
        "prose-p:mb-4 prose-p:mt-0 prose-p:leading-relaxed prose-p:before:hidden prose-p:after:hidden "
        "prose-blockquote:font-normal prose-blockquote:not-italic prose-blockquote:text-neutral-500 "
        "prose-blockquote:before:hidden prose-blockquote:after:hidden prose-code:my-0 prose-code:inline-block "
        "prose-code:rounded-md prose-code:bg-neutral-100 prose-code:px-2 prose-code:text-[85%] "
        "prose-code:font-normal prose-code:leading-relaxed prose-code:text-black prose-code:before:hidden "
        "prose-code:after:hidden prose-pre:mb-4 prose-pre:mt-0 prose-pre:whitespace-pre-wrap "
        "prose-pre:rounded-lg prose-pre:bg-neutral-100 prose-pre:px-3 prose-pre:py-3 prose-pre:text-base "
        "prose-pre:text-black prose-ol:mb-4 prose-ol:mt-1 prose-ol:pl-8 marker:prose-ol:text-black "
        "prose-ul:mb-4 prose-ul:mt-1 prose-ul:pl-8 marker:prose-ul:text-black prose-li:mb-0 "
        "prose-li:mt-0.5 prose-li:text-black first:prose-li:mt-0 prose-table:w-full prose-table:table-auto "
        "prose-table:border-collapse prose-th:break-words prose-th:text-center prose-th:font-semibold "
        "prose-td:break-words prose-td:px-4 prose-td:py-2 prose-td:text-left prose-img:mx-auto "
        "prose-img:my-12 prose-video:my-12 max-w-none overflow-auto py-5 text-black "
        # dark mode (not present on ollama.com — added for this site)
        "dark:prose-headings:text-neutral-200 dark:prose-blockquote:text-neutral-400 "
        "dark:prose-code:bg-neutral-800 dark:prose-code:text-neutral-200 dark:prose-pre:bg-neutral-900 "
        "dark:prose-pre:text-neutral-200 dark:prose-li:text-neutral-200 dark:prose-td:text-neutral-300 "
        "dark:prose-th:text-neutral-200 dark:prose-a:text-blue-400 dark:prose-strong:text-neutral-200 "
        "dark:marker:prose-ol:text-neutral-400 dark:marker:prose-ul:text-neutral-400 "
        "dark:text-neutral-200"
    )
    return f"""<div class="flex flex-1 flex-col py-8" id="readme">
  <div class="flex items-center justify-between pb-1">
    <h2 class="text-base font-semibold leading-6 text-neutral-900 dark:text-neutral-100">Readme</h2>
  </div>
  <div>
    <div id="display" class="{prose_cls}">
      {body}
    </div>
  </div>
</div>"""


def _header_section(m: dict) -> str:
    """Section 1: model name + stats + summary + badges.

    For user (non-official) models, ollama.com renders a namespace link followed
    by a "/" separator before the model name (e.g. "maternion / LightOnOCR-2").
    """
    name = esc(m["name"])
    desc = esc(m["description"])
    pulls = format_count(m["pulls"])
    updated = esc(m["updated"])
    updated_title = esc(m.get("updated_title") or "")
    model_link = url(esc(m["path"]))
    caps = capability_spans(m["capabilities"], m["cloud"])
    sizes = size_spans(m["sizes"])

    # For user models, prepend the namespace link + "/" separator.
    namespace_html = ""
    if not m.get("official") and "/" in m["path"].strip("/"):
        namespace = m["path"].strip("/").split("/")[0]
        namespace_esc = esc(namespace)
        namespace_link = url("/" + namespace_esc)
        namespace_html = (
            f'<a x-test-model-namespace class="text-xl sm:text-[28px] font-medium leading-normal decoration-1 underline-offset-4 hover:underline shrink-0" href="{namespace_link}">{namespace_esc}</a>'
            f'<span class="text-xl sm:text-[28px] font-medium px-1 shrink-0">/</span>'
        )

    return f"""<div class="flex flex-col space-y-3">
    <div class="flex items-center min-w-0">
      <div class="flex items-center min-w-0 space-x-2">
        <div class="flex items-center min-w-0">
          {namespace_html}<span class="text-xl tracking-tight sm:text-[28px] min-w-0 truncate font-medium leading-normal text-black dark:text-neutral-100 decoration-2">
            <a x-test-model-name href="{model_link}" title="{name}" class="underline-offset-[5px] hover:underline">{name}</a>
           </span>
         </div>
       </div>
     </div>
     <div class="flex flex-col space-y-2">
       <div class="flex flex-col space-y-2">
         <p class="flex space-x-5 text-[13px] font-medium text-neutral-500 dark:text-neutral-400">
           <span class="flex items-center">
             {SVG_DOWNLOAD}
             <span x-test-pull-count>{pulls}</span>
             <span class="hidden sm:flex">&nbsp;Downloads</span>
          </span>
          <span class="flex items-center" title="{updated_title}">
            {SVG_CLOCK}
            <span class="hidden sm:flex">Updated&nbsp;</span>
            <span x-test-updated>{updated}</span>
          </span>
        </p>
      </div>
      <h2 class="break-words text-neutral-800 dark:text-neutral-300">{desc}</h2>
      <div class="flex flex-wrap gap-2">
        {caps}
        {sizes}
      </div>
    </div>
  </div>"""


def _cloud_metrics_section(page_data: dict) -> str:
    """Render the cloud metrics (Usage/Context/Size) section for cloud models.

    Returns empty string if page_data has no cloud metrics (no usage_level and
    no context).
    """
    usage_level = (page_data.get("cloud_usage_level") or "").strip()
    active_slots = int(page_data.get("cloud_usage_active_slots") or 0)
    ctx = esc(page_data.get("cloud_context") or "")
    ctx_unit = esc(page_data.get("cloud_context_unit") or "")
    size = esc(page_data.get("cloud_size") or "")
    size_unit = esc(page_data.get("cloud_size_unit") or "")
    if not usage_level and not ctx:
        return ""
    active_bars = "".join(
        '<span x-test-model-cost-slot-active class="block h-1.5 w-5 rounded-full bg-neutral-900 dark:bg-neutral-100"></span>'
        for _ in range(active_slots)
    )
    inactive_bars = "".join(
        '<span x-test-model-cost-slot-inactive class="block h-1.5 w-5 rounded-full bg-neutral-200 dark:bg-neutral-700"></span>'
        for _ in range(4 - active_slots)
    )
    return f"""<div x-test-model-metrics class="!mt-8 grid grid-cols-3 overflow-hidden rounded-lg border border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
  <div x-test-model-cost x-test-model-metric="usage" class="min-h-24 min-w-0 border-neutral-200 dark:border-neutral-800 px-4 py-3 md:px-5 md:py-4 border-r">
    <div class="text-[13px] font-medium text-neutral-500">Usage</div>
    <div class="mt-3 flex min-w-0 flex-col gap-1">
      <div x-test-model-cost-value x-test-model-cost-level class="flex h-5 items-center gap-1">
        {active_bars}{inactive_bars}
      </div>
      <span class="min-w-0 break-words text-xs leading-tight text-neutral-700 dark:text-neutral-300 sm:text-[14px] sm:leading-5">{esc(usage_level)}</span>
    </div>
  </div>
  <div x-test-model-metric="context" class="min-h-24 min-w-0 border-neutral-200 dark:border-neutral-800 px-4 py-3 md:px-5 md:py-4 border-r">
    <div class="text-[13px] font-medium text-neutral-500">Context</div>
    <div class="mt-3 flex min-w-0 flex-col gap-1">
      <span class="shrink-0 text-xl font-medium leading-none text-black dark:text-neutral-100">{ctx}</span>
      <span class="min-w-0 break-words text-[13px] leading-tight text-neutral-700 dark:text-neutral-300 sm:text-sm">{ctx_unit}</span>
    </div>
  </div>
  <div x-test-model-metric="size" class="min-h-24 min-w-0 border-neutral-200 dark:border-neutral-800 px-4 py-3 md:px-5 md:py-4">
    <div class="text-[13px] font-medium text-neutral-500">Size</div>
    <div class="mt-3 flex min-w-0 flex-col gap-1">
      <span class="shrink-0 text-xl font-medium leading-none text-black dark:text-neutral-100">{size}</span>
      <span class="min-w-0 break-words text-[13px] leading-tight text-neutral-700 dark:text-neutral-300 sm:text-sm">{size_unit}</span>
    </div>
  </div>
</div>"""


def build_detail(m: dict, tags: list[dict]) -> None:
    name = m["name"]
    desc = m["description"]
    path = m["path"]
    slug_dir = PUBLIC / path.strip("/")
    slug_dir.mkdir(parents=True, exist_ok=True)

    full_name = m["name"]
    # For cloud-only models, use the :cloud tag in CLI commands
    if m.get("cloud_only"):
        full_name = f"{m['name']}:cloud"
    header = _header_section(m)
    usage = _usage_section(full_name)
    models_section = _detail_models_section(m, tags)

    page_data = load_model_page(m["path"])
    readme_section = _readme_section(page_data) if page_data else ""
    cloud_metrics = _cloud_metrics_section(page_data) if page_data else ""
    apps_section = _applications_section(page_data) if page_data else ""

    # Title: official models use just name, user models use owner/name
    title = name if m.get("official") else path.strip("/")

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html(title, desc)}
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html()}

<main class="mx-auto flex w-full max-w-[52rem] flex-col px-6 py-10 md:py-24 lg:px-8">
  {header}
  {cloud_metrics}
  <div class="py-8">
    {usage}
    {apps_section}
    {models_section}
  </div>
  {readme_section}
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


def _tags_tag_row(
    t: dict, model_path: str, latest_digest: str = "", show_mlx_badge: bool = False
) -> str:
    model_name = model_path.strip("/").split("/")[-1]
    full_tag_name = f"{model_name}:{t['name']}"
    full_tag_esc = esc(full_tag_name)
    tag_link = url(esc(model_path) + "/:" + esc(t["name"]) + "/")
    size = esc(t.get("size_text") or "") or "—"
    ctx = esc(t.get("context") or "") or "—"
    inp = esc(t.get("input_type") or "") or "—"
    digest = esc(t.get("digest") or "") or ""
    updated = esc(t.get("updated") or "") or ""
    raw_digest = t.get("digest") or ""
    show_latest = (
        bool(latest_digest) and raw_digest == latest_digest and t["name"] != "latest"
    )
    latest_badge = (
        '<span class="ml-2 inline-flex items-center rounded-full px-2 py-px text-xs font-medium border border-blue-500 text-blue-600 dark:text-blue-400 dark:border-blue-500">latest</span>'
        if show_latest
        else ""
    )
    mlx_badge = (
        '<span class="ml-2 inline-flex items-center rounded-full px-2 py-px text-xs font-medium border border-neutral-600 text-neutral-600 dark:border-neutral-400 dark:text-neutral-400">MLX</span>'
        if show_mlx_badge and t.get("format") == "mlx"
        else ""
    )
    usage_level = (t.get("usage_level") or "").strip()
    active_slots = int(t.get("usage_active_slots") or 0)
    is_cloud = bool(usage_level or active_slots > 0)
    if is_cloud:
        active_bars = "".join(
            '<span x-test-model-tag-usage-slot-active class="block h-1 w-4 rounded-full bg-neutral-800 dark:bg-neutral-200"></span>'
            for _ in range(active_slots)
        )
        inactive_bars = "".join(
            '<span x-test-model-tag-usage-slot-inactive class="block h-1 w-4 rounded-full bg-neutral-200 dark:bg-neutral-700"></span>'
            for _ in range(4 - active_slots)
        )
        bars_block = f"{active_bars}{inactive_bars}"
        size_cell = f'<p x-test-model-tag-cost class="col-span-2 flex items-center gap-0.5 text-neutral-500 dark:text-neutral-400 text-[13px]">{bars_block}</p>'
        size_inline = ""
    else:
        size_cell = f'<p x-test-model-tag-size class="col-span-2 text-neutral-500 dark:text-neutral-400 text-[13px]">{size}</p>'
        size_inline = size
    size_sep = f"{size_inline} • " if size_inline else ""
    usage_text = ""
    if is_cloud and usage_level:
        usage_text = f"{usage_level.capitalize()} Usage"
    usage_sep = f"{usage_text} • " if usage_text else ""
    return f"""<div class="group px-4 py-3">
  <a href="{tag_link}" class="md:hidden flex flex-col space-y-[6px] group">
    <div class="flex items-center font-medium">
      <div class="flex items-center justify-between w-full">
        <div>
          <span class="group-hover:underline">{full_tag_esc}</span>
          {latest_badge}
          {mlx_badge}
        </div>
      </div>
    </div>
    <div class="flex flex-col text-neutral-500 dark:text-neutral-400 text-[13px]">
      <span>
        <span class="font-mono">{digest}</span> • {usage_sep}{size_sep}{ctx} context window •
        <span class="hidden sm:inline">{inp} input • {updated}</span>
      </span>
      <div class="flex sm:hidden">{inp} input • {updated}</div>
    </div>
  </a>
  <div class="hidden md:flex flex-col space-y-[6px]">
    <div class="grid grid-cols-12 items-center">
      <span class="flex items-center font-medium col-span-6 group text-sm">
        <a href="{tag_link}" class="group-hover:underline">{full_tag_esc}</a>
        {latest_badge}
        {mlx_badge}
        <input class="command hidden" value="{full_tag_esc}" />
        <button class="hidden group-hover:inline-flex ml-1.5 text-neutral-500 hover:text-black dark:hover:text-white items-center" onclick="copyToClipboard(this)">
          {SVG_COPY}
        </button>
      </span>
      {size_cell}
      <p class="col-span-2 text-neutral-500 dark:text-neutral-400 text-[13px]">{ctx}</p>
      <div class="col-span-2 text-neutral-500 dark:text-neutral-400 text-[13px]">{inp}</div>
    </div>
    <div class="flex text-neutral-500 dark:text-neutral-500 text-xs items-center">
      <span class="font-mono text-[11px]">{digest}</span>&nbsp;·&nbsp;{updated}
    </div>
  </div>
</div>"""


def _tags_table_block(rows_html: str, count: int, fmt_id: str, visible: bool) -> str:
    hidden = "" if visible else " hidden"
    mobile_label = count if visible else 0
    count_label = "1 model" if mobile_label == 1 else f"{mobile_label} models"
    return (
        f'<div id="tags-table-{fmt_id}" class="fmt-table{hidden}">\n'
        f'  <div class="overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800">\n'
        f'    <div class="min-w-full divide-y divide-neutral-200 dark:divide-neutral-800">\n'
        f'      <div class="items-center grid bg-neutral-50 dark:bg-neutral-900 px-4 py-3 text-xs grid-cols-12 text-neutral-900 dark:text-neutral-100">\n'
        f'        <p class="col-span-6 hidden md:block">Name</p>\n'
        f'        <p class="block col-span-6 md:hidden">{count_label}</p>\n'
        f'        <p class="col-span-2 hidden md:block">Size / Usage</p>\n'
        f'        <p class="col-span-2 hidden md:block">Context</p>\n'
        f'        <p class="col-span-2 hidden md:block">Input</p>\n'
        f"      </div>\n"
        f"      {rows_html}\n"
        f"    </div>\n"
        f"  </div>\n"
        f"</div>"
    )


def build_tags_page(m: dict, tags: list[dict]) -> None:
    name = m["name"]
    desc = m["description"]
    path = m["path"]
    slug_dir = PUBLIC / path.strip("/") / "tags"
    slug_dir.mkdir(parents=True, exist_ok=True)

    has_m = has_mlx(tags)
    gguf_tags = [t for t in tags if t["format"] == "gguf"]
    mlx_tags = [t for t in tags if t["format"] == "mlx"]

    pills = (
        _fmt_pills("tags", len(tags), len(gguf_tags), len(mlx_tags)) if has_m else ""
    )

    latest_digest = ""
    for t in tags:
        if t["name"] == "latest":
            latest_digest = t.get("digest") or ""
            break

    rows_all = "\n".join(
        _tags_tag_row(t, path, latest_digest, show_mlx_badge=True) for t in tags
    )
    rows_gguf = "\n".join(
        _tags_tag_row(t, path, latest_digest, show_mlx_badge=False) for t in gguf_tags
    )
    rows_mlx = "\n".join(
        _tags_tag_row(t, path, latest_digest, show_mlx_badge=False) for t in mlx_tags
    )

    table_all = _tags_table_block(rows_all, len(tags), "all", True)
    table_gguf = (
        _tags_table_block(rows_gguf, len(gguf_tags), "gguf", False) if has_m else ""
    )
    table_mlx = (
        _tags_table_block(rows_mlx, len(mlx_tags), "mlx", False) if has_m else ""
    )

    header = _header_section(m)

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html(f"{name} Tags", f"Tags for {name}. {desc}")}
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html()}

<main class="relative mx-auto flex w-full max-w-[52rem] flex-col px-6 py-10 md:py-24 lg:px-8">
  <a href="{url(esc(m["path"]) + "/")}" class="text-sm text-neutral-500 dark:text-neutral-400 hover:underline absolute top-4 left-6 z-10" onclick="if(document.referrer&amp;&amp;document.referrer.includes(location.host)){{history.back();return false;}}">&larr; Back to {esc(name)}</a>
  {header}
  <section class="w-full max-w-full mt-8 mb-4 md:mt-16 md:mb-2">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-base font-semibold leading-6 text-neutral-900 dark:text-neutral-100">Tags</h2>
    </div>
    {pills}
    {table_all}
    {table_gguf}
    {table_mlx}
  </section>
</main>

{footer_html()}
{theme_script()}
<script src="{url("/assets/app.js")}"></script>
</body>
</html>"""

    (slug_dir / "index.html").write_text(page)


# --------------------------------------------------------------------------- #
# Tag detail page
# --------------------------------------------------------------------------- #


def _tag_header_section(m: dict, tag_name: str) -> str:
    """Header section for a tag page: shows <model_name>:<tag_name> with model name linking back.

    For user (non-official) models, prepend the namespace link + "/" separator
    before the model name (e.g. "maternion / LightOnOCR-2:latest").
    """
    name = esc(m["name"])
    desc = esc(m["description"])
    pulls = format_count(m["pulls"])
    updated = esc(m["updated"])
    updated_title = esc(m.get("updated_title") or "")
    model_link = url(esc(m["path"]))
    caps = capability_spans(m["capabilities"], m["cloud"])
    sizes = size_spans(m["sizes"])

    # For user models, prepend the namespace link + "/" separator.
    namespace_html = ""
    if not m.get("official") and "/" in m["path"].strip("/"):
        namespace = m["path"].strip("/").split("/")[0]
        namespace_esc = esc(namespace)
        namespace_link = url("/" + namespace_esc)
        namespace_html = (
            f'<a x-test-model-namespace class="text-xl sm:text-[28px] font-medium leading-normal decoration-1 underline-offset-4 hover:underline shrink-0" href="{namespace_link}">{namespace_esc}</a>'
            f'<span class="text-xl sm:text-[28px] font-medium px-1 shrink-0">/</span>'
        )

    return f"""<div class="flex flex-col space-y-3">
    <div class="flex items-center min-w-0">
      <div class="flex items-center min-w-0 space-x-2">
        <div class="flex items-center min-w-0">
          {namespace_html}<span class="text-xl tracking-tight sm:text-[28px] min-w-0 truncate font-medium leading-normal text-black dark:text-neutral-100 decoration-2">
            <a x-test-model-name href="{model_link}" title="{name}" class="underline-offset-[5px] hover:underline">{name}</a>:{esc(tag_name)}
           </span>
         </div>
       </div>
     </div>
    <div class="flex flex-col space-y-2">
      <div class="flex flex-col space-y-2">
        <p class="flex space-x-5 text-[13px] font-medium text-neutral-500 dark:text-neutral-400">
          <span class="flex items-center">
            {SVG_DOWNLOAD}
            <span x-test-pull-count>{pulls}</span>
            <span class="hidden sm:flex">&nbsp;Downloads</span>
          </span>
          <span class="flex items-center" title="{updated_title}">
            {SVG_CLOCK}
            <span class="hidden sm:flex">Updated&nbsp;</span>
            <span x-test-updated>{updated}</span>
          </span>
        </p>
      </div>
      <h2 class="break-words text-neutral-800 dark:text-neutral-300">{desc}</h2>
      <div class="flex flex-wrap gap-2">
        {caps}
        {sizes}
      </div>
    </div>
  </div>"""


def build_tag_page(m: dict, tag: dict, tp: dict | None) -> None:
    name = m["name"]
    tag_name = tag["name"]
    desc = m["description"]
    path = m["path"]
    model_name = path.strip("/").split("/")[-1]
    full_name = f"{m['name']}:{tag_name}"
    # Output dir: public/library/gemma4/:latest/
    tag_dir = PUBLIC / path.strip("/") / f":{tag_name}"
    tag_dir.mkdir(parents=True, exist_ok=True)

    header = _tag_header_section(m, tag_name)
    usage = _usage_section(full_name)

    cloud_metrics = _cloud_metrics_section(tp) if tp else ""
    apps_section = _applications_section(tp) if tp else ""

    if tp and (tp.get("files") or tp.get("manifest_digest")):
        details_section = _details_section(tp)
        readme_section = _readme_section(tp)
    elif tp:
        # Has tag page data but no files (cloud tag) — skip Details, show readme only
        details_section = ""
        readme_section = _readme_section(tp)
    else:
        # Fallback minimal details box from tag list data
        digest = esc(tag.get("digest") or "") or "—"
        size = esc(tag.get("size_text") or "") or "—"
        ctx = esc(tag.get("context") or "") or "—"
        inp = esc(tag.get("input_type") or "") or "—"
        updated = esc(tag.get("updated") or "") or "—"
        details_section = f"""<section id="file-explorer" class="flex flex-1 flex-col">
  <h2 class="text-base font-semibold leading-6 text-neutral-900 dark:text-neutral-100 mb-4">Details</h2>
  <div class="overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800">
    <div class="min-w-full divide-y divide-neutral-200 dark:divide-neutral-800">
      <div class="flex items-center justify-between bg-neutral-50 dark:bg-neutral-900 px-4 py-3 text-xs text-neutral-900 dark:text-neutral-100">
        <p>Updated {updated}</p>
        <p>{digest} · {size} ·</p>
      </div>
      <div class="px-4 py-3 text-[13px] text-neutral-500 dark:text-neutral-400 grid grid-cols-12 gap-2">
        <p class="col-span-3">Context</p><p class="col-span-9">{ctx}</p>
        <p class="col-span-3">Input</p><p class="col-span-9">{inp}</p>
        <p class="col-span-3">Size / Usage</p><p class="col-span-9">{size}</p>
      </div>
    </div>
  </div>
</section>"""
        readme_section = ""

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html(f"{model_name}:{tag_name}", f"{model_name}:{tag_name} — {desc}")}
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html()}

<main class="relative mx-auto flex w-full max-w-[52rem] flex-col px-6 py-10 md:py-24 lg:px-8">
  <a href="{url(esc(path) + "/tags/")}" class="text-sm text-neutral-500 dark:text-neutral-400 hover:underline absolute top-4 left-6 z-10" onclick="if(document.referrer&amp;&amp;document.referrer.includes(location.host)){{history.back();return false;}}">&larr; Back to tags</a>
  {header}
  {cloud_metrics}
  <div class="py-8">
    {usage}
    {apps_section}
    {details_section}
  </div>
  {readme_section}
</main>

{footer_html()}
{theme_script()}
<script src="{url("/assets/app.js")}"></script>
</body>
</html>"""

    (tag_dir / "index.html").write_text(page)


# --------------------------------------------------------------------------- #
# Blob detail page
# --------------------------------------------------------------------------- #


def _blob_metadata_html(blob: dict) -> str:
    metadata = blob.get("metadata") or []
    tensors = blob.get("tensors") or []

    # --- metadata rows (2-column key/value) ---
    meta_rows = []
    for m in metadata:
        key = esc(m.get("key") or "")
        value = esc(m.get("value") or "")
        meta_rows.append(
            '<li class="px-2 sm:px-4 pt-2 sm:pb-2 grid grid-cols-8">'
            '<div class="col-span-8 sm:col-span-4">'
            f'<div class="text-neutral-600 dark:text-neutral-400 sm:text-black dark:sm:text-neutral-200">{key}</div>'
            f'<div class="sm:hidden font-mono font-medium py-1">{value}</div>'
            "</div>"
            f'<div class="hidden sm:block col-span-4 font-mono">{value}</div>'
            "</li>"
        )

    out = ['<ul role="list">']
    # Metadata section header (light bg, same as ollama.com).
    out.append(
        '<div class="sticky top-0 border-y border-neutral-100 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900 text-sm font-semibold leading-6 text-neutral-900 dark:text-neutral-100">'
        '<div class="py-2 px-4 text-xs text-neutral-900 dark:text-neutral-100">Metadata</div>'
        "</div>"
    )
    out.extend(meta_rows)

    # --- tensor section (only if any tensors were scraped) ---
    if tensors:
        # Tensor section header.
        out.append(
            '<div class="sticky top-0 border-y border-neutral-200 dark:border-neutral-800 text-sm font-semibold leading-6 text-neutral-900 dark:text-neutral-100">'
            '<div class="py-2 px-4 text-xs text-neutral-900 dark:text-neutral-100">Tensor</div>'
            "</div>"
        )
        # Column header (desktop only).
        out.append(
            '<li class="px-4 py-2 grid-cols-8 text-xs font-semibold hidden sm:grid">'
            '<div class="col-span-4">Name</div>'
            '<div class="col-span-1">Type</div>'
            '<div class="col-span-3 sm:col-span-2">Shape</div>'
            "</li>"
        )
        # Tensor rows, emitting a group divider whenever the group changes.
        current_group = ""
        for t in tensors:
            group = t.get("group") or ""
            if group != current_group:
                current_group = group
                if group:
                    out.append(
                        '<div class="sticky top-0 border-y border-neutral-200 dark:border-neutral-800 text-sm font-semibold leading-6 text-neutral-900 dark:text-neutral-100">'
                        f'<div class="py-2 px-4 text-xs text-neutral-900 dark:text-neutral-100">{esc(group)}</div>'
                        "</div>"
                    )
            name = esc(t.get("name") or "")
            dtype = esc(t.get("dtype") or "")
            shape = esc(t.get("shape") or "")
            out.append(
                '<li class="px-4 py-2 grid grid-cols-8">'
                '<div class="col-span-5 sm:col-span-4 break-words">'
                f'<div class="text-neutral-600 dark:text-neutral-400 sm:text-black dark:sm:text-neutral-200">{name}</div>'
                f'<div class="sm:hidden text-xs font-mono">{dtype}</div>'
                "</div>"
                f'<div class="col-span-1 font-mono hidden sm:block">{dtype}</div>'
                f'<div class="col-span-3 font-mono">{shape}</div>'
                "</li>"
            )

    out.append("</ul>")
    return "\n".join(out)


def _blob_content_html(blob: dict) -> str:
    blob_type = (blob.get("blob_type") or "").lower()
    if blob_type == "model":
        return _blob_metadata_html(blob)
    # license / template / params / json -> raw text with per-line divs
    content = blob.get("content") or ""
    lines = content.split("\n")
    line_divs = "".join(f"<div>{esc(ln)}</div>" for ln in lines)
    return (
        '<div class="px-4 py-2 relative overflow-x-scroll font-mono text-sm whitespace-pre-wrap [counter-reset:line] before:absolute before:left-0 before:inline-block before:w-12 before:select-none before:text-right before:text-gray-400 dark:text-neutral-300 [&>div]:pl-14 [&>div]:pr-4 [&>div]:[counter-increment:line] [&>div]:before:absolute [&>div]:before:left-0 [&>div]:before:inline-block [&>div]:before:w-12 [&>div]:before:select-none [&>div]:before:text-right [&>div]:before:text-gray-400 dark:[&>div]:before:text-gray-600 [&>div]:before:content-[counter(line)]">'
        f"{line_divs}"
        "</div>"
    )


def build_blob_page(blob: dict) -> None:
    tag_full = blob.get("tag_full") or ""
    blob_url = blob.get("blob_url") or ""
    blob_type = blob.get("blob_type") or ""
    digest = blob.get("digest") or ""
    size = blob.get("size") or ""

    # Derive the on-disk path from blob_url, which is always the full path with
    # the colon-separated tag (e.g. /library/gpt-oss:120b/blobs/<digest>).
    # The tag page dir is PUBLIC / "library" / "gpt-oss" / ":120b", so the
    # blob page lives at PUBLIC / ... / ":120b" / "blobs" / <digest>.
    url_path = blob_url.strip("/")  # library/gpt-oss:120b/blobs/<digest>
    # Split off the trailing blobs/<digest> portion.
    if "/blobs/" in url_path:
        tag_part, digest_part = url_path.rsplit("/blobs/", 1)
    else:
        tag_part, digest_part = url_path, digest
    # tag_part = "library/gpt-oss:120b" -> model_path="library/gpt-oss", tag_name="120b"
    if ":" in tag_part:
        model_path, tag_name = tag_part.rsplit(":", 1)
    else:
        model_path, tag_name = tag_part, ""
    # Reconstruct a full tag_full if the scraped one lacks the namespace prefix.
    if not tag_full or "/" not in tag_full:
        tag_full = tag_part

    blob_dir = PUBLIC / model_path / f":{tag_name}" / "blobs" / (digest_part or digest)
    blob_dir.mkdir(parents=True, exist_ok=True)

    # Load model data for the header
    model_path_full = "/" + model_path  # e.g. "/library/gpt-oss"
    tag_name_from_blob = tag_name or "latest"

    # Find the model in models.json
    m = None
    try:
        models_data = json.loads((SCRAPER / "models.json").read_text())
        for model in models_data.get("models", []):
            if model["path"] == model_path_full:
                m = model
                break
    except Exception:
        pass

    header = _tag_header_section(m, tag_name_from_blob) if m else ""

    model_name = model_path.strip("/").split("/")[-1]
    model_link = url(esc(model_path))
    tag_page_url = url(esc(model_path) + "/:" + esc(tag_name) + "/")
    tag_full_esc = esc(tag_full)
    blob_type_esc = esc(blob_type)
    digest_esc = esc(digest_part or digest)
    size_esc = esc(size)
    content_html = _blob_content_html(blob)

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html(f"{model_name}:{tag_name} — {blob_type}", f"{model_name}:{tag_name} blob {blob_type}")}
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html()}

<main class="relative mx-auto flex w-full max-w-[52rem] flex-col px-6 py-10 md:py-24 lg:px-8">
  <a href="{tag_page_url}" class="text-sm text-neutral-500 dark:text-neutral-400 hover:underline absolute top-4 left-6 z-10" onclick="if(document.referrer&amp;&amp;document.referrer.includes(location.host)){{history.back();return false;}}">&larr; Back to {tag_full_esc}</a>
  {header}
  <div id="file-explorer" class="pt-12 pb-10">
    <div class="overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800 text-neutral-800 dark:text-neutral-200">
      <div class="min-w-full divide-y divide-neutral-200 dark:divide-neutral-800">
        <div class="flex items-center justify-between bg-neutral-50 dark:bg-neutral-900 px-4 py-3 text-xs text-neutral-900 dark:text-neutral-100">
          <div class="flex items-center">
            <a href="{tag_page_url}" class="min-w-0 font-medium text-black dark:text-neutral-100 hover:underline hover:decoration-[.75px] hover:underline-offset-[3px]">
              <span class="hidden sm:block">{esc(model_name)}:{esc(tag_name)}</span>
              <span class="sm:hidden">...</span>
            </a>
            <span class="px-2 font-light text-neutral-300 dark:text-neutral-700">/</span>
            <div>{blob_type_esc}</div>
          </div>
          <div>{digest_esc} · {size_esc}</div>
        </div>
        <div class="text-[13px]">
          {content_html}
        </div>
      </div>
    </div>
  </div>
</main>

{footer_html()}
{theme_script()}
<script src="{url("/assets/app.js")}"></script>
</body>
</html>"""

    (blob_dir / "index.html").write_text(page)


def copy_assets() -> None:
    assets = PUBLIC / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # Download vendored assets from ollama.com if missing.
    vendored = [
        ("tailwind.css", "https://ollama.com/public/tailwind.css"),
        ("prism.css", "https://ollama.com/public/vendor/prism/prism.css"),
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
    # Social icons for profile pages
    social_dir = assets / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    for icon in ["default", "github", "youtube", "hugging-face", "x", "linkedin"]:
        dst = social_dir / f"{icon}.svg"
        if dst.exists():
            continue
        try:
            import urllib.request

            urllib.request.urlretrieve(
                f"https://ollama.com/public/social/{icon}.svg", dst
            )
            print(f"  downloaded social/{icon}.svg")
        except Exception as e:
            print(f"  WARN: could not download social/{icon}.svg: {e}", file=sys.stderr)

    # Profile images (download if missing)
    for name, url in [
        (
            "maternion-profile.png",
            "https://ollama.com/public/assets/63fc5cbb-8a8d-4a1a-a991-1ae0c4ed6e99/27b965ab-5457-4bcf-974d-4c3074bf536b.png",
        ),
    ]:
        dst = assets / name
        if dst.exists():
            continue
        try:
            import urllib.request

            urllib.request.urlretrieve(url, dst)
            print(f"  downloaded {name}")
        except Exception as e:
            print(f"  WARN: could not download {name}: {e}", file=sys.stderr)

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
.dark .border-neutral-200 { border-color: #262626 !important; }
.dark .border-neutral-100 { border-color: #333333 !important; }
.dark .border-neutral-300 { border-color: #525252 !important; }
.dark .bg-white { background-color: #0a0a0a; }
.dark .bg-black\/5 { background-color: rgba(255,255,255,0.05); }
.dark .hover\:bg-black\/10:hover { background-color: rgba(255,255,255,0.10); }
.dark .hover\:bg-neutral-50:hover { background-color: #262626; }
.dark .placeholder\:text-neutral-500::placeholder { color: #737373; }
.dark .text-black { color: #fafafa; }
.dark a:focus\:underline:focus { text-decoration: underline; }

/* --- Prose table borders: match ollama.com style --- */
#display table { border-collapse: collapse; }
#display td, #display th { border-bottom: 1px solid #e5e7e0; }
#display thead th { border-bottom: 2px solid #d1d5db; }
#display tr:last-child td { border-bottom: 0; }
.dark #display td, .dark #display th { border-bottom-color: #262626; }
.dark #display thead th { border-bottom-color: #404040; }

/* --- Dark: neutral classes (official Tailwind v3 palette) --- */
.dark .dark\:bg-neutral-900 { background-color: #171717; }
.dark .dark\:bg-neutral-950 { background-color: #0a0a0a; }
.dark .dark\:bg-neutral-800 { background-color: #262626; }
.dark .dark\:bg-neutral-100 { background-color: #f5f5f5; }
.dark .dark\:bg-neutral-200 { background-color: #e5e5e5; }
.dark .dark\:bg-neutral-700 { background-color: #404040; }
.dark .dark\:bg-white\/5 { background-color: rgba(255,255,255,0.05); }
.dark .dark\:bg-white\/10 { background-color: rgba(255,255,255,0.10); }
.dark .dark\:bg-white { background-color: #ffffff; }
.dark .dark\:hover\:bg-white\/20:hover { background-color: rgba(255,255,255,0.20); }
.dark .dark\:hover\:bg-neutral-800:hover { background-color: #262626; }
.dark .dark\:hover\:bg-white:hover { background-color: #ffffff; }
.dark .dark\:text-neutral-100 { color: #f5f5f5; }
.dark .dark\:text-neutral-200 { color: #e5e5e5; }
.dark .dark\:text-neutral-300 { color: #d4d4d4; }
.dark .dark\:text-neutral-400 { color: #a3a3a3; }
.dark .dark\:text-neutral-500 { color: #737373; }
.dark .dark\:text-neutral-600 { color: #525252; }
.dark .dark\:text-neutral-700 { color: #a3a3a3; }
.dark .dark\:text-neutral-900 { color: #171717; }
.dark .dark\:border-neutral-700 { border-color: #404040 !important; }
.dark .dark\:border-neutral-800 { border-color: #262626 !important; }
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

/* --- Usage section dark mode --- */
.dark section[data-usage-section] .border-neutral-200 { border-color: #262626 !important; }
.dark .use-tab.text-neutral-900 { color: #fafafa; }
.dark .use-tab.text-neutral-400 { color: #737373; }
.dark .dark\:hover\:text-neutral-300:hover { color: #d4d4d4; }

/* --- File explorer dark mode --- */
.dark .bg-neutral-50 { background-color: #171717; }
/* Inner dividers: keep the same dark color as the outer border (neutral-800 = #262626)
   so the inside dividers match the outside border of the tag table / file-explorer.
   Use the same high-specificity child selector as the vendored tailwind.css
   (>:not([hidden])~:not([hidden])) plus !important so the dark color wins over
   the vendored light-mode rule (which sets border-color: rgb(229 229 229/...)). */
.dark .dark\:divide-neutral-800 > :not([hidden]) ~ :not([hidden]) { border-color: #262626 !important; }
.dark .divide-neutral-200 > :not([hidden]) ~ :not([hidden]) { border-color: #262626 !important; }
.dark .divide-gray-200 > :not([hidden]) ~ :not([hidden]) { border-color: #262626 !important; }

/* --- Readme / prose ---
   The vendored tailwind.css includes the `.prose` base styles and most
   `prose-*` variant utilities, but is missing a handful used by ollama.com's
   `<div id="display">` (the bracket/precise-value heading margins, the
   `prose-code:py-1` and `prose-code:text-[85%]` utilities, and the
   `prose-td code:*` descendant compound variants). It also contains NO
   `dark:` variants at all, so every dark:prose-* utility on the readme
   container is implemented here. */

/* Light-mode prose-* utilities missing from vendored tailwind.css */
.prose-headings\:mb-\[0\.7em\] :is(:where(h1):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mb-\[0\.7em\] :is(:where(h2):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mb-\[0\.7em\] :is(:where(h3):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mb-\[0\.7em\] :is(:where(h4):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mb-\[0\.7em\] :is(:where(h5):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mb-\[0\.7em\] :is(:where(h6):not(:where([class~=not-prose],[class~=not-prose] *))) { margin-bottom: 0.7em; }
.prose-headings\:mt-\[1\.25em\] :is(:where(h1):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mt-\[1\.25em\] :is(:where(h2):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mt-\[1\.25em\] :is(:where(h3):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mt-\[1\.25em\] :is(:where(h4):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mt-\[1\.25em\] :is(:where(h5):not(:where([class~=not-prose],[class~=not-prose] *))),
.prose-headings\:mt-\[1\.25em\] :is(:where(h6):not(:where([class~=not-prose],[class~=not-prose] *))) { margin-top: 1.25em; }
.prose-code\:text-\[85\%\] :is(:where(code):not(:where([class~=not-prose],[class~=not-prose] *))) { font-size: 85%; }
.prose-code\:py-1 :is(:where(code):not(:where([class~=not-prose],[class~=not-prose] *))) { padding-top: 0.25rem; padding-bottom: 0.25rem; }
/* `prose-td code:*` — code elements inside td cells within .prose tables */
.prose-td code\:display-inline-block :is(:where(td code):not(:where([class~=not-prose],[class~=not-prose] *))) { display: inline-block; }
.prose-td code\:bg-gray-200 :is(:where(td code):not(:where([class~=not-prose],[class~=not-prose] *))) { background-color: #e5e7eb; }
.prose-td code\:px-2 :is(:where(td code):not(:where([class~=not-prose],[class~=not-prose] *))) { padding-left: 0.5rem; padding-right: 0.5rem; }
.prose-td code\:py-1 :is(:where(td code):not(:where([class~=not-prose],[class~=not-prose] *))) { padding-top: 0.25rem; padding-bottom: 0.25rem; }
.prose-td code\:rounded-md :is(:where(td code):not(:where([class~=not-prose],[class~=not-prose] *))) { border-radius: 0.375rem; }

/* --- Dark-mode prose (no dark: classes ship in vendored tailwind.css) --- */
.dark .prose { color: #d4d4d4; }
.dark .prose h1, .dark .prose h2, .dark .prose h3, .dark .prose h4, .dark .prose h5, .dark .prose h6 { color: #f5f5f5; }
.dark .prose p { color: #d4d4d4; }
.dark .prose a { color: #60a5fa; }
.dark .prose code { background-color: #262626; color: #e5e5e5; }
.dark .prose pre { background-color: #171717; color: #e5e5e5; }
.dark .prose pre code { background-color: transparent; color: inherit; padding: 0; }
.dark .prose blockquote { color: #a3a3a3; border-left-color: #404040; }
.dark .prose ul, .dark .prose ol { color: #d4d4d4; }
.dark .prose li { color: #d4d4d4; }
.dark .prose li::marker { color: #a3a3a3; }
.dark .prose img { border-radius: 8px; }
.dark .prose strong { color: #f5f5f5; }
.dark .prose table { color: #d4d4d4; }
.dark .prose th { color: #f5f5f5; }
.dark .prose td { color: #d4d4d4; }
.dark .prose hr { border-color: #404040; }
/* dark:prose-* utilities on the readme container */
.dark .dark\:prose-headings\:text-neutral-200 :is(:where(h1):not(:where([class~=not-prose],[class~=not-prose] *))),
.dark .dark\:prose-headings\:text-neutral-200 :is(:where(h2):not(:where([class~=not-prose],[class~=not-prose] *))),
.dark .dark\:prose-headings\:text-neutral-200 :is(:where(h3):not(:where([class~=not-prose],[class~=not-prose] *))),
.dark .dark\:prose-headings\:text-neutral-200 :is(:where(h4):not(:where([class~=not-prose],[class~=not-prose] *))),
.dark .dark\:prose-headings\:text-neutral-200 :is(:where(h5):not(:where([class~=not-prose],[class~=not-prose] *))),
.dark .dark\:prose-headings\:text-neutral-200 :is(:where(h6):not(:where([class~=not-prose],[class~=not-prose] *))) { color: #e5e5e5; }
.dark .dark\:prose-blockquote\:text-neutral-400 :is(:where(blockquote):not(:where([class~=not-prose],[class~=not-prose] *))) { color: #a3a3a3; }
.dark .dark\:prose-code\:bg-neutral-800 :is(:where(code):not(:where([class~=not-prose],[class~=not-prose] *))) { background-color: #262626; }
.dark .dark\:prose-code\:text-neutral-200 :is(:where(code):not(:where([class~=not-prose],[class~=not-prose] *))) { color: #e5e5e5; }
.dark .dark\:prose-pre\:bg-neutral-900 :is(:where(pre):not(:where([class~=not-prose],[class~=not-prose] *))) { background-color: #171717; }
.dark .dark\:prose-pre\:text-neutral-200 :is(:where(pre):not(:where([class~=not-prose],[class~=not-prose] *))) { color: #e5e5e5; }
.dark .dark\:prose-li\:text-neutral-200 :is(:where(li):not(:where([class~=not-prose],[class~=not-prose] *))) { color: #e5e5e5; }
.dark .dark\:prose-td\:text-neutral-300 :is(:where(td):not(:where([class~=not-prose],[class~=not-prose] *))) { color: #d4d4d4; }
.dark .dark\:prose-th\:text-neutral-200 :is(:where(th):not(:where([class~=not-prose],[class~=not-prose] *))) { color: #e5e5e5; }
.dark .dark\:prose-a\:text-blue-400 :is(:where(a):not(:where([class~=not-prose],[class~=not-prose] *))) { color: #60a5fa; }
.dark .dark\:prose-strong\:text-neutral-200 :is(:where(strong):not(:where([class~=not-prose],[class~=not-prose] *))) { color: #e5e5e5; }
.dark .dark\:marker\:prose-ol\:text-neutral-400 :is(:where(ol):not(:where([class~=not-prose],[class~=not-prose] *))) ::marker { color: #a3a3a3; }
.dark .dark\:marker\:prose-ul\:text-neutral-400 :is(:where(ul):not(:where([class~=not-prose],[class~=not-prose] *))) ::marker { color: #a3a3a3; }
.dark .dark\:text-neutral-200 { color: #e5e5e5; }

/* --- text-green-700 for code snippets (dark mode only; light is in tailwind.css) --- */
.dark .text-green-700 { color: #4ade80; }

/* --- fmt pill radio dark mode --- */
.dark .peer:checked ~ label { background-color: #1e1b4b; border-color: #6366f1; }
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
    var isCloud = card.getAttribute('data-cloud') === 'true';
    var isCloudOnly = card.getAttribute('data-cloud-only') === 'true';
    var matchText = !q || title.indexOf(q) !== -1 || desc.indexOf(q) !== -1;
    var matchCaps = caps.length === 0 || caps.every(function(c) { return cardCaps.indexOf(c) !== -1; });
    var matchCloud = cloudFilter === 'all'
      || (cloudFilter === 'cloud' && isCloud)
      || (cloudFilter === 'local' && !isCloudOnly);
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

// --- Usage section: tab switching + copy ---
function switchUsageTab(btn, tabName) {
  var section = btn.closest('section');
  section.querySelectorAll('.use-tab').forEach(function(tab) {
    tab.classList.remove('text-neutral-900', 'font-medium', 'underline', 'decoration-1', 'underline-offset-[7px]');
    tab.classList.add('text-neutral-400');
  });
  btn.classList.remove('text-neutral-400');
  btn.classList.add('text-neutral-900', 'font-medium', 'underline', 'decoration-1', 'underline-offset-[7px]');
  section.querySelectorAll('.use-panel').forEach(function(panel) { panel.classList.add('hidden'); });
  var activePanel = section.querySelector('.use-panel[data-panel="' + tabName + '"]');
  if (activePanel) activePanel.classList.remove('hidden');
  section.querySelectorAll('.use-link').forEach(function(link) { link.classList.add('hidden'); });
  var activeLink = section.querySelector('.use-link[data-link="' + tabName + '"]');
  if (activeLink) activeLink.classList.remove('hidden');
}
window.switchUsageTab = switchUsageTab;

function copyUsageCode(btn) {
  var section = btn.closest('section');
  var activePanel = section.querySelector('.use-panel:not(.hidden)');
  if (!activePanel) return;
  var pre = activePanel.querySelector('pre');
  if (!pre) return;
  navigator.clipboard.writeText(pre.textContent).then(function() {
    var copyIcon = btn.querySelector('.copy-icon');
    var checkIcon = btn.querySelector('.check-icon');
    if (copyIcon) copyIcon.classList.add('hidden');
    if (checkIcon) checkIcon.classList.remove('hidden');
    setTimeout(function() {
      if (copyIcon) copyIcon.classList.remove('hidden');
      if (checkIcon) checkIcon.classList.add('hidden');
    }, 2000);
  });
}
window.copyUsageCode = copyUsageCode;

// --- Format pill radio filters (detail + tags pages) ---
function initFmtFilters() {
  var radios = document.querySelectorAll('.fmt-radio');
  if (!radios.length) return;
  radios.forEach(function(radio) {
    radio.addEventListener('change', function() {
      var fmt = radio.getAttribute('data-fmt');
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
  initFmtFilters();
});
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def build_profile_page(username: str) -> None:
    """Build a user profile page (e.g. /maternion) mirroring ollama.com layout."""
    profile_path = HERE / "scraper" / f"profile_{username}.json"
    if not profile_path.exists():
        print(f"  profile {username}: no data, skipping")
        return

    profile = json.loads(profile_path.read_text())
    bio = esc(profile.get("bio", ""))
    links = profile.get("links", [])
    model_paths = profile.get("models", [])

    # Model data: use profile's embedded card data, fall back to models.json
    all_models = load_models()
    models_by_path = {m["path"]: m for m in all_models}
    profile_models = []
    for m in model_paths:
        if isinstance(m, dict):
            profile_models.append(m)
        elif isinstance(m, str) and m in models_by_path:
            profile_models.append(models_by_path[m])

    # Build model cards (reuse render_card)
    # Compute profile-specific ranks: popular = pulls desc, newest = updated_title desc.
    # Merge with global ranks so models present there keep their global ranks; the
    # rest get local ranks so the sort dropdown works on the profile page too.
    global_ranks = load_ranks()
    profile_ranks = dict(global_ranks)

    # Popular rank from pulls (descending)
    popular_order = sorted(
        profile_models,
        key=lambda m: m.get("pulls", 0),
        reverse=True,
    )
    for rank, m in enumerate(popular_order):
        nm = m["name"]
        pr = profile_ranks.setdefault(nm, {})
        pr["popular_rank"] = rank

    # Newest rank from updated_title (descending date)
    from datetime import datetime as _dt

    def _parse_updated(s: str) -> _dt:
        try:
            return _dt.strptime(s, "%b %d, %Y %I:%M %p UTC")
        except Exception:
            return _dt.min

    newest_order = sorted(
        profile_models,
        key=lambda m: _parse_updated(m.get("updated_title") or ""),
        reverse=True,
    )
    for rank, m in enumerate(newest_order):
        nm = m["name"]
        pr = profile_ranks.setdefault(nm, {})
        pr["newest_rank"] = rank

    # Default server-side order: popular (pulls descending) — matches ?sort=popular default
    sorted_models = sorted(
        profile_models,
        key=lambda m: profile_ranks.get(m["name"], {}).get("popular_rank", 9999),
    )
    cards_html = ""
    for m in sorted_models:
        tags = load_tags(m["path"])
        cards_html += render_card(m, tags, profile_ranks)

    if not cards_html:
        cards_html = '<p class="text-neutral-500 dark:text-neutral-400 py-8">No models found.</p>'

    # Sort options — profile page only has Popular / Newest (per ollama.com)
    sort_options = [
        ("popular", "Popular"),
        ("newest", "Newest"),
    ]
    opt_html = "\n".join(
        f'        <option value="{v}">{l}</option>' for v, l in sort_options
    )

    # Links HTML
    links_html = ""
    for link in links:
        link_url = esc(link["url"])
        label = esc(link["label"])
        links_html += f"""              <span class="inline-flex gap-x-2 items-center">
                <div class="inline-flex items-center space-x-1">
                  <div class="inline-flex items-center space-x-1">
                    <img src="{url("/assets/social/default.svg")}" class="w-4 h-4" alt="default icon" onload="setDisplayIcon(this, '{link_url}'); this.onload=null" />
                    <a href="//{link_url}" target="_blank" class="hover:underline text-sm text-neutral-700 dark:text-neutral-300">
                      {label}
                    </a>
                  </div>
                </div>
              </span>
"""

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html(username, bio)}
    <script>
      function getIcon(url) {{
        url = url.toLowerCase();
        if (url.includes('x.com') || url.includes('twitter.com')) return 'x';
        if (url.includes('github.com')) return 'github';
        if (url.includes('linkedin.com')) return 'linkedin';
        if (url.includes('youtube.com')) return 'youtube';
        if (url.includes('hf.co') || url.includes('huggingface.co') || url.includes('huggingface.com')) return 'hugging-face';
        return 'default';
      }}
      function setDisplayIcon(imgElement, url) {{
        var icon = getIcon(url);
        imgElement.src = '{url("/assets/social/")}' + icon + '.svg';
        imgElement.alt = icon + ' icon';
      }}
    </script>
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html("")}

<main class="mx-auto flex w-full max-w-2xl flex-col px-6 py-5 md:py-12 lg:px-8">
  <div class="grid grid-cols-4 gap-4 md:gap-0">
    <div class="col-span-1">
      <div class="flex w-20 flex-col items-center md:w-28">
        <div class="group relative h-20 w-20 overflow-hidden rounded-full md:h-28 md:w-28">
          <img src="{url("/assets/maternion-profile.png")}" alt="profile" class="absolute inset-0 h-full w-full border border-neutral-300 object-cover rounded-full" />
        </div>
      </div>
    </div>
    <div class="col-span-3">
      <div class="flex flex-grow flex-col">
        <div class="flex flex-row items-center justify-between">
          <span class="text-[28px] font-medium tracking-tight">{esc(username)}</span>
        </div>
        <div class="space-y-1">
          <div class="my-2">
            <h2 class="break-words sm:max-w-lg">
              <span>{bio}</span>
            </h2>
          </div>
          <div class="flex flex-col space-y-0.5 w-fit">
{links_html}          </div>
        </div>
      </div>
    </div>
  </div>

  <input type="hidden" id="sort-value" name="o" value="popular">

  <div id="searchresults" class="w-full space-y-2 mt-8">
    <div class="flex flex-wrap items-center justify-between gap-2">
      <div class="sm:hidden relative">
        <select id="mobile-sort-select" class="absolute inset-0 w-6 px-3 py-1 opacity-0 appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:ring focus:outline-none focus:ring-blue-300 focus:ring-opacity-75 focus:border-blue-400 dark:focus:border-blue-600">
{opt_html}
        </select>
        <div class="w-6 px-3.5 py-1.5 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 flex items-center justify-center pointer-events-none">
          <span class="text-neutral-900 dark:text-neutral-100 text-xs font-medium">&#x21C5;</span>
        </div>
      </div>
      <div class="hidden sm:block ml-auto">
        <select id="desktop-sort-select" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:ring focus:outline-none focus:ring-blue-300 focus:ring-opacity-75 focus:border-blue-400 dark:focus:border-blue-600 min-w-[120px] text-sm px-3 py-1.5">
{opt_html}
        </select>
      </div>
    </div>

    <ul role="list" id="card-list" class="grid grid-cols-1 gap-y-3">
{cards_html}
    </ul>
    <p id="no-results" class="hidden py-12 text-center text-neutral-400 dark:text-neutral-600">No models found.</p>
  </div>
</main>

{footer_html()}
{theme_script()}
<script src="{url("/assets/app.js")}"></script>
</body>
</html>"""

    out_dir = PUBLIC / username
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page)
    print(f"  profile {username}: {len(profile_models)} models")


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

    # index/search page -> /index.html (main page)
    print("building index.html ...")
    build_index(models, load_ranks())

    # /search/ redirect -> / (backwards compat)
    print("building /search/ redirect ...")
    root_url = url("/")
    (PUBLIC / "search").mkdir(parents=True, exist_ok=True)
    (PUBLIC / "search" / "index.html").write_text(
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        f'<meta http-equiv="refresh" content="0; url={root_url}">\n'
        "<title>Ollama Search</title>\n"
        "</head>\n"
        f'<body>Redirecting to <a href="{root_url}">search</a>...</body>\n'
        "</html>\n"
    )

    # profile pages (e.g. /maternion) + profile model detail pages
    print("building profile pages ...")
    build_profile_page("maternion")

    # Load profile models and add them to the build list
    import json as _json

    _all_models = list(models)
    for _username in ["maternion"]:
        _pf = HERE / "scraper" / f"profile_{_username}.json"
        if _pf.exists():
            _pdata = _json.loads(_pf.read_text())
            _existing_paths = {m["path"] for m in _all_models}
            for _m in _pdata.get("models", []):
                if isinstance(_m, dict) and _m["path"] not in _existing_paths:
                    _all_models.append(_m)

    # model detail + tags + per-tag pages
    print(f"building model detail + tags + tag pages ({len(_all_models)} models) ...")
    tag_pages_built = 0
    blob_pages_built = 0
    for i, m in enumerate(_all_models, 1):
        tags = load_tags(m["path"])
        build_detail(m, tags)
        build_tags_page(m, tags)
        for t in tags:
            tp = load_tag_page(m["path"], t["name"])
            build_tag_page(m, t, tp)
            tag_pages_built += 1
            if tp:
                for f in tp.get("files", []):
                    blob_url = f.get("blob_url") or f.get("url") or ""
                    if blob_url:
                        bp = load_blob_page(blob_url)
                        if bp:
                            build_blob_page(bp)
                            blob_pages_built += 1
        if i % 50 == 0:
            print(f"  {i}/{len(_all_models)}")
    print(
        f"built {len(_all_models)} model pages + tags pages + {tag_pages_built} tag pages + {blob_pages_built} blob pages"
    )

    # write the catalog JSON for client-side use
    (PUBLIC / "assets" / "models.json").write_text(json.dumps(models, indent=2))
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

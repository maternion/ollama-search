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

# Set of model paths that are new this scrape run (from scraper/new_models.json).
# Populated in main() and used by render_card() and _header_section() to show
# a "NEW" badge. Empty when new_models.json doesn't exist (first run).
_NEW_MODELS: set[str] = set()

# Context-length ticks (in tokens) at which KV-cache memory is sampled for the
# graph panel. The per-tag curve uses every tick strictly below the tag's
# context limit, plus a final endpoint at the limit itself.
# Extended past 256K so long-context models (e.g. llama4 10M) get real data
# points — not just a single straight segment — across the log region.
GRAPH_TICKS = [
    0,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
]


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


# --- Readme HTML sanitizer ------------------------------------------------- #
# Allowlist-based sanitizer for scraped readme HTML. Strips <script>, on*
# event handlers, javascript: URLs, and any tags/attributes not in the
# allowlist. This is defense-in-depth: ollama.com likely sanitizes
# server-side when rendering markdown, but community models (frob,
# huihui_ai, maternion profiles) could potentially inject XSS if
# ollama.com's own sanitization is ever bypassed.

_README_ALLOWED_TAGS = {
    "a",
    "b",
    "blockquote",
    "br",
    "code",
    "col",
    "colgroup",
    "dd",
    "del",
    "details",
    "div",
    "dl",
    "dt",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "i",
    "img",
    "ins",
    "kbd",
    "li",
    "mark",
    "ol",
    "p",
    "pre",
    "q",
    "s",
    "small",
    "span",
    "strong",
    "sub",
    "summary",
    "sup",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
    "var",
}

_README_ALLOWED_ATTRS = {
    "a": {"href", "rel", "title"},
    "img": {"src", "alt", "width", "height"},
    "td": {"align", "colspan", "rowspan"},
    "th": {"align", "colspan", "rowspan"},
    "tr": {"align"},
    "table": {"align"},
    "col": {"align", "span"},
    "colgroup": {"align", "span"},
    "span": {"class"},
    "div": {"class"},
    "code": {"class"},
    "details": {"open"},
}

# Tags that should be dropped entirely (tag + content removed)
_README_DROP_TAGS = {
    "script",
    "style",
    "iframe",
    "object",
    "embed",
    "form",
    "input",
    "meta",
    "link",
}

_VOID_TAGS = {"br", "hr", "img", "col", "input", "meta", "link", "embed"}

_TAG_RE = re.compile(
    r"</?(\w+)((?:\s+[^>]*)?)(/?)>|&[a-zA-Z#0-9]+;|([^<]+)",
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(\w[\w-]*)\s*=\s*(["\'])(.*?)\2|(\w[\w-]*)(?=\s|$|/|>)')


def _sanitize_url(url_val: str) -> str:
    u = url_val.strip()
    low = u.lower()
    if low.startswith(("http://", "https://", "mailto:", "ftp://", "data:", "/", "#")):
        return u
    if low.startswith("javascript:") or low.startswith("vbscript:"):
        return ""
    return u


def _esc_text(s: str) -> str:
    """Escape literal < and > in text content while preserving existing HTML
    entities (e.g. &amp;, &#34;) that were already in the scraped HTML."""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def _esc_attr(s: str) -> str:
    """Escape quotes in attribute values while preserving existing HTML
    entities. Unlike html.escape, does not re-escape & in existing entities."""
    return s.replace('"', "&quot;").replace("'", "&#x27;")


def sanitize_readme_html(html_str: str) -> str:
    if not html_str:
        return ""
    out: list[str] = []
    i = 0
    n = len(html_str)

    while i < n:
        if html_str[i] == "<":
            m = _TAG_RE.match(html_str, i)
            if not m:
                out.append("&lt;")
                i += 1
                continue

            if m.group(4) is not None or m.group(0).startswith("&"):
                out.append(m.group(0))
                i = m.end()
                continue

            tag_name = m.group(1).lower()
            raw_attrs = m.group(2) or ""
            self_closing = m.group(3) == "/"
            is_closing = m.group(0).startswith("</")

            if tag_name in _README_DROP_TAGS:
                if not is_closing and not self_closing and tag_name not in _VOID_TAGS:
                    end_tag = f"</{tag_name}"
                    end_idx = html_str.lower().find(end_tag, m.end())
                    if end_idx != -1:
                        close_idx = html_str.find(">", end_idx)
                        i = close_idx + 1 if close_idx != -1 else m.end()
                    else:
                        i = m.end()
                else:
                    i = m.end()
                continue

            if tag_name not in _README_ALLOWED_TAGS:
                i = m.end()
                continue

            if is_closing:
                out.append(f"</{tag_name}>")
                i = m.end()
                continue

            allowed_attrs = _README_ALLOWED_ATTRS.get(tag_name, set())
            clean_attrs: list[str] = []
            for am in _ATTR_RE.finditer(raw_attrs):
                attr_name = (am.group(1) or am.group(4)).lower()
                attr_val = am.group(3) if am.group(3) is not None else ""

                if not attr_name or attr_name.startswith("on"):
                    continue
                if attr_name not in allowed_attrs:
                    continue

                if attr_name in ("href", "src"):
                    attr_val = _sanitize_url(attr_val)
                    if not attr_val:
                        continue

                clean_attrs.append(f'{attr_name}="{_esc_attr(attr_val)}"')

            attr_str = (" " + " ".join(clean_attrs)) if clean_attrs else ""
            if tag_name in _VOID_TAGS or self_closing:
                out.append(f"<{tag_name}{attr_str} />")
            else:
                out.append(f"<{tag_name}{attr_str}>")
            i = m.end()
        else:
            end = html_str.find("<", i)
            if end == -1:
                out.append(_esc_text(html_str[i:]))
                break
            out.append(_esc_text(html_str[i:end]))
            i = end

    return "".join(out)


def slugify(path: str) -> str:
    return path.strip("/").replace("/", "__")


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
    return [m for m in data["models"] if m["path"] not in IGNORELIST]


def load_new_model_paths() -> set[str]:
    """Load the set of model paths that are new this scrape run.

    Written by the scraper's save_new_models() — a diff of current vs
    previous models.json. Used to render a "NEW" badge on model cards
    and detail pages.
    """
    fp = SCRAPER / "new_models.json"
    if fp.exists():
        try:
            data = json.loads(fp.read_text())
            return set(data.get("paths", []))
        except Exception:
            pass
    return set()


def load_ranks() -> dict:
    rf = SCRAPER / "sort_ranks.json"
    if rf.exists():
        return json.loads(rf.read_text())
    return {}


def _parse_updated_title(s: str):
    """Parse an ollama.com absolute update timestamp e.g. "Nov 19, 2023 1:58 PM UTC".

    Returns a sortable datetime (datetime.min on failure). Used to derive
    the "updated" sort order (most-recent tag update), which is distinct
    from the "newest" sort order (model creation/first-publish, sourced
    from ollama.com's /library?sort=newest and stored as newest_rank).
    """
    from datetime import datetime as _dt

    try:
        return _dt.strptime(s, "%b %d, %Y %I:%M %p UTC")
    except Exception:
        return _dt.min


def load_profile_ranks() -> dict:
    """Load popular_rank and newest_rank for profile models, keyed by path.

    Profile pages on ollama.com support ?sort=popular (pulls) and ?sort=newest.
    The scraper stores both orderings in profile_<username>.json.
    Returns a dict: {model_path: {popular_rank: int, newest_rank: int}}.
    """
    ranks: dict[str, dict] = {}
    for username in ("maternion", "frob", "huihui_ai"):
        pf = SCRAPER / f"profile_{username}.json"
        if not pf.exists():
            continue
        pdata = json.loads(pf.read_text())
        for rank, path in enumerate(pdata.get("popular_order", [])):
            ranks.setdefault(path, {})["popular_rank"] = rank
        for rank, path in enumerate(pdata.get("newest_order", [])):
            ranks.setdefault(path, {})["newest_rank"] = rank
    return ranks


def load_tags(model_path: str, model: dict | None = None) -> list[dict]:
    tf = TAGS_DIR / f"{slugify(model_path)}.json"
    if tf.exists():
        return json.loads(tf.read_text()).get("tags", [])
    if model and model.get("tags"):
        return model["tags"]
    return []


def has_mlx(tags: list[dict]) -> bool:
    return any(t["format"] == "mlx" for t in tags)


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


_BLOB_LOAD_CACHE: dict[str, dict | None] = {}


def load_blob_page(blob_url: str) -> dict | None:
    # Blobs are stored once per digest at BLOBS_DIR/<digest>.json. Extract the
    # digest from the blob URL (last path segment before any query string):
    # /library/model:tag/blobs/<digest> -> <digest>.
    digest = blob_url.rstrip("/").rsplit("/blobs/", 1)[-1].split("?", 1)[0]
    if not digest or "/" in digest or not re.match(r"^[a-fA-F0-9]+$", digest):
        return None
    if digest in _BLOB_LOAD_CACHE:
        return _BLOB_LOAD_CACHE[digest]
    bf = BLOBS_DIR / f"{digest}.json"
    if not bf.exists():
        _BLOB_LOAD_CACHE[digest] = None
        return None
    result = json.loads(bf.read_text())
    _BLOB_LOAD_CACHE[digest] = result
    return result


def _tag_url(model_path: str, tag_name: str) -> str:
    """Build the local URL for a tag page, matching ollama.com's scheme:
    /library/model:tag (colon directly after the model name, no / before it).
    """
    return url(esc(model_path) + ":" + esc(tag_name) + "/")


def _blob_href(blob_url: str) -> str:
    """Return local blob page URL if blob data exists, else external ollama.com URL.

    blob_url already uses the colon-attached form (e.g.
    /library/model:tag/blobs/<digest>), so it maps directly to the on-disk
    directory structure (library/model:tag/blobs/<digest>/index.html).
    """
    if blob_url and load_blob_page(blob_url):
        return url(blob_url + "/")
    return "https://ollama.com" + blob_url


# --------------------------------------------------------------------------- #
# Shared HTML fragments
# --------------------------------------------------------------------------- #


def head_html(title: str, description: str) -> str:
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
      <a class="hover:underline focus:underline focus:outline-none focus:ring-0" href="{url("/pricing")}">Pricing</a>
    </div>
    <div class="flex-grow justify-center items-center hidden md:flex">
      <div class="relative w-full" style="max-width: 448px;">
        <form action="{url("")}" autocomplete="off" id="nav-search-form">
          <div class="relative flex w-full appearance-none bg-black/5 dark:bg-white/5 border border-neutral-100 dark:border-neutral-800 items-center rounded-full" hx-on:focusout="var rt=event.relatedTarget;if(rt&&(this.contains(rt)||document.getElementById('searchpreview').contains(rt)))return;var sp=document.getElementById('searchpreview');if(sp)sp.classList.add('hidden');">
            <span class="pl-2 text-2xl text-neutral-500 dark:text-neutral-400">{SVG_SEARCH}</span>
            <input id="navbar-input" name="q" type="text" class="resize-none rounded-full border-0 py-2.5 bg-transparent text-sm w-full placeholder:text-neutral-500 dark:placeholder:text-neutral-500 focus:outline-none focus:ring-0 dark:text-neutral-200" placeholder="Search models" autocomplete="off" hx-on:keydown="if(event.key==='Enter'){{event.preventDefault();var v=this.value.trim();window.location.href=v?'{url("/?q=")}'+encodeURIComponent(v):'{url("/")}';return;}}if(event.key==='Escape'){{event.preventDefault();this.value='';this.blur();var sp=document.getElementById('searchpreview');if(sp)sp.classList.add('hidden');return;}}if(event.key==='Tab'){{var sp=document.getElementById('searchpreview');if(sp)sp.classList.add('hidden');return;}}if(event.key==='ArrowDown'){{var first=document.querySelector('#search-preview-list a:first-of-type');if(first)first.focus();event.preventDefault();}}if(event.key==='ArrowUp'){{var last=document.getElementById('view-all-link');if(last)last.focus();event.preventDefault();}}var sp=document.getElementById('searchpreview');if(sp)sp.classList.remove('hidden');" hx-on:focus="var sp=document.getElementById('searchpreview');if(sp&&this.value.trim())sp.classList.remove('hidden');" />
          </div>
        </form>
        <div id="searchpreview" class="hidden absolute left-0 right-0 top-12 z-50" style="width: calc(100% + 2px); margin-left: -1px;"></div>
      </div>
    </div>
    <div class="hidden lg:flex xl:flex-1 items-center space-x-2 justify-end ml-6 xl:ml-0">
      <button id="theme-toggle" class="flex cursor-pointer items-center rounded-full bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 text-lg px-3 py-1.5 text-black dark:text-neutral-200 whitespace-nowrap" title="Toggle dark mode">
        <span class="dark:hidden">{SVG_MOON}</span>
        <span class="hidden dark:block">{SVG_SUN}</span>
      </button>
      <a class="flex cursor-pointer items-center rounded-full bg-neutral-800 dark:bg-neutral-100 text-lg px-4 py-1.5 text-white dark:text-neutral-900 hover:bg-black dark:hover:bg-white whitespace-nowrap focus:bg-black dark:focus:bg-white" href="{url("/download")}">Download</a>
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
        <a href="{url("/download")}" class="hover:underline">Download</a>
        <a href="https://ollama.com/blog" class="hover:underline">Blog</a>
        <a href="https://docs.ollama.com" class="hover:underline">Docs</a>
        <a href="https://github.com/ollama/ollama" class="hover:underline">GitHub</a>
        <a href="{url("/pricing")}" class="hover:underline">Pricing</a>
      </div>
    </div>
  </div>
  <div class="py-4 md:hidden">
    <ul class="flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-neutral-500 dark:text-neutral-400">
      <li><a href="{url("/download")}" class="hover:underline">Download</a></li>
      <li><a href="https://ollama.com/blog" class="hover:underline">Blog</a></li>
      <li><a href="https://docs.ollama.com" class="hover:underline">Docs</a></li>
      <li><a href="https://github.com/ollama/ollama" class="hover:underline">GitHub</a></li>
      <li><a href="{url("/pricing")}" class="hover:underline">Pricing</a></li>
    </ul>
    <div class="mt-2 text-center text-xs text-neutral-500 dark:text-neutral-400">&copy; 2026 Ollama · <a href="{url("/maternion/")}" class="hover:underline">Maternion</a></div>
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


def pricing_tab_script_head() -> str:
    """Inline script for <head> on the pricing page — sets html.tab-* class from
    location.hash BEFORE the body parses, so CSS can show the correct panel
    immediately (no FOUC). Mirrors the dark-mode theme script pattern."""
    return """<script>
(function() {
  var h = window.location.hash;
  document.documentElement.classList.add(h === '#teams' ? 'tab-teams' : 'tab-individuals');
})();
</script>"""


def theme_script() -> str:
    """Toggle handler — placed at end of body."""
    return r"""<script>
(function() {
  function toggle() {
    var isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  }
  function initSearch() {
    document.getElementById('theme-toggle')?.addEventListener('click', toggle);
    document.getElementById('theme-toggle-mobile')?.addEventListener('click', toggle);
  }
  window.initSearch = initSearch;
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSearch);
  } else {
    initSearch();
  }
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


RENDERER_ARCHS = {
    "gemma4",
    "olmo3",
    "qwen35",
    "qwen35moe",
    "qwen3vl",
    "qwen3vlmoe",
    "glm4moelite",
    "glmocr",
    "deepseekocr",
    "cohere2moe",
    "nemotron_h",
    "nemotron_h_moe",
    "nemotron_h_omni",
    "qwen3moe",
    "qwen3next",
    "lfm2",
    "lfm2moe",
}


def _model_blob_metadata(model_path: str) -> list[dict]:
    """Load metadata from the model blob of the latest tag page."""
    slug = model_path.strip("/").replace("/", "__")
    tp_file = TAG_PAGES_DIR / f"{slug}__latest.json"
    if not tp_file.exists():
        return []
    tp = json.loads(tp_file.read_text())
    for f in tp.get("files") or []:
        if f.get("type") == "model":
            digest = f.get("blob_url", "").rsplit("/blobs/", 1)[-1].split("?")[0]
            bf = BLOBS_DIR / f"{digest}.json"
            if bf.exists():
                b = json.loads(bf.read_text())
                return b.get("metadata") or []
    return []


def _template_blob_content(model_path: str) -> str | None:
    """Load the template blob content from the latest tag page, or None if no template row."""
    slug = model_path.strip("/").replace("/", "__")
    tp_file = TAG_PAGES_DIR / f"{slug}__latest.json"
    if not tp_file.exists():
        return None
    tp = json.loads(tp_file.read_text())
    for f in tp.get("files") or []:
        if f.get("type") == "template":
            digest = f.get("blob_url", "").rsplit("/blobs/", 1)[-1].split("?")[0]
            bf = BLOBS_DIR / f"{digest}.json"
            if bf.exists():
                b = json.loads(bf.read_text())
                return b.get("content") or ""
    return None


def _namespaced_name(m: dict) -> str:
    """Return the display name with namespace for /x models.

    /x/* models are official but carry the "x" namespace (Ollama's
    experimental image models), so their titles should be "x/<name>"
    rather than just "<name>". Library and user models use their own
    conventions (library: just name; user: owner/name via the namespace
    link, with the <title> already using the full path).
    """
    path = m["path"].strip("/")
    if path.startswith("x/"):
        return f"x/{m['name']}"
    return m["name"]


def _has_moe(model_path: str, tags: list[dict] | None = None) -> bool:
    """Check if any tag's model blob metadata has expert_count."""
    slug = model_path.strip("/").replace("/", "__")
    # Check all tags, not just latest
    tag_names = []
    if tags:
        tag_names = [t.get("name", "") for t in tags if t.get("name")]
    if not tag_names:
        tag_names = ["latest"]
    for tag_name in tag_names:
        tp_file = TAG_PAGES_DIR / f"{slug}__{tag_name}.json"
        if not tp_file.exists():
            continue
        tp = json.loads(tp_file.read_text())
        for f in tp.get("files") or []:
            if f.get("type") == "model":
                digest = f.get("blob_url", "").rsplit("/blobs/", 1)[-1].split("?")[0]
                bf = BLOBS_DIR / f"{digest}.json"
                if bf.exists():
                    b = json.loads(bf.read_text())
                    for md in b.get("metadata") or []:
                        if "expert_count" in md.get("key", ""):
                            return True
    return False


def _classify_template(model_path: str) -> str:
    """Classify template type: base, renderer, or jinja.

    - base: has a template blob with real Go template content
    - renderer: no template blob, or template is trivial ({{ .Prompt }})
    - jinja: no template blob but has a system blob (embedded Jinja)
    - base (no template/system): embed/text/code models
    """
    slug = model_path.strip("/").replace("/", "__")
    tp_file = TAG_PAGES_DIR / f"{slug}__latest.json"
    if not tp_file.exists():
        return "base"
    tp = json.loads(tp_file.read_text())
    types = {f.get("type") for f in (tp.get("files") or [])}
    if "template" in types:
        content = _template_blob_content(model_path)
        if content is not None and len(content.strip()) <= 20:
            return "renderer"
        return "base"
    if "system" in types:
        return "jinja"
    # No template, no system — check arch for renderer
    arch = ""
    for md in _model_blob_metadata(model_path):
        if md.get("key") == "general.architecture":
            arch = md.get("value", "")
            break
    if arch in RENDERER_ARCHS:
        return "renderer"
    return "base"


def parse_context_to_tokens(s: str) -> int:
    """Parse a context string like '256K', '1M', '512', '10M', '-' into token count."""
    if not s or s == "-":
        return 0
    s = s.strip().upper()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([KM])?$", s)
    if not m:
        return 0
    num = float(m.group(1))
    unit = m.group(2)
    if unit == "K":
        return int(num * 1024)
    if unit == "M":
        return int(num * 1048576)
    return int(num)


def render_card(
    m: dict,
    tags: list[dict] | None = None,
    ranks: dict | None = None,
    profile_ranks: dict | None = None,
) -> str:
    name = esc(m["name"])
    name_raw = m["name"]
    owner = m.get("owner") or ""
    official = m.get("official", True)
    # /x/* models are official (Ollama's experimental image models) but carry
    # the "x" namespace — display them as "x/model" like user models.
    is_x = m["path"].strip("/").startswith("x/")
    if is_x:
        display_name = f"x/{name}"
    else:
        display_name = f"{owner}/{name}" if owner and not official else name
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
    # Official models: look up by name in sort_ranks.json
    # Profile models: look up by path in profile_ranks (from profile page orderings)
    if official:
        r = (ranks or {}).get(name_raw, {})
    else:
        r = (profile_ranks or {}).get(m["path"], {})
    data_attrs = (
        f'data-popular-rank="{r.get("popular_rank", 9999)}" '
        f'data-newest-rank="{r.get("newest_rank", 9999)}" '
        f'data-oldest-rank="{r.get("oldest_rank", 9999)}" '
        f'data-updated-rank="{r.get("updated_rank", 9999)}" '
        f'data-pulls="{m["pulls"]}" '
        f'data-tag-count="{tag_count}" '
        f'data-sizes-count="{len(m["sizes"])}" '
        f'data-sizes="{" ".join(m.get("sizes", []))}" '
        f'data-context="{max((parse_context_to_tokens(t.get("context", "-")) for t in m.get("tags", [])), default=0)}" '
        f'data-name="{esc(name_raw).lower()}" '
        f'data-path="{esc(m["path"].strip("/").lower())}" '
        f'data-cloud="{str(m.get("cloud", False)).lower()}" '
        f'data-cloud-only="{str(m.get("cloud_only", False)).lower()}" '
        f'data-official="{str(m.get("official", True)).lower()}" '
        f'data-audio="{"true" if "audio" in (m.get("capabilities") or []) else "false"}" '
        f'data-mlx="{"true" if any((t.get("format") == "mlx") for t in (m.get("tags") or [])) else "false"}" '
        f'data-moe="{"true" if _has_moe(m["path"], m.get("tags")) else "false"}" '
        f'data-mtp="{"true" if any("-mtp" in (t.get("name", "").lower()) for t in (m.get("tags") or [])) else "false"}" '
        f'data-image="{"true" if "image" in (m.get("capabilities") or []) else "false"}" '
        f'data-template-type="{_classify_template(m["path"])}"'
    )

    # MLX pill for models that have MLX variants (black bg, white text, same size as other pills)
    fmt_chip = ""
    if tags and has_mlx(tags):
        fmt_chip = (
            '<span class="inline-flex items-center rounded-md '
            "bg-neutral-900 px-2 py-0.5 text-xs font-medium text-white "
            'dark:bg-white dark:text-neutral-900 sm:text-[13px]">MLX</span>'
        )

    # NEW badge for models that are new this scrape run
    new_badge = ""
    if m["path"] in _NEW_MODELS:
        new_badge = (
            '<span class="ml-2 inline-flex items-center rounded-full '
            "bg-[#ddf4ff] dark:bg-blue-950/50 px-2 py-0.5 text-xs font-medium "
            'text-blue-600 dark:text-blue-400 sm:text-[13px]">NEW</span>'
        )

    return f"""  <li x-test-model {data_attrs} class="flex items-baseline border-b border-neutral-200 dark:border-neutral-800 py-6">
  <a href="{href}" class="group w-full">
    <div class="flex flex-col mb-1" title="{esc(display_name)}">
      <div class="flex items-center min-w-0">
        <h2 class="truncate text-xl font-medium underline-offset-2 group-hover:underline md:text-2xl dark:text-neutral-100">
          <span x-test-search-response-title>{esc(display_name)}</span>
        </h2>
        {new_badge}
      </div>
      <p class="max-w-lg break-words text-neutral-800 dark:text-neutral-300 text-md">{desc}</p>
    </div>
    <div class="flex flex-col">
      <div class="flex flex-wrap items-center space-x-2">
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
    # Load profile-specific ranks (keyed by path, not name)
    profile_ranks = load_profile_ranks()

    # Augment ranks with sort orders the scraper does not provide.
    # updated_rank: most-recent tag update, descending (newest update = 0)
    # oldest_rank: model creation, ascending (oldest model = 0)
    updated_order = sorted(
        models,
        key=lambda m: _parse_updated_title(m.get("updated_title") or ""),
        reverse=True,
    )
    for rank, m in enumerate(updated_order):
        r = profile_ranks if not m.get("official", True) else ranks
        key = m["path"] if not m.get("official", True) else m["name"]
        r.setdefault(key, {})["updated_rank"] = rank
    oldest_order = sorted(
        models,
        key=lambda m: (
            (profile_ranks if not m.get("official", True) else ranks)
            .get(m["path"] if not m.get("official", True) else m["name"], {})
            .get("newest_rank", 9999)
            == 9999,
            -(profile_ranks if not m.get("official", True) else ranks)
            .get(m["path"] if not m.get("official", True) else m["name"], {})
            .get("newest_rank", 9999),
        ),
    )
    for rank, m in enumerate(oldest_order):
        r = profile_ranks if not m.get("official", True) else ranks
        key = m["path"] if not m.get("official", True) else m["name"]
        nr = r.get(key, {}).get("newest_rank", 9999)
        r.setdefault(key, {})["oldest_rank"] = 9999 if nr == 9999 else rank

    sorted_models = sorted(
        models,
        key=lambda m: (
            not m.get("official", True),
            (profile_ranks if not m.get("official", True) else ranks)
            .get(m["path"] if not m.get("official", True) else m["name"], {})
            .get("popular_rank", 9999),
        ),
    )
    cards = "\n".join(
        render_card(m, load_tags(m["path"], m), ranks, profile_ranks)
        for m in sorted_models
    )

    # Capability filter chips (Embedding/Vision/Tools/Thinking only).
    # Image/MLX/MTP/Audio live exclusively inside the More dropdown.
    chip_labels = ["Embedding", "Vision", "Tools", "Thinking"]
    chip_values = ["embedding", "vision", "tools", "thinking"]
    chip_ids = ["cap-embedding", "cap-vision", "cap-tools", "cap-thinking"]
    chip_classes = ["cap-filter", "cap-filter", "cap-filter", "cap-filter"]
    chip_data_attrs = [
        'data-cap="embedding"',
        'data-cap="vision"',
        'data-cap="tools"',
        'data-cap="thinking"',
    ]
    chips = []
    for label, val, cid, cls, dattr in zip(
        chip_labels, chip_values, chip_ids, chip_classes, chip_data_attrs
    ):
        chips.append(
            f"""        <div class="relative inline-block">
          <input type="checkbox" name="c" value="{val}" id="{cid}" class="peer sr-only {cls}" {dattr}>
          <label for="{cid}" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 focus:outline-none focus:ring-0 focus:ring-transparent min-md:hover:bg-neutral-100 dark:min-md:hover:bg-neutral-800 select-none">{label}</label>
        </div>"""
        )
    chips_html = "\n".join(chips)

    # Size dropdown: popup button in narrow mode, inline panel in wide mode (CSS controls)
    size_dropdown = """        <div class="relative inline-block">
         <button id="size-filter-btn" type="button" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center select-none hover:bg-neutral-50 dark:hover:bg-neutral-900">
          Size
        </button>
          <div id="size-filter-panel" class="hidden absolute z-50 bg-white dark:bg-black border border-neutral-200 dark:border-neutral-800 rounded-3xl pl-4 pr-3 py-2 shadow-lg md:left-0 md:translate-x-0" style="width: calc(100vw - 1rem); max-width: 420px;">
           <!-- Slider + Reset side by side -->
           <div class="flex items-center gap-3">
             <!-- Slider area (flex-1) -->
             <div class="relative flex-1" id="size-slider-container" style="height: 42px;">
               <!-- Tick labels (top row) -->
               <div class="relative" style="height: 20px;">
                 <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400" style="left: 0%; transform: translateX(-25%);" data-tick="0">&lt;1b</button>
                 <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 20%;" data-tick="6">6b</button>
                 <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 40%;" data-tick="12">12b</button>
                 <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 60%;" data-tick="32">32b</button>
                 <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 80%;" data-tick="128">128b</button>
                 <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400" style="left: 100%; transform: translateX(-75%);" data-tick="500">&gt;500b</button>
               </div>
               <!-- Pill track + handles -->
               <div class="relative" style="height: 22px;">
                 <!-- Background pill -->
                 <div class="absolute left-0 right-0 rounded-full bg-neutral-200" id="size-slider-track" style="top: 50%; transform: translateY(-50%); height: 6px; background-color: #e5e5e5;"></div>
                 <!-- Filled portion -->
                 <div id="size-slider-fill" class="absolute rounded-full bg-cyan-500 dark:bg-cyan-950" style="top: 50%; transform: translateY(-50%); height: 6px; left: 0%; width: 100%;"></div>
                 <!-- Dots inside pill at breakpoint positions -->
                 <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 20%;"></div>
                 <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 40%;"></div>
                 <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 60%;"></div>
                 <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 80%;"></div>
                 <!-- Min handle -->
                 <button type="button" id="size-handle-min" class="absolute z-10 cursor-pointer touch-none" style="top: 50%; left: 0%; transform: translate(-50%, -50%); width: 10px; height: 10px;">
                   <div class="rounded-full bg-cyan-500 dark:bg-cyan-950" style="width: 10px; height: 10px;"></div>
                   <span id="size-min-tooltip" class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 whitespace-nowrap rounded border px-1 py-0.5 text-xs bg-neutral-100 text-neutral-900 dark:bg-cyan-950 dark:text-white dark:border-cyan-800 hidden">&lt; 1b</span>
                 </button>
                 <!-- Max handle -->
                 <button type="button" id="size-handle-max" class="absolute z-10 cursor-pointer touch-none" style="top: 50%; left: 100%; transform: translate(-50%, -50%); width: 10px; height: 10px;">
                   <div class="rounded-full bg-cyan-500 dark:bg-cyan-950" style="width: 10px; height: 10px;"></div>
                   <span id="size-max-tooltip" class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 whitespace-nowrap rounded border px-1 py-0.5 text-xs bg-neutral-100 text-neutral-900 dark:bg-cyan-950 dark:text-white dark:border-cyan-800 hidden">&gt; 500b</span>
                 </button>
               </div>
               <!-- Hidden range inputs for value storage -->
               <input type="range" id="size-min" min="0" max="500" value="0" step="1" class="sr-only">
               <input type="range" id="size-max" min="0" max="500" value="500" step="1" class="sr-only">
             </div>
             <!-- Reset pill to the right of the slider -->
             <a id="size-filter-reset" class="text-sm text-red-500 dark:text-red-400 hover:text-white hover:bg-red-500 dark:hover:bg-red-950 dark:hover:text-red-200 cursor-pointer rounded-full px-3 py-1 bg-transparent transition-colors shrink-0">Reset</a>
           </div>
         </div>
       </div>"""

    # Context dropdown: popup button in narrow mode, inline panel in wide mode (CSS controls)
    context_dropdown = """        <div class="relative inline-block">
          <button id="context-filter-btn" type="button" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center select-none hover:bg-neutral-50 dark:hover:bg-neutral-900">
           Context
         </button>
          <div id="context-filter-panel" class="hidden absolute z-50 bg-white dark:bg-black border border-neutral-200 dark:border-neutral-800 rounded-3xl pl-4 pr-3 py-2 shadow-lg md:left-0 md:translate-x-0" style="width: calc(100vw - 1rem); max-width: 420px;">
            <!-- Slider + Reset side by side -->
            <div class="flex items-center gap-3">
              <!-- Slider area (flex-1) -->
              <div class="relative flex-1" id="context-slider-container" style="height: 42px;">
                <!-- Tick labels (top row) — positioned by log2 value -->
                <div class="relative" style="height: 20px;">
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400" style="left: 0%; transform: translateX(-25%);" data-ctx-tick="0">0</button>
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 11.11%;" data-ctx-tick="4096">4K</button>
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 22.22%;" data-ctx-tick="8192">8K</button>
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 33.33%;" data-ctx-tick="16384">16K</button>
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 44.44%;" data-ctx-tick="32768">32K</button>
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 55.56%;" data-ctx-tick="65536">64K</button>
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 66.67%;" data-ctx-tick="131072">128K</button>
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 77.78%;" data-ctx-tick="262144">256K</button>
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 88.89%;" data-ctx-tick="524288">512K</button>
                  <button type="button" class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400" style="left: 100%; transform: translateX(-75%);" data-ctx-tick="1048576">&gt;1M</button>
                </div>
                <!-- Pill track + handles -->
                <div class="relative" style="height: 22px;">
                  <!-- Background pill -->
                  <div class="absolute left-0 right-0 rounded-full bg-neutral-200" id="context-slider-track" style="top: 50%; transform: translateY(-50%); height: 6px; background-color: #e5e5e5;"></div>
                  <!-- Filled portion -->
                  <div id="context-slider-fill" class="absolute rounded-full bg-cyan-500 dark:bg-cyan-950" style="top: 50%; transform: translateY(-50%); height: 6px; left: 0%; width: 100%;"></div>
                  <!-- Dots inside pill at breakpoint positions -->
                  <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 11.11%;"></div>
                  <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 22.22%;"></div>
                  <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 33.33%;"></div>
                  <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 44.44%;"></div>
                  <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 55.56%;"></div>
                  <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 66.67%;"></div>
                  <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 77.78%;"></div>
                  <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 88.89%;"></div>
                  <!-- Min handle -->
                  <button type="button" id="context-handle-min" class="absolute z-10 cursor-pointer touch-none" style="top: 50%; left: 0%; transform: translate(-50%, -50%); width: 10px; height: 10px;">
                    <div class="rounded-full bg-cyan-500 dark:bg-cyan-950" style="width: 10px; height: 10px;"></div>
                    <span id="context-min-tooltip" class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 whitespace-nowrap rounded border px-1 py-0.5 text-xs bg-neutral-100 text-neutral-900 dark:bg-cyan-950 dark:text-white dark:border-cyan-800 hidden">0</span>
                  </button>
                  <!-- Max handle -->
                  <button type="button" id="context-handle-max" class="absolute z-10 cursor-pointer touch-none" style="top: 50%; left: 100%; transform: translate(-50%, -50%); width: 10px; height: 10px;">
                    <div class="rounded-full bg-cyan-500 dark:bg-cyan-950" style="width: 10px; height: 10px;"></div>
                    <span id="context-max-tooltip" class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 whitespace-nowrap rounded border px-1 py-0.5 text-xs bg-neutral-100 text-neutral-900 dark:bg-cyan-950 dark:text-white dark:border-cyan-800 hidden">&gt; 1M</span>
                  </button>
                </div>
                <!-- Hidden range inputs for value storage -->
                <input type="range" id="context-min" min="0" max="1048576" value="0" step="1" class="sr-only">
                <input type="range" id="context-max" min="0" max="1048576" value="1048576" step="1" class="sr-only">
              </div>
              <!-- Reset pill to the right of the slider -->
              <a id="context-filter-reset" class="text-sm text-red-500 dark:text-red-400 hover:text-white hover:bg-red-500 dark:hover:bg-red-950 dark:hover:text-red-200 cursor-pointer rounded-full px-3 py-1 bg-transparent transition-colors shrink-0">Reset</a>
            </div>
          </div>
        </div>"""

    # More dropdown: popup button in narrow mode. In wide mode, JS moves
    # #more-pills into the Filters row, #arch-content into #arch-section,
    # and #tpl-content into #tpl-section. CSS hides the More button in wide mode.
    more_dropdown = """        <div class="relative inline-block">
         <button id="more-filter-btn" type="button" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center select-none hover:bg-neutral-50 dark:hover:bg-neutral-900">
          More
        </button>
        <div id="more-filter-panel" class="hidden absolute z-50 bg-white dark:bg-black border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-lg md:left-0 md:translate-x-0" style="width: max-content; max-width: calc(100vw - 1rem);">
          <div class="flex flex-col p-4" style="gap: 12px;" id="more-content">
            <!-- Row 1: Audio + Image + MLX + MTP left-aligned -->
            <div class="flex gap-1.5" id="more-pills">
              <div class="relative inline-block">
                <input type="checkbox" id="more-audio" class="more-filter peer sr-only" data-more="audio">
                <label for="more-audio" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">Audio</label>
              </div>
              <div class="relative inline-block">
                <input type="checkbox" id="more-image" class="more-filter peer sr-only" data-more="image">
                <label for="more-image" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">Image</label>
              </div>
              <div class="relative inline-block">
                <input type="checkbox" id="more-mlx" class="more-filter peer sr-only" data-more="mlx">
                <label for="more-mlx" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">MLX</label>
              </div>
              <div class="relative inline-block">
                <input type="checkbox" id="more-mtp" class="more-filter peer sr-only" data-more="mtp">
                <label for="more-mtp" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">MTP</label>
              </div>
            </div>
            <!-- Row 2: Architecture (All / Dense / MoE) -->
            <div id="arch-content">
              <div class="text-xs text-neutral-500 dark:text-neutral-400 mb-1.5">Architecture</div>
              <div class="flex flex-wrap gap-1.5">
                <div class="relative inline-block">
                  <input type="radio" name="moe-filter" value="all" id="moe-all" class="moe-radio peer sr-only" checked>
                  <label for="moe-all" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">All</label>
                </div>
                <div class="relative inline-block">
                  <input type="radio" name="moe-filter" value="dense" id="moe-dense" class="moe-radio peer sr-only">
                  <label for="moe-dense" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">Dense</label>
                </div>
                <div class="relative inline-block">
                  <input type="radio" name="moe-filter" value="moe" id="moe-moe" class="moe-radio peer sr-only">
                  <label for="moe-moe" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">MoE</label>
                </div>
              </div>
            </div>
            <!-- Row 3: Template (at bottom) -->
            <div id="tpl-content">
              <div class="text-xs text-neutral-500 dark:text-neutral-400 mb-1.5">Template</div>
              <div class="flex flex-wrap gap-1.5">
                <div class="relative inline-block">
                  <input type="radio" name="tpl-filter" value="all" id="tpl-all" class="tpl-radio peer sr-only" checked>
                  <label for="tpl-all" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">All</label>
                </div>
                <div class="relative inline-block">
                  <input type="radio" name="tpl-filter" value="base" id="tpl-base" class="tpl-radio peer sr-only">
                  <label for="tpl-base" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">Base Template</label>
                </div>
                <div class="relative inline-block">
                  <input type="radio" name="tpl-filter" value="renderer" id="tpl-renderer" class="tpl-radio peer sr-only">
                  <label for="tpl-renderer" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">Renderer/Parser</label>
                </div>
                <div class="relative inline-block">
                  <input type="radio" name="tpl-filter" value="jinja" id="tpl-jinja" class="tpl-radio peer sr-only">
                  <label for="tpl-jinja" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">Jinja</label>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>"""

    # Architecture and Template are now inside more_dropdown (same as main branch).
    # These empty variables keep the template working.
    arch_html = ""
    template_html = ""

    # Cloud dropdown: All models / Cloud only / Local only (compact, same as original)
    cloud_dropdown = """        <select id="cloud-filter" class="px-3 py-1 text-sm font-medium rounded-full cursor-pointer border border-neutral-200 text-neutral-800 dark:text-neutral-300 dark:border-neutral-800 bg-white dark:bg-neutral-950 focus:outline-none focus:ring-0 appearance-none">
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

    # --- Graph panel data: KV cache memory curves for ALL models ---
    # Computed at build time by memcalc from scraped blob metadata. The result
    # is written to public/assets/graph-data.json and referenced by the page
    # via window.GRAPH_DATA_URL. Any unexpected failure degrades to an empty
    # graph so the build never breaks.
    graph = {"ticks": GRAPH_TICKS, "models": {}, "by_path": {}}
    try:
        from memcalc.parse import extract_hparams as _extract_hparams
        from memcalc.dispatch import compute_memory_at_context as _compute_mem

        _blobs_dir = HERE / "scraper" / "blobs"
        # Per-digest hparams cache so duplicate digests (e.g. "latest") are
        # parsed at most once.  Sentinel: missing file -> None, parse error
        # -> None; a successful parse -> the hparams dict.
        _hp_cache: dict[str, dict | None] = {}

        def _resolve_blob_digest(model_path: str, tag_name: str) -> str | None:
            """Resolve a (model_path, tag_name) to a blob digest via the tag page."""
            mp = model_path.strip("/").replace("/", "__")
            tp = TAG_PAGES_DIR / f"{mp}__{tag_name}.json"
            if not tp.exists():
                return None
            try:
                tp_data = json.loads(tp.read_text())
            except Exception:
                return None
            for f in tp_data.get("files", []):
                if f.get("type") == "model" and f.get("blob_url"):
                    return f["blob_url"].rsplit("/", 1)[-1]
            return None

        def _get_hparams(digest: str) -> dict | None:
            """Return cached hparams for a digest, parsing the blob on first use."""
            if digest in _hp_cache:
                return _hp_cache[digest]
            bp = _blobs_dir / f"{digest}.json"
            hp: dict | None = None
            if bp.exists():
                try:
                    hp = _extract_hparams(json.loads(bp.read_text()))
                except Exception:
                    hp = None
            _hp_cache[digest] = hp
            return hp

        def _kv_gib(hp: dict, ctx: int) -> float | None:
            """kv_gib at a context, or None if the family is "none"/error."""
            try:
                r = _compute_mem(hp, ctx, 2.0)
            except Exception:
                return None
            if r.get("family") == "none":
                return None
            return round(r["kv_bytes"] / 1073741824, 4)

        _stats_tags = 0
        _stats_skipped = 0
        for m in models:
            if m.get("cloud_only"):
                continue
            model_key = m["name"].lower()
            model_path = m.get("path", "")
            sizes_list = m.get("sizes") or []
            tags_out: dict[str, dict] = {}
            # Dedupe within a model by curve values: quantization doesn't
            # change KV-cache geometry, so identical v arrays are one curve.
            seen: dict[tuple, str] = {}
            max_c = 0
            for tag in m.get("tags", []):
                tname = tag.get("name", "")
                if tname == "cloud" or tname.endswith("-cloud"):
                    _stats_skipped += 1
                    continue
                c = parse_context_to_tokens(tag.get("context", "-"))
                if c <= 0:
                    _stats_skipped += 1
                    continue
                digest = _resolve_blob_digest(model_path, tname)
                if not digest:
                    _stats_skipped += 1
                    continue
                hp = _get_hparams(digest)
                if not hp:
                    _stats_skipped += 1
                    continue
                # Build the value array: kv_gib at every tick strictly < c,
                # then a final endpoint at ctx=c.
                v: list[float] = []
                ok = True
                for t in GRAPH_TICKS:
                    if t >= c:
                        break
                    # Tick 0: compute at ctx=0; on error fall back to ctx=1.
                    if t == 0:
                        g = _kv_gib(hp, 0)
                        if g is None:
                            g = _kv_gib(hp, 1)
                    else:
                        g = _kv_gib(hp, t)
                    if g is None:
                        ok = False
                        break
                    v.append(g)
                if ok:
                    # Final endpoint at ctx=c.
                    g = _kv_gib(hp, c)
                    if g is None:
                        ok = False
                    else:
                        v.append(g)
                if not ok:
                    _stats_skipped += 1
                    continue
                v_tuple = tuple(v)
                if v_tuple in seen:
                    # Same curve already kept under another tag name.
                    # Prefer: non-"latest", a plain size name, shorter name.
                    kept = seen[v_tuple]

                    def _rank(nm: str) -> tuple:
                        return (
                            nm == "latest",
                            nm not in sizes_list and not nm[0].isdigit(),
                            len(nm),
                        )

                    if _rank(tname) < _rank(kept):
                        del tags_out[kept]
                        tags_out[tname] = {"c": c, "v": v}
                        seen[v_tuple] = tname
                    continue
                seen[v_tuple] = tname
                tags_out[tname] = {"c": c, "v": v}
                _stats_tags += 1
                if c > max_c:
                    max_c = c
            if tags_out:
                sorted_tags = dict(
                    sorted(tags_out.items(), key=lambda kv: (kv[1]["c"], kv[0]))
                )
                # Merge: multiple models can share the same name (e.g. /library/lfm2
                # with 24b tags and /maternion/lfm2 with 8b tags). Union their tags
                # into one graph entry rather than overwriting.
                if model_key in graph["models"]:
                    existing = graph["models"][model_key]
                    existing_seen = {
                        tuple(t["v"]): tn for tn, t in existing["tags"].items()
                    }
                    for tn, td in sorted_tags.items():
                        vt = tuple(td["v"])
                        if vt in existing_seen:
                            # Dedupe by curve; prefer shorter/non-latest name.
                            kept = existing_seen[vt]

                            def _rank2(nm: str) -> tuple:
                                return (
                                    nm == "latest",
                                    nm[0].isdigit() is False,
                                    len(nm),
                                )

                            if _rank2(tn) < _rank2(kept):
                                existing["tags"].pop(kept, None)
                                existing["tags"][tn] = td
                                existing_seen[vt] = tn
                            continue
                        existing["tags"][tn] = td
                        existing_seen[vt] = tn
                    existing["tags"] = dict(
                        sorted(
                            existing["tags"].items(), key=lambda kv: (kv[1]["c"], kv[0])
                        )
                    )
                    existing["ctx"] = max(existing["ctx"], max_c)
                else:
                    graph["models"][model_key] = {"ctx": max_c, "tags": sorted_tags}
                # Path-keyed entry: NOT merged — each model path keeps exactly
                # its own tags.  Used by detail pages so /maternion/lfm2 shows
                # only its own 8b tags, not the 24b curve from /library/lfm2.
                # NOTE: must copy sorted_tags — the name-merge below mutates the
                # dict stored in graph["models"][model_key] in-place, and that
                # same object is the sorted_tags reference for the first model
                # with a given name.
                path_key = (m.get("path") or "").strip("/").lower()
                if path_key:
                    graph["by_path"][path_key] = {
                        "ctx": max_c,
                        "tags": dict(sorted_tags),
                    }
        # Sort models dict by key for deterministic output.
        graph["models"] = dict(sorted(graph["models"].items()))
        graph["by_path"] = dict(sorted(graph["by_path"].items()))
        print(
            f"graph-data: {len(graph['models'])} models, "
            f"{_stats_tags} tags included, {_stats_skipped} tags skipped"
        )
    except Exception:
        graph = {"ticks": GRAPH_TICKS, "models": {}, "by_path": {}}

    # Write the graph data JSON (compact) to public/assets/graph-data.json.
    (PUBLIC / "assets").mkdir(parents=True, exist_ok=True)
    (PUBLIC / "assets" / "graph-data.json").write_text(
        json.dumps(graph, separators=(",", ":"))
    )

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html("Ollama", "Search for models on Ollama.")}
    <script>var q = new URLSearchParams(window.location.search).get('q'); if (q) {{ document.title = q + ' \u00b7 Ollama'; }}</script>
    <script>document.documentElement.classList.add('js-init')</script>
  </head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html("models")}

<main class="w-full px-6 py-5 md:py-12 lg:px-8">
  <input type="hidden" id="sort-value" name="o" value="popular">

  <div id="page-wrapper" class="mx-auto">
    <!-- Mobile search bar (shown only below md, where the navbar search is hidden) -->
    <div class="flex md:hidden justify-between space-x-2 items-center mb-2">
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

    <!-- Narrow mode: top row with filters + sort (horizontal) -->
    <!-- Wide mode: sidebar in flow (left), results centered (right), sort absolute -->
    <div id="top-row">
      <div id="filter-container">
        <div class="filter-section" id="caps-section">
          <div class="filter-label text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-2">Filters</div>
          <div class="flex flex-wrap items-center gap-1.5" id="caps-row">
{chips_html}
{cloud_dropdown}
          </div>
        </div>
        <div class="filter-section" id="size-section">
          <div class="filter-label text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-2">Size</div>
{size_dropdown}
        </div>
        <div class="filter-section" id="context-section">
          <div class="filter-label text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-2">Context</div>
{context_dropdown}
        </div>
        <div class="filter-section" id="more-section">
          <div class="filter-label text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-2">More</div>
{more_dropdown}
        </div>
        <div class="filter-section" id="arch-section" style="display:none">
          <div class="filter-label text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-2">Architecture</div>
          <div id="arch-target"></div>
        </div>
        <div class="filter-section" id="tpl-section" style="display:none">
          <div class="filter-label text-xs font-medium text-neutral-500 dark:text-neutral-400 mb-2">Template</div>
          <div id="tpl-target"></div>
        </div>
      </div>
      <!-- Sort dropdown (narrow mode: inline with pills; wide mode: JS moves to results-area) -->
      <div id="sort-container">
        <div class="hidden sm:block shrink-0">
          <select id="desktop-sort-select" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:ring focus:outline-none focus:ring-blue-300 focus:ring-opacity-75 focus:border-blue-400 dark:focus:border-blue-600 min-w-[120px] text-sm px-3 py-1.5">
{opt_html}
          </select>
        </div>
      </div>
    </div>

    <!-- Results area: in wide mode this is a flex item that centers the model list -->
    <div id="results-area">
      <div id="searchresults" class="w-full space-y-2">
        <ul role="list" id="card-list" class="grid grid-cols-1">
{cards}
        </ul>
        <p id="no-results" class="hidden py-12 text-center text-neutral-400 dark:text-neutral-600">No models found.</p>
      </div>
    </div>

    <!-- Graph panel: KV cache memory vs context length -->
    <div id="graph-panel">
      <div class="flex items-center justify-between mb-3">
        <div id="graph-subtitle" class="text-sm font-semibold text-neutral-700 dark:text-neutral-300">Models in view</div>
        <div class="flex items-center gap-1.5">
          <button type="button" id="graph-hide-toggle" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:outline-none text-xs px-2 py-1">Hide graph</button>
          <button type="button" id="graph-filters-toggle" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:outline-none text-xs px-2 py-1">Hide filters</button>
          <span class="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 text-xs px-2 py-1">Context vs Memory</span>
        </div>
      </div>
      <svg id="graph-svg" viewBox="0 0 560 360" preserveAspectRatio="xMidYMid meet" class="w-full"></svg>
      <div id="graph-range-container" class="relative" style="height: 42px;">
        <div class="relative" style="height: 22px;">
          <div class="absolute left-0 right-0 rounded-full" id="graph-range-track" style="top: 50%; transform: translateY(-50%); height: 6px; background-color: #e5e5e5;"></div>
          <div id="graph-range-fill" class="absolute rounded-full bg-cyan-500 dark:bg-cyan-950" style="top: 50%; transform: translateY(-50%); height: 6px; left: 0%; width: 100%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 7.692%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 15.385%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 23.077%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 30.769%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 38.462%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 46.154%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 53.846%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 61.538%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 69.231%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 76.923%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 84.615%;"></div>
          <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 92.308%;"></div>
          <button type="button" id="graph-range-handle" class="absolute z-10 cursor-pointer touch-none" style="top: 50%; left: 100%; transform: translate(-50%, -50%); width: 10px; height: 10px;">
            <div class="rounded-full bg-cyan-500 dark:bg-cyan-950" style="width: 10px; height: 10px;"></div>
          </button>
        </div>
        <div class="relative" style="height: 20px;">
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400" style="left: 0%; transform: translateX(-25%);">0</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 7.692%;">4K</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 15.385%;">8K</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 23.077%;">16K</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 30.769%;">32K</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 38.462%;">64K</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 46.154%;">128K</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 53.846%;">256K</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 61.538%;">512K</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 69.231%;">1M</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 76.923%;">2M</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 84.615%;">4M</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 92.308%;">8M</span>
          <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400" style="left: 100%; transform: translateX(-75%);">Full</span>
        </div>
      </div>
      <div id="graph-legend-row" class="flex items-center justify-between mt-2">
        <div id="graph-legend" class="flex flex-wrap gap-x-3 gap-y-1 text-xs"></div>
        <div id="graph-toggles" class="flex gap-1 shrink-0 ml-2">
          <button type="button" id="graph-logy" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:outline-none text-xs px-2 py-1">logY</button>
          <button type="button" id="graph-all" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:outline-none text-xs px-2 py-1">All</button>
          <button type="button" id="graph-none" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:outline-none text-xs px-2 py-1">None</button>
        </div>
      </div>
      <div id="graph-tooltip" class="hidden fixed pointer-events-none z-50 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2.5 py-1.5 text-xs shadow-lg"></div>
    </div>
  </div>
</main>

{footer_html()}
{theme_script()}
<script>window.GRAPH_DATA_URL = "{url("/assets/graph-data.json")}";</script>
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
    p = prefix
    return f"""<div class="flex flex-wrap gap-2 mb-4">
  <div class="relative inline-block">
    <input type="radio" name="fmt-{p}" value="all" id="fmt-{p}-all" class="peer sr-only fmt-radio" data-fmt="all" checked>
    <label for="fmt-{p}-all" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 dark:border-neutral-800 text-neutral-800 dark:text-neutral-300 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">All ({all_count})</label>
  </div>
  <div class="relative inline-block">
    <input type="radio" name="fmt-{p}" value="gguf" id="fmt-{p}-gguf" class="peer sr-only fmt-radio" data-fmt="gguf">
    <label for="fmt-{p}-gguf" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 dark:border-neutral-800 text-neutral-800 dark:text-neutral-300 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">GGUF ({gguf_count})</label>
  </div>
  <div class="relative inline-block">
    <input type="radio" name="fmt-{p}" value="mlx" id="fmt-{p}-mlx" class="peer sr-only fmt-radio" data-fmt="mlx">
    <label for="fmt-{p}-mlx" class="px-3 py-1 text-sm font-medium rounded-3xl cursor-pointer text-center border border-neutral-200 dark:border-neutral-800 text-neutral-800 dark:text-neutral-300 inline-flex items-center justify-center peer-checked:bg-neutral-100 dark:peer-checked:bg-neutral-800 select-none">MLX ({mlx_count})</label>
  </div>
</div>"""


def _detail_tag_rows(
    tags_subset: list[dict],
    model_path: str,
    latest_digest: str = "",
    show_mlx_badge: bool = False,
    link_tags: bool = True,
) -> str:
    """Render tag rows (mobile + desktop) for the detail page Models table.

    link_tags=False renders tag names as plain text (no hyperlink), matching
    ollama.com's /x model pages where tags are not clickable.
    """
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
        tag_link = _tag_url(model_path, t["name"])
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
            usage_text = (
                f"{' '.join(w.capitalize() for w in usage_level.split())} Usage · "
            )
        if link_tags:
            tag_name_cell_mobile = f'<p class="block group-hover:underline text-sm font-medium text-neutral-800 dark:text-neutral-200">{full_tag_esc}</p>'
            tag_name_cell_desktop = f'<a href="{tag_link}" class="block group-hover:underline text-sm font-medium text-neutral-800 dark:text-neutral-200">{full_tag_esc}</a>'
        else:
            tag_name_cell_mobile = f'<p class="block text-sm font-medium text-neutral-800 dark:text-neutral-200">{full_tag_esc}</p>'
            tag_name_cell_desktop = f'<span class="block text-sm font-medium text-neutral-800 dark:text-neutral-200">{full_tag_esc}</span>'
        rows.append(
            f'      <{'a href="' + tag_link + '"' if link_tags else "div"} class="sm:hidden flex flex-col space-y-[6px] group text-[13px] px-4 py-3">\n'
            f'        <span class="flex items-center">\n'
            f"          {tag_name_cell_mobile}\n"
            f"          {latest_badge}\n"
            f"          {mlx_badge}\n"
            f"        </span>\n"
            f'        <p class="flex text-neutral-500 dark:text-neutral-400">{usage_text}{inline_text}{ctx} context window · {inp} · {updated}</p>\n'
            f"      </{('a' if link_tags else 'div')}>\n"
            f'      <div class="hidden group px-4 py-3 sm:grid sm:grid-cols-12 text-[13px]">\n'
            f'        <span class="col-span-6 flex items-center">\n'
            f"          {tag_name_cell_desktop}\n"
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
    # Sizeless models have no size tags to curate against, so show every tag:
    # "latest" first, then the remaining tags in their original order.
    if not sizes:
        ordered: list[str] = ["latest"] if "latest" in by_name else []
        ordered += [t["name"] for t in tags if t["name"] not in ordered]
        seen = set()
        out = []
        for n in ordered:
            if n in by_name and n not in seen:
                seen.add(n)
                out.append(by_name[n])
        return out
    ordered: list[str] = ["latest"]
    # Base size tags that exist
    ordered += [s for s in sizes if s in by_name]
    # MLX counterparts that exist (only {size}-mlx on the main page;
    # other MLX quants like mlx-mxfp8, mlx-bf16 are on the /tags page)
    ordered += [f"{s}-mlx" for s in sizes if f"{s}-mlx" in by_name]
    # Generic cloud tag
    if "cloud" in by_name:
        ordered.append("cloud")
    # Cloud-only size tags: X-cloud where X is not a downloadable size
    for n in sorted(by_name.keys()):
        if n.endswith("-cloud") and n != "cloud" and n[:-6] not in by_name:
            ordered.append(n)
    # If no size-named tags matched, show all tags (latest first)
    size_matched = any(s in by_name for s in sizes)
    if not size_matched:
        ordered = ["latest"] if "latest" in by_name else []
        ordered += [t["name"] for t in tags if t["name"] not in ordered]
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
        mobile_label = n
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
    # /x model pages don't link tags (matching ollama.com — tags are plain text
    # there, not clickable hyperlinks). Library and user models do link them.
    link_tags = not m["path"].strip("/").startswith("x/")
    rows_all = _detail_tag_rows(
        main, m["path"], latest_digest, show_mlx_badge=True, link_tags=link_tags
    )
    rows_gguf = _detail_tag_rows(
        main_gguf, m["path"], latest_digest, show_mlx_badge=False, link_tags=link_tags
    )
    rows_mlx = _detail_tag_rows(
        main_mlx, m["path"], latest_digest, show_mlx_badge=False, link_tags=link_tags
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
        body = sanitize_readme_html(readme)
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

    new_badge = ""
    if m["path"] in _NEW_MODELS:
        new_badge = (
            '<span class="ml-2 inline-flex items-center rounded-full '
            "bg-[#ddf4ff] dark:bg-blue-950/50 px-2 py-0.5 text-xs font-medium "
            'text-blue-600 dark:text-blue-400">NEW</span>'
        )

    # For user models, prepend the namespace link + "/" separator.
    # /x/* models are official but also use the "x" namespace (Ollama's
    # experimental image models on ollama.com/x), so show it there too.
    namespace_html = ""
    is_x = m["path"].strip("/").startswith("x/")
    if (not m.get("official") and "/" in m["path"].strip("/")) or is_x:
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
        {new_badge}
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
      <div class="flex flex-wrap items-center gap-2">
        {caps}
        {sizes}
      </div>
    </div>
  </div>"""


def _cloud_metrics_section(page_data: dict) -> str:
    """Render the cloud metrics section for cloud models.

    Two layouts depending on data:
    - Cost layout (newer cloud-only models like kimi-k3): Cost /1M tokens with
      input/cached/output prices, plus Context and Size in a 2-col sub-grid.
    - Usage layout (older cloud models): Usage bars + level, Context, Size in
      a 3-col grid.

    Returns empty string if page_data has no cloud metrics.
    """
    usage_level = (page_data.get("cloud_usage_level") or "").strip()
    active_slots = int(page_data.get("cloud_usage_active_slots") or 0)
    ctx = esc(page_data.get("cloud_context") or "")
    ctx_unit = esc(page_data.get("cloud_context_unit") or "")
    size = esc(page_data.get("cloud_size") or "")
    size_unit = esc(page_data.get("cloud_size_unit") or "")
    cost_input = esc(page_data.get("cloud_cost_input") or "")
    cost_cached = esc(page_data.get("cloud_cost_cached") or "")
    cost_output = esc(page_data.get("cloud_cost_output") or "")
    has_cost = bool(cost_input or cost_cached or cost_output)
    has_usage = bool(usage_level or active_slots > 0)
    if not has_cost and not has_usage and not ctx:
        return ""

    # --- Cost layout (new) ---
    if has_cost:
        return f"""<div x-test-model-metrics class="!mt-8 grid grid-cols-3 overflow-hidden rounded-lg border border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
  <div x-test-model-cost x-test-model-metric="usage" class="min-h-24 min-w-0 border-neutral-200 dark:border-neutral-800 px-4 py-3 md:px-5 md:py-4 border-r">
    <div class="flex items-center justify-between gap-2">
      <span class="text-[13px] font-medium text-neutral-500 dark:text-neutral-400">Cost</span>
      <span class="text-xs text-neutral-400 dark:text-neutral-500">/1M tokens</span>
    </div>
    <div class="mt-3 grid grid-cols-1 md:grid-cols-3 gap-1">
      <div class="flex min-w-0 flex-col gap-1 text-left">
        <div class="shrink-0 truncate text-xl font-medium leading-none text-black tabular-nums dark:text-neutral-100">{cost_input}</div>
        <div class="truncate text-xs leading-tight text-neutral-700 dark:text-neutral-300">input</div>
      </div>
      <div class="flex min-w-0 flex-col gap-1 text-left">
        <div class="shrink-0 truncate text-xl font-medium leading-none text-black tabular-nums dark:text-neutral-100">{cost_cached}</div>
        <div class="truncate text-xs leading-tight text-neutral-700 dark:text-neutral-300">cached</div>
      </div>
      <div class="flex min-w-0 flex-col gap-1 text-left">
        <div class="shrink-0 truncate text-xl font-medium leading-none text-black tabular-nums dark:text-neutral-100">{cost_output}</div>
        <div class="truncate text-xs leading-tight text-neutral-700 dark:text-neutral-300">output</div>
      </div>
    </div>
  </div>
  <div x-test-model-metric="context" class="min-h-24 min-w-0 border-neutral-200 dark:border-neutral-800 px-4 py-3 md:px-5 md:py-4 border-r flex flex-col justify-center">
    <div class="text-[13px] font-medium text-neutral-500 dark:text-neutral-400">Context</div>
    <div class="mt-3 flex min-w-0 flex-col gap-1">
      <span class="shrink-0 text-xl font-medium leading-none text-black dark:text-neutral-100">{ctx}</span>
      <span class="min-w-0 break-words text-[13px] leading-tight text-neutral-700 dark:text-neutral-300 sm:text-sm">{ctx_unit}</span>
    </div>
  </div>
  <div x-test-model-metric="size" class="min-h-24 min-w-0 border-neutral-200 dark:border-neutral-800 px-4 py-3 md:px-5 md:py-4 flex flex-col justify-center">
    <div class="text-[13px] font-medium text-neutral-500 dark:text-neutral-400">Size</div>
    <div class="mt-3 flex min-w-0 flex-col gap-1">
      <span class="shrink-0 text-xl font-medium leading-none text-black dark:text-neutral-100">{size}</span>
      <span class="min-w-0 break-words text-[13px] leading-tight text-neutral-700 dark:text-neutral-300 sm:text-sm">{size_unit}</span>
    </div>
  </div>
</div>"""

    # --- Usage layout (old) ---
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
    <div class="text-[13px] font-medium text-neutral-500 dark:text-neutral-400">Usage</div>
    <div class="mt-3 flex min-w-0 flex-col gap-1">
      <div x-test-model-cost-value x-test-model-cost-level class="flex h-5 items-center gap-1">
        {active_bars}{inactive_bars}
      </div>
      <span class="min-w-0 break-words text-xs leading-tight text-neutral-700 dark:text-neutral-300 sm:text-[14px] sm:leading-5">{esc(usage_level)}</span>
    </div>
  </div>
  <div x-test-model-metric="context" class="min-h-24 min-w-0 border-neutral-200 dark:border-neutral-800 px-4 py-3 md:px-5 md:py-4 border-r">
    <div class="text-[13px] font-medium text-neutral-500 dark:text-neutral-400">Context</div>
    <div class="mt-3 flex min-w-0 flex-col gap-1">
      <span class="shrink-0 text-xl font-medium leading-none text-black dark:text-neutral-100">{ctx}</span>
      <span class="min-w-0 break-words text-[13px] leading-tight text-neutral-700 dark:text-neutral-300 sm:text-sm">{ctx_unit}</span>
    </div>
  </div>
  <div x-test-model-metric="size" class="min-h-24 min-w-0 border-neutral-200 dark:border-neutral-800 px-4 py-3 md:px-5 md:py-4">
    <div class="text-[13px] font-medium text-neutral-500 dark:text-neutral-400">Size</div>
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

    # Use just the model name for /library/ models; include namespace for /x/ and profile models
    path = m["path"]
    if path.startswith("/library/"):
        full_name = m["name"]
    else:
        full_name = path.strip("/")
    # For cloud-only models, use the :cloud tag in CLI commands
    if m.get("cloud_only"):
        full_name = f"{full_name}:cloud"
    header = _header_section(m)
    usage = _usage_section(full_name)
    models_section = _detail_models_section(m, tags)

    page_data = load_model_page(m["path"])
    readme_section = _readme_section(page_data) if page_data else ""
    cloud_metrics = _cloud_metrics_section(page_data) if page_data else ""
    apps_section = _applications_section(page_data) if page_data else ""

    graph_key = m["path"].strip("/").lower()

    # Title: official /library models use just the name; user models and /x/*
    # experimental models (official but namespaced) use the full path.
    title = (
        name
        if (m.get("official") and not path.strip("/").startswith("x/"))
        else path.strip("/")
    )

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
  <div id="graph-panel" class="detail-graph">
    <div class="flex items-center justify-between mb-3">
      <div id="graph-subtitle" class="text-sm font-semibold text-neutral-700 dark:text-neutral-300">&nbsp;</div>
    </div>
    <svg id="graph-svg" viewBox="0 0 560 360" preserveAspectRatio="xMidYMid meet" class="w-full"></svg>
    <div id="graph-range-container" class="relative" style="height: 42px;">
      <div class="relative" style="height: 22px;">
        <div class="absolute left-0 right-0 rounded-full" id="graph-range-track" style="top: 50%; transform: translateY(-50%); height: 6px; background-color: #e5e5e5;"></div>
        <div id="graph-range-fill" class="absolute rounded-full bg-cyan-500 dark:bg-cyan-950" style="top: 50%; transform: translateY(-50%); height: 6px; left: 0%; width: 100%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 7.692%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 15.385%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 23.077%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 30.769%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 38.462%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 46.154%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 53.846%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 61.538%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 69.231%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 76.923%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 84.615%;"></div>
        <div class="absolute rounded-full slider-dot" style="top: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; left: 92.308%;"></div>
        <button type="button" id="graph-range-handle" class="absolute z-10 cursor-pointer touch-none" style="top: 50%; left: 100%; transform: translate(-50%, -50%); width: 10px; height: 10px;">
          <div class="rounded-full bg-cyan-500 dark:bg-cyan-950" style="width: 10px; height: 10px;"></div>
        </button>
      </div>
      <div class="relative" style="height: 20px;">
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400" style="left: 0%; transform: translateX(-25%);">0</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 7.692%;">4K</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 15.385%;">8K</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 23.077%;">16K</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 30.769%;">32K</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 38.462%;">64K</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 46.154%;">128K</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 53.846%;">256K</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 61.538%;">512K</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 69.231%;">1M</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 76.923%;">2M</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 84.615%;">4M</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400 -translate-x-1/2" style="left: 92.308%;">8M</span>
        <span class="absolute top-0 select-none whitespace-nowrap px-1 py-0.5 text-xs text-neutral-500 dark:text-neutral-400" style="left: 100%; transform: translateX(-75%);">Full</span>
      </div>
    </div>
    <div id="graph-legend-row" class="flex items-center justify-between mt-2">
      <div id="graph-legend" class="flex flex-wrap gap-x-3 gap-y-1 text-xs"></div>
      <div id="graph-toggles" class="flex gap-1 shrink-0 ml-2">
        <button type="button" id="graph-logy" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:outline-none text-xs px-2 py-1">logY</button>
        <button type="button" id="graph-all" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:outline-none text-xs px-2 py-1">All</button>
        <button type="button" id="graph-none" class="appearance-none cursor-pointer rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800 focus:outline-none text-xs px-2 py-1">None</button>
      </div>
    </div>
    <div id="graph-tooltip" class="hidden fixed pointer-events-none z-50 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2.5 py-1.5 text-xs shadow-lg"></div>
  </div>
</main>

{footer_html()}
{theme_script()}
<script>window.GRAPH_DATA_URL = "{url("/assets/graph-data.json")}"; window.GRAPH_MODEL = {json.dumps(graph_key)}; window.GRAPH_MODEL_TITLE = {json.dumps(title)};</script>
<script src="{url("/assets/app.js")}"></script>
</body>
</html>"""

    (slug_dir / "index.html").write_text(page)


# --------------------------------------------------------------------------- #
# Tags page
# --------------------------------------------------------------------------- #


def _tags_tag_row(
    t: dict,
    model_path: str,
    latest_digest: str = "",
    show_mlx_badge: bool = False,
    link_tags: bool = True,
) -> str:
    model_name = model_path.strip("/").split("/")[-1]
    full_tag_name = f"{model_name}:{t['name']}"
    full_tag_esc = esc(full_tag_name)
    tag_link = _tag_url(model_path, t["name"])
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
        usage_text = f"{' '.join(w.capitalize() for w in usage_level.split())} Usage"
    usage_sep = f"{usage_text} • " if usage_text else ""
    # /x models don't have per-tag detail pages, so render tag names as plain
    # text (no hyperlink) on the tags page too.
    if link_tags:
        mobile_open = (
            f'<a href="{tag_link}" class="md:hidden flex flex-col space-y-[6px] group">'
        )
        mobile_close = "</a>"
        name_mobile = f'<span class="group-hover:underline">{full_tag_esc}</span>'
        name_desktop = (
            f'<a href="{tag_link}" class="group-hover:underline">{full_tag_esc}</a>'
        )
    else:
        mobile_open = '<div class="md:hidden flex flex-col space-y-[6px] group">'
        mobile_close = "</div>"
        name_mobile = f"<span>{full_tag_esc}</span>"
        name_desktop = f"<span>{full_tag_esc}</span>"
    return f"""<div class="group px-4 py-3">
  {mobile_open}
    <div class="flex items-center font-medium">
      <div class="flex items-center justify-between w-full">
        <div>
          {name_mobile}
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
  {mobile_close}
  <div class="hidden md:flex flex-col space-y-[6px]">
    <div class="grid grid-cols-12 items-center">
      <span class="flex items-center font-medium col-span-6 group text-sm">
        {name_desktop}
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

    # /x models don't have per-tag detail pages, so render tags as plain text
    # on the tags page too (no hyperlinks to non-existent tag pages).
    link_tags = not path.strip("/").startswith("x/")
    rows_all = "\n".join(
        _tags_tag_row(t, path, latest_digest, show_mlx_badge=True, link_tags=link_tags)
        for t in tags
    )
    rows_gguf = "\n".join(
        _tags_tag_row(t, path, latest_digest, show_mlx_badge=False, link_tags=link_tags)
        for t in gguf_tags
    )
    rows_mlx = "\n".join(
        _tags_tag_row(t, path, latest_digest, show_mlx_badge=False, link_tags=link_tags)
        for t in mlx_tags
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
{head_html(f"{_namespaced_name(m)} Tags", f"Tags for {_namespaced_name(m)}. {desc}")}
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

    new_badge = ""
    if m["path"] in _NEW_MODELS:
        new_badge = (
            '<span class="ml-2 inline-flex items-center rounded-full '
            "bg-[#ddf4ff] dark:bg-blue-950/50 px-2 py-0.5 text-xs font-medium "
            'text-blue-600 dark:text-blue-400">NEW</span>'
        )

    # For user models, prepend the namespace link + "/" separator.
    # /x/* models are official but also use the "x" namespace (Ollama's
    # experimental image models on ollama.com/x), so show it there too.
    namespace_html = ""
    is_x = m["path"].strip("/").startswith("x/")
    if (not m.get("official") and "/" in m["path"].strip("/")) or is_x:
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
        {new_badge}
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
      <div class="flex flex-wrap items-center gap-2">
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
    # Use just the model name for /library/ models; include namespace for /x/ and profile models
    if path.startswith("/library/"):
        full_name = f"{m['name']}:{tag_name}"
    else:
        full_name = f"{path.strip('/')}:{tag_name}"
    # Output dir: public/library/gemma4:latest/  (colon attached to model name,
    # matching ollama.com's URL scheme — no / before the colon).
    tag_dir = PUBLIC / (path.strip("/") + f":{tag_name}")
    tag_dir.mkdir(parents=True, exist_ok=True)

    header = _tag_header_section(m, tag_name)
    usage = _usage_section(full_name)

    cloud_metrics = _cloud_metrics_section(tp) if tp else ""
    apps_section = _applications_section(tp) if tp else ""

    # The readme is stored once per model (in the model page JSON), not per
    # tag. Load it from the model page so tag pages share the single source.
    if tp and (tp.get("files") or tp.get("manifest_digest")):
        model_page = load_model_page(path)
        details_section = _details_section(tp)
        readme_section = _readme_section(model_page) if model_page else ""
    elif tp:
        # Has tag page data but no files (cloud tag) — skip Details, show readme only
        model_page = load_model_page(path)
        details_section = ""
        readme_section = _readme_section(model_page) if model_page else ""
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
{head_html(f"{_namespaced_name(m)}:{tag_name}", f"{_namespaced_name(m)}:{tag_name} — {desc}")}
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


_MODELS_BY_PATH = None


def _get_models_by_path():
    global _MODELS_BY_PATH
    if _MODELS_BY_PATH is None:
        data = json.loads((SCRAPER / "models.json").read_text())
        _MODELS_BY_PATH = {m["path"]: m for m in data.get("models", [])}
    return _MODELS_BY_PATH


# Digest -> rendered blob body HTML (the metadata/tensors/content block). The
# body is invariant for a given digest; only the back-link/tag header wrapper
# varies per (model, tag), so it is rendered once per digest and reused.
_BLOB_BODY_CACHE: dict[str, str] = {}


def build_blob_page(blob: dict, target_blob_url: str = "") -> None:
    tag_full = blob.get("tag_full") or ""
    blob_url = blob.get("blob_url") or ""
    blob_type = blob.get("blob_type") or ""
    digest = blob.get("digest") or ""
    size = blob.get("size") or ""

    # Blob data is stored once per digest under the canonical blob_url where it
    # was first scraped (e.g. /library/lfm2.5:latest/blobs/<digest>). When two
    # tags share a digest (alias tags, e.g. :8b and :latest), the tag page's
    # file links point to each tag's own path (e.g. /library/lfm2.5:8b/blobs/…)
    # but build_blob_page would otherwise only ever emit a page under the
    # canonical blob_url. Passing target_blob_url overrides the on-disk path /
    # back-link so a blob page is also built at the alias tag's path. Without
    # this, alias tags 404 on their blob pages.
    if target_blob_url:
        blob_url = target_blob_url

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

    blob_dir = (
        PUBLIC / (model_path + f":{tag_name}") / "blobs" / (digest_part or digest)
    )
    blob_dir.mkdir(parents=True, exist_ok=True)

    # Load model data for the header
    model_path_full = "/" + model_path  # e.g. "/library/gpt-oss"
    tag_name_from_blob = tag_name or "latest"

    # Find the model in models.json
    m = None
    try:
        m = _get_models_by_path().get(model_path_full)
    except Exception:
        pass

    header = _tag_header_section(m, tag_name_from_blob) if m else ""

    model_name = model_path.strip("/").split("/")[-1]
    # /x models carry the "x" namespace in their display title.
    if model_path.strip("/").startswith("x/"):
        display_model_name = f"x/{model_name}"
    else:
        display_model_name = model_name
    model_link = url(esc(model_path))
    tag_page_url = _tag_url(model_path, tag_name)
    tag_full_esc = esc(tag_full)
    blob_type_esc = esc(blob_type)
    digest_esc = esc(digest_part or digest)
    size_esc = esc(size)

    # The body (metadata/tensors/content) is digest-invariant — render once per
    # digest and cache it. Only the tag-specific wrapper is rendered fresh.
    cache_key = digest or digest_part
    content_html = _BLOB_BODY_CACHE.get(cache_key)
    if content_html is None:
        content_html = _blob_content_html(blob)
        _BLOB_BODY_CACHE[cache_key] = content_html

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html(f"{display_model_name}:{tag_name} — {blob_type}", f"{display_model_name}:{tag_name} blob {blob_type}")}
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

    # Profile images (download from profile data if available)
    for _uname in ["maternion", "frob", "huihui_ai", "x"]:
        _pf = HERE / "scraper" / f"profile_{_uname}.json"
        if not _pf.exists():
            continue
        _pdata = json.loads(_pf.read_text())
        _avatar = _pdata.get("avatar", "")
        if not _avatar:
            continue
        _avatar_url = (
            _avatar if _avatar.startswith("http") else f"https://ollama.com{_avatar}"
        )
        _ext = ".png" if ".png" in _avatar else ".jpg"
        _name = f"{_uname}-profile{_ext}"
        dst = assets / _name
        if dst.exists():
            continue
        try:
            import urllib.request

            urllib.request.urlretrieve(_avatar_url, dst)
            print(f"  downloaded {_name}")
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
.dark .dark\:hover\:bg-neutral-900:hover { background-color: #171717; }
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
.dark .dark\:placeholder\:text-neutral-500::placeholder { color: #737373; }
.dark .dark\:focus\:bg-white:focus { background-color: #ffffff; }

/* --- Dark: colored badge classes (Tailwind shade inversion) --- */
/* Capability badges: indigo-50 bg → indigo-950, indigo-600 text → indigo-400 */
.dark .dark\:bg-indigo-950\/50 { background-color: rgba(30, 27, 75, 0.5); }
.dark .dark\:text-indigo-400 { color: #818cf8; }
/* Cloud badge: cyan-50 bg → cyan-950, cyan-500 text → cyan-400 */
.dark .dark\:bg-cyan-950\/50 { background-color: rgba(8, 51, 68, 0.5); }
.dark .dark\:bg-cyan-950 { background-color: #083344; }
.dark .dark\:text-cyan-400 { color: #22d3ee; }
.dark .dark\:border-cyan-800 { border-color: #155e75; }
/* Size badges: blue-50 bg → blue-950, blue-600 text → blue-400 */
.dark .dark\:bg-blue-950\/50 { background-color: rgba(23, 37, 84, 0.5); }
.dark .dark\:text-blue-400 { color: #60a5fa; }
/* Tabs: blue-500 border stays blue-500, blue-600 text → blue-400 */
.dark .dark\:border-blue-500 { border-color: #3b82f6; }
/* Focus state */
.dark .dark\:focus\:border-blue-600:focus { border-color: #2563eb; }
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

/* --- Readme / prose ---
   The vendored tailwind.css includes the `.prose` base styles and most
   `prose-*` variant utilities, but is missing a handful used by ollama.com's
   `<div id="display">` (the bracket/precise-value heading margins, the
   `prose-code:text-[85%]` utility, and the
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

/* --- pill active state (purple) for peer-checked and JS-toggled buttons --- */
.dark .peer:checked ~ label { background-color: #1e1b4b !important; border-color: #1e1b4b !important; }

/* --- search preview dropdown --- */
#searchpreview { max-height: 24rem; overflow-y: auto; }
.dark select option { background-color: #0a0a0a; color: #e5e5e5; }
.dark .dark\:text-gray-600 { color: #a3a3a3; }
.dark .dark\:hover\:bg-white\/5:hover { background-color: rgba(255,255,255,0.05); }
 .dark .dark\:focus\:bg-white\/5:focus { background-color: rgba(255,255,255,0.05); }

/* Pricing tier images: invert in dark mode (they're black line art on white) */
.dark img.pricing-tier-img { filter: invert(1); }
/* Pricing price block: fixed min height (min-h-[3rem] not in vendored tailwind) */
.min-h-\[3rem\] { min-height: 3rem; }
/* ml-auto not in vendored tailwind.css */
.ml-auto { margin-left: auto; }
/* Reset button hover bg colors (Tailwind red-50 light, red-950 dark) */
.hover\:bg-red-50:hover { background-color: #fef2f2; }
.hover\:text-white:hover { color: #fff; }
.hover\:bg-red-500:hover { background-color: #ef4444; }
.dark .dark\:text-red-400 { color: #f87171; }
.dark .dark\:hover\:bg-red-950:hover { background-color: #450a0a; }
.dark .dark\:hover\:text-red-200:hover { color: #fecaca; }
/* Reset button click feedback (active state) */
#size-filter-reset:active,
#context-filter-reset:active { background-color: #fca5a5 !important; color: #991b1b !important; }
.dark #size-filter-reset:active,
.dark #context-filter-reset:active { background-color: #7f1d1d !important; color: #fecaca !important; }
/* Pricing: mb-5 not in vendored tailwind.css (teams card price block gap) */
.mb-5 { margin-bottom: 1.25rem; }
/* Pricing tabs: hide panels based on html.tab-* class set by head script (FOUC fix).
   When neither class is present (JS disabled), both panels show — graceful fallback. */
html:not(.tab-teams):not(.tab-individuals) #pricing-individuals,
html:not(.tab-teams):not(.tab-individuals) #pricing-teams { display: block; }
html.tab-individuals #pricing-individuals { display: block; }
html.tab-individuals #pricing-teams { display: none; }
html.tab-teams #pricing-individuals { display: none; }
html.tab-teams #pricing-teams { display: block; }
/* When tab-teams is set, show teams FAQ and hide main FAQ; vice versa for tab-individuals.
   Default (no class): show main FAQ, hide teams FAQ (matches individuals default). */
html.tab-teams #pricing-faq { display: none; }
html.tab-teams #pricing-teams-faq { display: block; }
html:not(.tab-teams) #pricing-teams-faq { display: none; }

/* Force-hide native select arrow (appearance-none not enough on some browsers) */
#cloud-filter {
  -webkit-appearance: none !important;
  -moz-appearance: none !important;
  appearance: none !important;
  text-indent: 1px;
  text-overflow: "";
  background-image: none !important;
}
#cloud-filter::-ms-expand { display: none; }

/* --- Size slider: neutral classes missing from vendored tailwind.css --- */
/* Light mode only — do NOT override any dark: classes */
.bg-cyan-500 { background-color: #cffafe; }
.hover\:bg-neutral-100:hover { background-color: #f5f5f5; }
/* Dark mode slider colors */
/* Slider track: dark mode grey (override the purple !important on dark:bg-neutral-800) */
.dark #size-slider-track,
.dark #context-slider-track,
.dark #graph-range-track { background-color: #262626 !important; }
/* Slider dots: inactive (subtle grey on track) */
.slider-dot { background-color: #d4d4d4; }
.dark .slider-dot { background-color: #404040; }
/* Slider dots: active (inside fill — cyan tint) */
.slider-dot.active { background-color: #06b6d4; }
.dark .slider-dot.active { background-color: #22d3ee; }
/* gap-1.5 missing from vendored tailwind.css */
.gap-1\.5 { gap: 0.375rem; }
/* md:grid-cols-3 missing from vendored tailwind.css */
@media (min-width: 768px) {
  .md\:grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
}
/* peer-checked:bg-neutral-800 missing from vendored tailwind.css */
.peer-checked\:bg-neutral-800 { --tw-bg-opacity: 1; background-color: rgb(38 38 38 / var(--tw-bg-opacity)) !important; }

/* --- Responsive dropdown panel positioning --- */
/* Mobile (<768px): position: fixed relative to viewport so the panel is always
   fully visible (no overflow off either edge). left/right pin it to the viewport
   with 0.5rem gutters; width:auto lets it stretch to fill. top is set via JS
   (viewport-relative, computed from the button's position) when the panel opens.
   Desktop (>=768px): position: absolute relative to the button wrapper; left-0
   aligns it under the button; inline width/max-width apply for natural sizing. */
@media (max-width: 767px) {
  #size-filter-panel,
  #more-filter-panel,
  #context-filter-panel {
    position: fixed !important;
    left: 0.5rem !important;
    right: 0.5rem !important;
    width: auto !important;
    max-width: none !important;
    transform: none !important;
    /* top is supplied via JS (var(--panel-top)); fallback keeps it just under
       the viewport top so it is never off-screen before JS runs. */
    top: var(--panel-top, 4rem);
  }
  /* Mobile: hide reset buttons so slider takes full panel width */
  #size-filter-reset,
  #context-filter-reset { display: none !important; }
}
@media (min-width: 768px) {
  .md\:left-0 { left: 0 !important; }
  .md\:translate-x-0 { --tw-translate-x: 0 !important; transform: translate(0,var(--tw-translate-y)) rotate(var(--tw-rotate)) skewX(var(--tw-skew-x)) skewY(var(--tw-skew-y)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y)) !important; }
  /* On desktop, clear any viewport-relative top that JS set while in mobile view,
     so the inline `top: calc(100% + 6px)` (relative to the button wrapper) applies. */
  #size-filter-panel,
  #more-filter-panel,
  #context-filter-panel {
    top: calc(100% + 6px) !important;
  }
}

/* === Responsive sidebar / horizontal layout switching === */

/* FOUC fix: hide page content until JS initialization completes (layoutFilters + applyFilters) */
.js-init #page-wrapper { visibility: hidden; }

/* Default (narrow): horizontal pills on top, model list at max-w-2xl */
#page-wrapper { max-width: 42rem; } /* max-w-2xl = 672px */
#top-row {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}
#filter-container {
  position: static;
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  row-gap: 0.5rem;
  column-gap: 0.375rem;
}
#results-area {
  position: relative;
  transition: margin-left 0.35s ease;
}
#sort-container {
  position: static;
  flex-shrink: 0;
}
/* Narrow: caps section takes full width (forces Size/More to next row) */
.filter-section { display: block; }
#caps-section { width: 100%; }
#size-section, #context-section, #more-section { display: inline-block; }
.filter-label { display: none; }
#arch-section, #tpl-section { display: none !important; }

/* Wide (>= 1200px): sidebar in flow (left), results centered (right) */
@media (min-width: 1200px) {
  #page-wrapper {
    max-width: calc(420px + 2.5rem + 36rem); /* sidebar + gap + model list */
    margin: 0 auto;
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    gap: 2.5rem; /* 40px gap between sidebar and results */
  }

  /* Sidebar: sticky so it follows scroll, fixed width, vertical stack.
     top: 57px keeps it below the sticky navbar (lg:static so only matters < 1024px,
     but the sidebar only shows >= 1200px so navbar is static there — top: 0 is safe). */
  #top-row {
    display: block;
    margin-bottom: 0;
    flex-shrink: 0;
    width: 420px;
    padding-top: 1rem;
    position: sticky;
    top: 0;
    max-height: 100vh;
    overflow-y: auto;
    align-self: flex-start;
  }
  #filter-container {
    position: static;
    display: block;
    flex: none;
    width: auto;
    flex-wrap: nowrap;
    row-gap: 0;
  }
  /* Wide: sections stack vertically */
  .filter-section { display: block; margin-bottom: 2rem; }
  .filter-label { display: block; }
  /* Override narrow-mode sizing */
  #caps-section { width: auto; }
  #size-section { display: block; }
  #context-section { display: block; }

  /* Hide inner Architecture/Template labels in wide mode (section labels are enough) */
  #arch-content > div:first-child,
  #tpl-content > div:first-child { display: none; }

  /* Hide popup buttons, show panels inline */
  #size-filter-btn { display: none; }
  #context-filter-btn { display: none; }
  #more-filter-btn { display: none; }
  #more-section { display: none !important; }

  /* Size panel: inline, full width */
  #size-filter-panel {
    display: block !important;
    position: static !important;
    width: 100% !important;
    max-width: 100% !important;
    box-shadow: none !important;
    border: 1px solid #e5e5e5;
    border-radius: 1rem;
    margin-top: 0;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
  }
  /* Make the size wrapper full-width block */
  #size-section > .relative.inline-block {
    display: block !important;
    width: 100% !important;
  }
  /* Context panel: inline, full width */
  #context-filter-panel {
    display: block !important;
    position: static !important;
    width: 100% !important;
    max-width: 100% !important;
    box-shadow: none !important;
    border: 1px solid #e5e5e5;
    border-radius: 1rem;
    margin-top: 0;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
  }
  /* Make the context wrapper full-width block */
  #context-section > .relative.inline-block {
    display: block !important;
    width: 100% !important;
  }

  /* arch-section and tpl-section shown by JS */
  #arch-section, #tpl-section { display: none; }
  #arch-section.active, #tpl-section.active { display: block !important; }

  /* Results area: flex-1, centers model list within remaining space */
  #results-area {
    position: relative;
    flex: 1 1 0%;
    min-width: 0;
    max-width: 36rem; /* max-w-xl = 576px */
    margin: 0 auto;
    padding-top: 1rem;
  }

  /* Sort dropdown: absolute, top-right of results area */
  #sort-container {
    position: absolute;
    top: 1rem;
    right: 0;
  }
  #searchresults { margin-top: 0; }
}

/* Tier 3 (>= 1500px): model list independently centered, sidebar attached on left */
@media (min-width: 1500px) {
  #page-wrapper {
    max-width: none;
    display: block;
    position: relative;
  }
  /* Sidebar: absolute, attached to left of centered model list */
  #top-row {
    position: absolute;
    right: calc(50% + 18rem + 2.5rem); /* 50% + half-model-list + gap */
    width: 420px;
    top: 0;
    padding-top: 1rem;
    margin-bottom: 0;
  }
  /* Results: independently centered in viewport */
  #results-area {
    position: relative;
    max-width: 36rem;
    margin-left: calc(50% - 18rem);
    margin-right: 0;
    padding-top: 1rem;
  }
  /* Sort: absolute, top-right of results area */
  #sort-container {
    position: absolute;
    top: 1rem;
    right: 0;
  }
}

/* Tier 2b (1080px–1199.98px): graph panel on right, model list anchored left.
   Applies exactly when the >=1200 sidebar is absent but viewport is wide enough
   for a graph. Must come AFTER the 1200-tier block so it wins the cascade for the
   properties it sets (the 1200-tier does not match in this range, but ordering
   keeps intent explicit). */
@media (min-width: 1080px) and (max-width: 1199.98px) {
  #page-wrapper { max-width: none; margin: 0; }   /* list goes full width, not centered 42rem */
  #top-row { max-width: 41rem; }                  /* pills row never runs under the fixed graph */
  #results-area { max-width: 36rem; margin-left: 2rem; margin-right: 0; }  /* list anchored left */
  #graph-panel {
    visibility: visible; opacity: 1; pointer-events: auto;
    left: calc(2rem + 36rem + 2.5rem);            /* right of the left-anchored list */
    right: 2rem;
    width: auto;
    max-width: none;
  }
}

/* --- Graph panel: KV cache memory vs context length --- */
#graph-svg { width: 100%; flex: 1 1 0; min-height: 0; display: block; }
#graph-panel {
  position: fixed;
  visibility: hidden;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.3s ease, visibility 0.3s ease, left 0.35s ease, right 0.35s ease, width 0.35s ease;
  /* shared chrome so both tiers share look: */
  padding: 1.25rem;
  border: 1px solid #e5e5e5;
  border-radius: 1rem;
  background: #ffffff;
  z-index: 30;
  top: 5rem;
  display: flex;
  flex-direction: column;
  height: calc(100vh - 6rem);
  overflow: hidden;
}
.dark #graph-panel { border-color: #262626; background: #0a0a0a; }
#graph-legend-row { flex: 0 1 auto; overflow-y: auto; min-height: 0; max-height: 35%; }
/* Filters show/hide toggle lives in the graph header; only meaningful where
   the sidebar exists (>= 1500px tier) */
#graph-filters-toggle { display: none; }
@media (min-width: 1500px) {
  #graph-filters-toggle { display: inline-flex; align-items: center; }
}
/* Graph hide/show toggle: visible whenever the graph is visible (>= 1080px) */
#graph-hide-toggle { display: none; }
@media (min-width: 1080px) {
  #graph-hide-toggle { display: inline-flex; align-items: center; }
}
/* Tier 3 (>= 1500px): graph fixed to viewport, right of centered results */
@media (min-width: 1500px) {
  #graph-panel {
    visibility: visible;
    opacity: 1;
    pointer-events: auto;
    right: 2rem;
    width: clamp(320px, calc(50vw - 18rem - 2.5rem - 2rem), 560px);
  }
  /* Scrolled past the (offscreen) filter sidebar — or filters manually hidden
     via the graph-panel toggle: graph widens to fill the right side, junction
     at screen center, 4rem gap between list and graph, and 4rem right margin
     (same as the gap). <main> has lg:px-8 (2rem) padding that shifts the
     in-flow list right; we subtract it from margin-left. */
  body.filters-offscreen, body.filters-hidden { --graph-w: min(800px, calc(50vw - 6rem)); }
  body.filters-offscreen #results-area, body.filters-hidden #results-area {
    width: 36rem;
    /* list right edge at 50vw - 2rem, minus 2rem main padding */
    margin-left: max(0rem, calc(50vw - 36rem - 2rem - 2rem));
    margin-right: 0;
  }
  body.filters-offscreen #graph-panel, body.filters-hidden #graph-panel {
    /* panel left edge at 50vw + 2rem, right edge at 4rem from viewport right */
    width: var(--graph-w);
    right: 4rem;
  }
  /* Manual filters toggle: sidebar hidden on demand, expanded graph layout */
  body.filters-hidden #top-row { display: none; }
  /* Manual graph toggle: graph hidden, sidebar shows in its normal position.
     The sidebar re-appears (undo filters-hidden/filters-offscreen) and the
     results area returns to its normal centered width. */
  body.graph-hidden #graph-panel {
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
  }
  body.graph-hidden #top-row { display: block !important; }
  body.graph-hidden #results-area {
    width: auto;
    margin-left: calc(50% - 18rem);
    margin-right: 0;
  }
  /* Also apply at the 1080px–1199px tier */
}
@media (min-width: 1080px) and (max-width: 1199.98px) {
  body.graph-hidden #graph-panel {
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
  }
}

/* Graph SVG styling */
#graph-svg .graph-line { fill: none; stroke-width: 2; }
#graph-svg .graph-axis { stroke: #d4d4d4; stroke-width: 1; }
.dark #graph-svg .graph-axis { stroke: #404040; }
#graph-svg .graph-grid { stroke: #e5e5e5; stroke-width: 0.5; }
.dark #graph-svg .graph-grid { stroke: #262626; }
#graph-svg .graph-label { font-size: 11px; fill: #737373; }
.dark #graph-svg .graph-label { fill: #a3a3a3; }
#graph-svg .graph-dot { stroke: #fff; stroke-width: 1.5; cursor: pointer; transition: r 0.15s; }
.dark #graph-svg .graph-dot { stroke: #0a0a0a; }
#graph-svg .graph-dot:hover { r: 3.5; }
#graph-legend .legend-item { display: inline-flex; align-items: center; gap: 0.25rem; white-space: nowrap; }
#graph-legend .legend-swatch { display: inline-block; width: 9px; height: 9px; border-radius: 2.5px; flex: 0 0 auto; }
#graph-legend { min-width: 0; flex: 1 1 auto; }
#graph-toggles { flex: 0 0 auto; }
#graph-legend .legend-model { display: inline-flex; align-items: center; gap: 0.3rem; flex-wrap: wrap; }
#graph-legend .legend-model b { font-weight: 600; }
#graph-legend button.legend-item, #graph-legend button.legend-model {
  cursor: pointer; user-select: none; font: inherit; color: inherit;
  background: none; border: 0; padding: 0; text-align: left;
}
#graph-legend .legend-item[data-key]:hover, #graph-legend .legend-model[data-key]:hover { opacity: 0.7; }
#graph-legend .legend-off { opacity: 0.35; }
#graph-legend .legend-off:hover { opacity: 0.5; }
#graph-legend .legend-more { color: #737373; font-style: italic; }
.dark #graph-legend .legend-more { color: #a3a3a3; }
#graph-legend .legend-focus { cursor: pointer; }
#graph-legend .legend-focused { text-decoration: underline; text-underline-offset: 2px; }

/* --- Detail-page graph panel (model detail pages) ---
   The index #graph-panel is position:fixed and hidden by default, shown via
   media queries and body class toggles. On detail pages the panel lives in
   normal document flow inside the 52rem <main> column. These overrides
   neutralise every fixed/hidden/media/toggle rule that would otherwise apply,
   and keep the panel hidden until JS confirms graph data exists (graph-ready)
   so embedding/image-gen models show nothing instead of an empty box. */
#graph-panel.detail-graph,
body.filters-offscreen #graph-panel.detail-graph,
body.filters-hidden #graph-panel.detail-graph {
  position: static !important;
  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
  width: auto !important;
  right: auto !important;
  max-width: none !important;
  max-height: none !important;
  height: auto !important;
  overflow-y: visible !important;
  top: auto !important;
  left: auto !important;
  margin-top: 2rem;
}
#graph-panel.detail-graph { display: none; }
#graph-panel.detail-graph.graph-ready { display: block; }
#graph-panel.detail-graph #graph-svg {
  flex: none;
  aspect-ratio: 560 / 360;
  height: auto;
}
#graph-panel.detail-graph #graph-legend-row {
  flex: 0 0 auto;
  max-height: none;
  overflow: visible;
}
"""

APP_JS = r"""// ollama-search frontend logic.
// Filtering, sorting, dark-mode, tab switching, copy-to-clipboard.

var NAV_MODELS = null;
var NAV_BASE = (function() {
  var s = document.querySelector('script[src*="assets/app.js"]');
  if (s) {
    var src = s.getAttribute('src');
    var idx = src.indexOf('assets/app.js');
    if (idx > 0) return src.substring(0, idx);
  }
  return '/';
})();

function loadNavModels(cb) {
  if (NAV_MODELS) { cb(NAV_MODELS); return; }
  var cached = null;
  try { cached = sessionStorage.getItem('nav-models'); } catch (e) {}
  if (cached) {
    try { NAV_MODELS = JSON.parse(cached); cb(NAV_MODELS); return; } catch (e) {}
  }
  fetch(NAV_BASE + 'assets/models.json').then(function(r) { return r.json(); }).then(function(data) {
    NAV_MODELS = data;
    try { sessionStorage.setItem('nav-models', JSON.stringify(data)); } catch (e) {}
    cb(NAV_MODELS);
  }).catch(function() { cb([]); });
}

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function renderNavSuggest(query) {
  var sp = document.getElementById('searchpreview');
  if (!sp) return;
  var q = query.toLowerCase().trim();
  if (!q) { sp.classList.add('hidden'); sp.innerHTML = ''; return; }

  var html = '<div class="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl w-full shadow-2xl shadow-black/5 overflow-hidden" id="search-preview-container" tabindex="0">';
  html += '<div role="list" id="search-preview-list" class="group">';

  var results = [];
  for (var i = 0; i < NAV_MODELS.length; i++) {
    var m = NAV_MODELS[i];
    var name = m.name || '';
    var desc = m.description || '';
    if (name.toLowerCase().indexOf(q) !== -1 || desc.toLowerCase().indexOf(q) !== -1) {
      results.push(m);
    }
  }
  results.sort(function(a, b) { return (b.pulls || 0) - (a.pulls || 0); });
  var top = results.slice(0, 5);

  if (top.length === 0) {
    html += '<div class="px-6 py-4 text-neutral-800 dark:text-neutral-300 text-sm">No models found.</div>';
  } else {
    for (var i = 0; i < top.length; i++) {
      var m = top[i];
      var path = m.path || ('/library/' + m.name);
      html += '<div result>';
      html += '<a tabindex="0" href="' + NAV_BASE + path.replace(/^\//, '') + '" class="flex items-center h-16 px-6 py-4 hover:bg-neutral-50 dark:hover:bg-white/5 focus:ring-0 focus:outline-none focus:bg-neutral-50 dark:focus:bg-white/5">';
      html += '<div class="min-w-0 flex-1">';
      html += '<h2 class="text-sm font-medium truncate dark:text-neutral-100">' + escHtml(m.name) + '</h2>';
      html += '<p class="text-xs text-gray-600 dark:text-gray-600 truncate">' + escHtml(m.description) + '</p>';
      html += '</div></a></div>';
    }
  }

  html += '</div>';
  html += '<a tabindex="0" id="view-all-link" href="' + NAV_BASE + '?q=' + encodeURIComponent(query) + '" class="' + (top.length === 0 ? 'hidden' : '') + ' block px-6 py-3 border-t border-neutral-200 dark:border-neutral-800 text-center text-sm font-semibold hover:bg-neutral-50 dark:hover:bg-white/5 focus:bg-neutral-50 dark:focus:bg-white/5 focus:outline-none focus:ring-0 dark:text-neutral-200">View all &#8594;</a>';
  html += '</div>';

  sp.innerHTML = html;
  sp.classList.remove('hidden');
}

var navSuggestTimer = null;
function initNavSuggest() {
  var input = document.getElementById('navbar-input');
  var sp = document.getElementById('searchpreview');
  if (!input || !sp) return;

  input.addEventListener('input', function() {
    if (navSuggestTimer) clearTimeout(navSuggestTimer);
    navSuggestTimer = setTimeout(function() {
      var v = input.value;
      if (!v.trim()) { sp.classList.add('hidden'); sp.innerHTML = ''; return; }
      loadNavModels(function() { renderNavSuggest(v); });
    }, 100);
  });

  sp.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      sp.classList.add('hidden');
      input.focus();
      e.preventDefault();
      return;
    }
    if (e.key === 'Enter') {
      var el = document.activeElement;
      if (el && el.tagName === 'A') { el.click(); e.preventDefault(); return; }
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      var items = Array.from(sp.querySelectorAll('#search-preview-list a, #view-all-link'));
      var ci = items.indexOf(document.activeElement);
      var ni = e.key === 'ArrowDown' ? ci + 1 : ci - 1;
      if (ni >= items.length) ni = 0;
      if (ni < 0) ni = items.length - 1;
      if (items[ni]) items[ni].focus();
      e.preventDefault();
    }
  });
}

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
  var a = document.activeElement;
  var input = (a && (a.id === 'form-input' || a.id === 'navbar-input'))
    ? a
    : (document.getElementById('form-input') || document.getElementById('navbar-input'));
  return input ? input.value.toLowerCase().trim() : '';
}

function getCloudFilter() {
  var el = document.getElementById('cloud-filter');
  return el ? el.value : 'all';
}

function getSizeMin() {
  var el = document.getElementById('size-min');
  return el ? parseInt(el.value) : 0;
}
function getSizeMax() {
  var el = document.getElementById('size-max');
  return el ? parseInt(el.value) : 500;
}

function getContextMin() {
  var el = document.getElementById('context-min');
  return el ? parseInt(el.value) : 0;
}
function getContextMax() {
  var el = document.getElementById('context-max');
  return el ? parseInt(el.value) : 1048576;
}

function filtersToParams() {
  var p = new URLSearchParams();
  var q = getQuery();
  if (q) p.set('q', q);
  getSelectedCaps().forEach(function(c) { p.append('c', c); });
  var o = getSort();
  if (o && o !== 'popular') p.set('o', o);
  var cloud = getCloudFilter();
  if (cloud && cloud !== 'all') p.set('cloud', cloud);
  var smin = getSizeMin(), smax = getSizeMax();
  if (smin !== 0) p.set('smin', String(smin));
  if (smax !== 500) p.set('smax', String(smax));
  var cmin = getContextMin(), cmax = getContextMax();
  if (cmin !== 0) p.set('cmin', String(cmin));
  if (cmax !== 1048576) p.set('cmax', String(cmax));
  var ma = document.getElementById('more-audio');
  if (ma && ma.checked) p.set('audio', '1');
  var ml = document.getElementById('more-mlx');
  if (ml && ml.checked) p.set('mlx', '1');
  var mt = document.getElementById('more-mtp');
  if (mt && mt.checked) p.set('mtp', '1');
  var mi = document.getElementById('more-image');
  if (mi && mi.checked) p.set('image', '1');
  var moe = document.querySelector('input[name="moe-filter"]:checked');
  if (moe && moe.value !== 'all') p.set('moe', moe.value);
  var tpl = document.querySelector('input[name="tpl-filter"]:checked');
  if (tpl && tpl.value !== 'all') p.set('tpl', tpl.value);
  return p;
}

function syncUrlToFilters() {
  if (!document.getElementById('card-list')) return;
  var p = filtersToParams();
  var qs = p.toString();
  var url = location.pathname + (qs ? '?' + qs : '') + location.hash;
  history.replaceState(null, '', url);
}

function applyUrlToFilters() {
  var p = new URLSearchParams(location.search);
  var q = p.get('q') || '';
  var formInput = document.getElementById('form-input');
  var navInput = document.getElementById('navbar-input');
  if (formInput) formInput.value = q;
  if (navInput) navInput.value = q;
  var caps = p.getAll('c');
  document.querySelectorAll('.cap-filter').forEach(function(cb) {
    cb.checked = caps.indexOf(cb.getAttribute('data-cap')) !== -1;
  });
  var o = p.get('o') || 'popular';
  var ds = document.getElementById('desktop-sort-select');
  var ms = document.getElementById('mobile-sort-select');
  if (ds) ds.value = o;
  if (ms) ms.value = o;
  var cloud = p.get('cloud') || 'all';
  var cf = document.getElementById('cloud-filter');
  if (cf) cf.value = cloud;
  var smin = parseInt(p.get('smin'), 10);
  var smax = parseInt(p.get('smax'), 10);
  var sizeMin = document.getElementById('size-min');
  var sizeMax = document.getElementById('size-max');
  if (sizeMin && !isNaN(smin)) sizeMin.value = Math.max(0, Math.min(500, smin));
  if (sizeMax && !isNaN(smax)) sizeMax.value = Math.max(0, Math.min(500, smax));
  // Update the dual-handle slider visuals to match the restored values.
  // updateSizeVisuals() is visuals-only (no applyFilters) to avoid a double
  // applyFilters on init; applyFilters() runs right after applyUrlToFilters.
  updateSizeVisuals();
  var cmin = parseInt(p.get('cmin'), 10);
  var cmax = parseInt(p.get('cmax'), 10);
  var ctxMin = document.getElementById('context-min');
  var ctxMax = document.getElementById('context-max');
  if (ctxMin && !isNaN(cmin)) ctxMin.value = Math.max(0, Math.min(1048576, cmin));
  if (ctxMax && !isNaN(cmax)) ctxMax.value = Math.max(0, Math.min(1048576, cmax));
  updateContextVisuals();
  function setMore(id, on) { var el = document.getElementById(id); if (el) el.checked = !!on; }
  setMore('more-audio', p.get('audio') === '1');
  setMore('more-mlx', p.get('mlx') === '1');
  setMore('more-mtp', p.get('mtp') === '1');
  setMore('more-image', p.get('image') === '1');
  var moe = p.get('moe') || 'all';
  var moeR = document.querySelector('input[name="moe-filter"][value="' + moe + '"]');
  if (moeR) moeR.checked = true;
  var tpl = p.get('tpl') || 'all';
  var tplR = document.querySelector('input[name="tpl-filter"][value="' + tpl + '"]');
  if (tplR) tplR.checked = true;
}

// Piecewise mapping between slider position (0-100%) and size value (0-500 billions)
// Breakpoints: 0%->0, 20%->6, 40%->12, 60%->32, 80%->128, 100%->500
var SIZE_BP = [
  {pct: 0, val: 0},
  {pct: 20, val: 6},
  {pct: 40, val: 12},
  {pct: 60, val: 32},
  {pct: 80, val: 128},
  {pct: 100, val: 500}
];
function valToPct(v) {
  v = Math.max(0, Math.min(500, v));
  for (var i = 0; i < SIZE_BP.length - 1; i++) {
    var a = SIZE_BP[i], b = SIZE_BP[i + 1];
    if (v >= a.val && v <= b.val) {
      return a.pct + (v - a.val) / (b.val - a.val) * (b.pct - a.pct);
    }
  }
  return v <= 0 ? 0 : 100;
}
function pctToVal(pct) {
  pct = Math.max(0, Math.min(100, pct));
  for (var i = 0; i < SIZE_BP.length - 1; i++) {
    var a = SIZE_BP[i], b = SIZE_BP[i + 1];
    if (pct >= a.pct && pct <= b.pct) {
      return Math.round(a.val + (pct - a.pct) / (b.pct - a.pct) * (b.val - a.val));
    }
  }
  return pct <= 0 ? 0 : 500;
}

// Piecewise mapping between slider position (0-100%) and context value (0-1048576 tokens)
// Breakpoints: log2-positioned — each doubling step gets equal slider width.
// Steps: 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M = 9 doublings.
// 0% maps to 0 (the "<4K" region), then 9 equal segments of 100/9 ≈ 11.11% each.
var CONTEXT_BP = [
  {pct: 0,       val: 0},
  {pct: 11.111,  val: 4096},
  {pct: 22.222,  val: 8192},
  {pct: 33.333,  val: 16384},
  {pct: 44.444,  val: 32768},
  {pct: 55.556,  val: 65536},
  {pct: 66.667,  val: 131072},
  {pct: 77.778,  val: 262144},
  {pct: 88.889,  val: 524288},
  {pct: 100,     val: 1048576}
];
function ctxValToPct(v) {
  v = Math.max(0, Math.min(1048576, v));
  for (var i = 0; i < CONTEXT_BP.length - 1; i++) {
    var a = CONTEXT_BP[i], b = CONTEXT_BP[i + 1];
    if (v >= a.val && v <= b.val) {
      return a.pct + (v - a.val) / (b.val - a.val) * (b.pct - a.pct);
    }
  }
  return v <= 0 ? 0 : 100;
}
function ctxPctToVal(pct) {
  pct = Math.max(0, Math.min(100, pct));
  for (var i = 0; i < CONTEXT_BP.length - 1; i++) {
    var a = CONTEXT_BP[i], b = CONTEXT_BP[i + 1];
    if (pct >= a.pct && pct <= b.pct) {
      return Math.round(a.val + (pct - a.pct) / (b.pct - a.pct) * (b.val - a.val));
    }
  }
  return pct <= 0 ? 0 : 1048576;
}

function sizeToBillions(s) {
  if (!s) return null;
  s = s.toLowerCase();
  // MoE: e.g. "8x7b" -> use trailing factor 7b
  var moeMatch = s.match(/^(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)([bm])$/);
  if (moeMatch) {
    s = moeMatch[2] + moeMatch[3];
  }
  // "e2b" -> 2b
  var eMatch = s.match(/^e(\d+(?:\.\d+)?)([bm])$/);
  if (eMatch) {
    s = eMatch[1] + eMatch[2];
  }
  var m = s.match(/^(\d+(?:\.\d+)?)([bm])$/);
  if (!m) return null;
  var num = parseFloat(m[1]);
  return m[2] === 'b' ? num : num / 1000;
}

function matchSizeRange(cardSizesAttr) {
  var minB = getSizeMin();
  var maxB = getSizeMax();
  if (minB === 0 && maxB === 500) return true; // no filter
  var sizes = (cardSizesAttr || '').split(/\s+/).filter(Boolean);
  if (sizes.length === 0) return false;
  var bs = [];
  for (var i = 0; i < sizes.length; i++) {
    var b = sizeToBillions(sizes[i]);
    if (b != null) bs.push(b);
  }
  if (bs.length === 0) return false;
  for (var i = 0; i < bs.length; i++) {
    if (bs[i] >= minB && bs[i] <= maxB) return true;
  }
  return false;
}

function matchContextRange(cardCtxAttr) {
  var cmin = getContextMin();
  var cmax = getContextMax();
  if (cmin === 0 && cmax === 1048576) return true; // no filter
  var maxCtx = parseInt(cardCtxAttr || '0', 10);
  if (maxCtx === 0) return true; // models with no context data pass all filters
  return maxCtx >= cmin && maxCtx <= cmax;
}

// Update the dual-handle size slider visuals (fill, handles, tooltips, button
// label) from the current size-min/size-max input values. Visuals only — does
// NOT call applyFilters, so it's safe to call from applyUrlToFilters before the
// initial applyFilters() runs.
function updateSizeVisuals() {
  var sizeMin = document.getElementById('size-min');
  var sizeMax = document.getElementById('size-max');
  if (!sizeMin || !sizeMax) return;
  var mn = parseInt(sizeMin.value);
  var mx = parseInt(sizeMax.value);
  if (mn > mx) { var tmp = mn; mn = mx; mx = tmp; sizeMin.value = mn; sizeMax.value = mx; }
  var pmin = valToPct(mn);
  var pmax = valToPct(mx);
  var sizeFill = document.getElementById('size-slider-fill');
  var sizeHandleMin = document.getElementById('size-handle-min');
  var sizeHandleMax = document.getElementById('size-handle-max');
  var sizeMinTip = document.getElementById('size-min-tooltip');
  var sizeMaxTip = document.getElementById('size-max-tooltip');
  var sizeBtnLabel = document.getElementById('size-filter-btn');
  if (sizeFill) { sizeFill.style.left = pmin + '%'; sizeFill.style.width = (pmax - pmin) + '%'; }
  if (sizeHandleMin) sizeHandleMin.style.left = pmin + '%';
  if (sizeHandleMax) sizeHandleMax.style.left = pmax + '%';
  function sizeLabel(v) {
    if (v === 0) return '<1b';
    if (v >= 500) return '>500b';
    return v + 'b';
  }
  if (sizeMinTip) sizeMinTip.textContent = sizeLabel(mn);
  if (sizeMaxTip) sizeMaxTip.textContent = sizeLabel(mx);
  if (sizeBtnLabel) {
    sizeBtnLabel.textContent = (mn === 0 && mx === 500) ? 'Size' : 'Size: ' + sizeLabel(mn) + ' - ' + sizeLabel(mx);
  }
  // Toggle active class on dots within the fill range
  var sizeContainer = document.getElementById('size-slider-container');
  if (sizeContainer) {
    sizeContainer.querySelectorAll('.slider-dot').forEach(function(dot) {
      var dotPct = parseFloat(dot.style.left);
      dot.classList.toggle('active', dotPct >= pmin && dotPct <= pmax);
    });
  }
}

function updateContextVisuals() {
  var ctxMin = document.getElementById('context-min');
  var ctxMax = document.getElementById('context-max');
  if (!ctxMin || !ctxMax) return;
  var mn = parseInt(ctxMin.value);
  var mx = parseInt(ctxMax.value);
  if (mn > mx) { var tmp = mn; mn = mx; mx = tmp; ctxMin.value = mn; ctxMax.value = mx; }
  var pmin = ctxValToPct(mn);
  var pmax = ctxValToPct(mx);
  var ctxFill = document.getElementById('context-slider-fill');
  var ctxHandleMin = document.getElementById('context-handle-min');
  var ctxHandleMax = document.getElementById('context-handle-max');
  var ctxMinTip = document.getElementById('context-min-tooltip');
  var ctxMaxTip = document.getElementById('context-max-tooltip');
  var ctxBtnLabel = document.getElementById('context-filter-btn');
  if (ctxFill) { ctxFill.style.left = pmin + '%'; ctxFill.style.width = (pmax - pmin) + '%'; }
  if (ctxHandleMin) ctxHandleMin.style.left = pmin + '%';
  if (ctxHandleMax) ctxHandleMax.style.left = pmax + '%';
  function ctxLabel(v) {
    if (v === 0) return '0';
    if (v >= 1048576) return '>1M';
    if (v >= 1024 * 1024) return (v / (1024 * 1024)) + 'M';
    if (v >= 1024) return Math.round(v / 1024) + 'K';
    return String(v);
  }
  if (ctxMinTip) ctxMinTip.textContent = ctxLabel(mn);
  if (ctxMaxTip) ctxMaxTip.textContent = ctxLabel(mx);
  if (ctxBtnLabel) {
    ctxBtnLabel.textContent = (mn === 0 && mx === 1048576) ? 'Context' : 'Context: ' + ctxLabel(mn) + ' - ' + ctxLabel(mx);
  }
  // Toggle active class on dots within the fill range
  var ctxContainer = document.getElementById('context-slider-container');
  if (ctxContainer) {
    ctxContainer.querySelectorAll('.slider-dot').forEach(function(dot) {
      var dotPct = parseFloat(dot.style.left);
      dot.classList.toggle('active', dotPct >= pmin && dotPct <= pmax);
    });
  }
}

function applyFilters() {
  var q = getQuery();
  var caps = getSelectedCaps();
  var sort = getSort();
  var cloudFilter = getCloudFilter();
  var list = document.getElementById('card-list');
  if (!list) return;
  var cards = Array.from(list.querySelectorAll('li[x-test-model]'));
  var moreAudio = document.getElementById('more-audio');
  var moreMlx = document.getElementById('more-mlx');
  var moreMtp = document.getElementById('more-mtp');
  var moreImage = document.getElementById('more-image');
  var moeRadio = document.querySelector('input[name="moe-filter"]:checked');
  var moreAudioOn = moreAudio && moreAudio.checked;
  var moreMlxOn = moreMlx && moreMlx.checked;
  var moreMtpOn = moreMtp && moreMtp.checked;
  var moreImageOn = moreImage && moreImage.checked;
  var moeVal = moeRadio ? moeRadio.value : 'all';
  var tplRadio = document.querySelector('input[name="tpl-filter"]:checked');
  var tplVal = tplRadio ? tplRadio.value : 'all';
  // Filter
  var visible = 0;
  cards.forEach(function(card) {
    var title = card.querySelector('[x-test-search-response-title]') ? card.querySelector('[x-test-search-response-title]').textContent.toLowerCase() : '';
    var desc = card.querySelector('p.break-words') ? card.querySelector('p.break-words').textContent.toLowerCase() : '';
    var cardCaps = [];
    card.querySelectorAll('[x-test-capability]').forEach(function(el) { cardCaps.push(el.textContent.toLowerCase()); });
    var isCloud = card.getAttribute('data-cloud') === 'true';
    var isCloudOnly = card.getAttribute('data-cloud-only') === 'true';
    var cardSizes = card.getAttribute('data-sizes') || '';
    var cardContext = card.getAttribute('data-context') || '0';
    var isOfficial = card.getAttribute('data-official') !== 'false';
    var matchSize = matchSizeRange(cardSizes);
    var matchContext = matchContextRange(cardContext);
    var cardPath = (card.getAttribute('data-path') || '').toLowerCase();
    var matchText = !q || title.indexOf(q) !== -1 || desc.indexOf(q) !== -1 || cardPath.indexOf(q) !== -1;
    var matchCaps = caps.length === 0 || caps.every(function(c) { return cardCaps.indexOf(c) !== -1; });
    var matchCloud = cloudFilter === 'all'
      || (cloudFilter === 'cloud' && isCloud)
      || (cloudFilter === 'local' && !isCloudOnly);
    var isAudio = card.getAttribute('data-audio') === 'true';
    var isMlx = card.getAttribute('data-mlx') === 'true';
    var isMoe = card.getAttribute('data-moe') === 'true';
    var isMtp = card.getAttribute('data-mtp') === 'true';
    var isImage = card.getAttribute('data-image') === 'true';
    var cardTpl = card.getAttribute('data-template-type') || 'base';
    var matchMoreAudio = !moreAudioOn || isAudio;
    var matchMoreMlx = !moreMlxOn || isMlx;
    var matchMoreMtp = !moreMtpOn || isMtp;
    var matchMoreImage = !moreImageOn || isImage;
    var matchMoe = moeVal === 'all' || (moeVal === 'moe' && isMoe) || (moeVal === 'dense' && !isMoe);
    var matchTpl = tplVal === 'all' || cardTpl === tplVal;
    var show = matchText && matchCaps && matchCloud && matchSize && matchContext && matchMoreAudio && matchMoreMlx && matchMoreMtp && matchMoreImage && matchMoe && matchTpl;
    if (show && !q && !isOfficial && !window.IS_PROFILE_PAGE) show = false;
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
    'tags': 'data-tag-count',
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
    } else if (sort === 'popular' || sort === 'newest' || sort === 'oldest') {
      var ra = parseFloat(a.getAttribute(attr) || '9999');
      var rb = parseFloat(b.getAttribute(attr) || '9999');
      // Official models always sort before non-official (profile) models,
      // since the two groups have separate rank spaces (library /library?sort=popular
      // vs profile page ordering) and otherwise interleave.
      var aOff = a.getAttribute('data-official') !== 'false';
      var bOff = b.getAttribute('data-official') !== 'false';
      if (aOff !== bOff) {
        cmp = aOff ? -1 : 1;
      } else if (ra !== 9999 || rb !== 9999) {
        cmp = ra - rb;
      } else if (sort === 'popular') {
        var pa = parseFloat(a.getAttribute('data-pulls') || '0');
        var pb = parseFloat(b.getAttribute('data-pulls') || '0');
        cmp = pb - pa;
      } else if (sort === 'newest') {
        var ua = parseFloat(a.getAttribute('data-updated-rank') || '9999');
        var ub = parseFloat(b.getAttribute('data-updated-rank') || '9999');
        cmp = ua - ub;
      } else {
        var ua = parseFloat(a.getAttribute('data-updated-rank') || '9999');
        var ub = parseFloat(b.getAttribute('data-updated-rank') || '9999');
        cmp = ub - ua;
      }
    } else {
      var na = parseFloat(va) || 0;
      var nb = parseFloat(vb) || 0;
      cmp = na - nb;
      if (descending) cmp = -cmp;
    }
    return cmp;
  });
  cards.forEach(function(c) { list.appendChild(c); });

  // Update Size/More pill active state based on non-default filter values
  var sizeBtnEl = document.getElementById('size-filter-btn');
  if (sizeBtnEl) {
    var sMin = getSizeMin();
    var sMax = getSizeMax();
    var sizeActive = !(sMin === 0 && sMax === 500);
    sizeBtnEl.classList.toggle('bg-neutral-100', sizeActive);
    sizeBtnEl.classList.toggle('dark:bg-neutral-800', sizeActive);
  }
  var ctxBtnEl = document.getElementById('context-filter-btn');
  if (ctxBtnEl) {
    var cMin = getContextMin();
    var cMax = getContextMax();
    var ctxActive = !(cMin === 0 && cMax === 1048576);
    ctxBtnEl.classList.toggle('bg-neutral-100', ctxActive);
    ctxBtnEl.classList.toggle('dark:bg-neutral-800', ctxActive);
  }
  var moreBtnEl = document.getElementById('more-filter-btn');
  if (moreBtnEl) {
    var moreActive = false;
    var ma = document.getElementById('more-audio');
    var ml = document.getElementById('more-mlx');
    var mt = document.getElementById('more-mtp');
    var mi = document.getElementById('more-image');
    if ((ma && ma.checked) || (ml && ml.checked) || (mt && mt.checked) || (mi && mi.checked)) moreActive = true;
    var mr = document.querySelector('input[name="moe-filter"]:checked');
    if (mr && mr.value !== 'all') moreActive = true;
    var tr = document.querySelector('input[name="tpl-filter"]:checked');
    if (tr && tr.value !== 'all') moreActive = true;
    moreBtnEl.classList.toggle('bg-neutral-100', moreActive);
    moreBtnEl.classList.toggle('dark:bg-neutral-800', moreActive);
  }
  syncUrlToFilters();
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

// Move filter elements between narrow-mode (More popup) and wide-mode (sidebar sections)
// based on viewport width. appendChild preserves IDs and event listeners.
function layoutFilters() {
  var wide = window.innerWidth >= 1200;
  var topRow = document.getElementById('top-row');
  var resultsArea = document.getElementById('results-area');
  var sortContainer = document.getElementById('sort-container');
  var capsRow = document.getElementById('caps-row');
  var morePills = document.getElementById('more-pills');
  var archContent = document.getElementById('arch-content');
  var tplContent = document.getElementById('tpl-content');
  var moreContent = document.getElementById('more-content');
  var archTarget = document.getElementById('arch-target');
  var tplTarget = document.getElementById('tpl-target');
  var archSection = document.getElementById('arch-section');
  var tplSection = document.getElementById('tpl-section');
  var moreSection = document.getElementById('more-section');

  if (wide) {
    // Move sort to results-area (absolute positioning at top-right)
    if (sortContainer && resultsArea && sortContainer.parentElement !== resultsArea)
      resultsArea.insertBefore(sortContainer, resultsArea.firstChild);
    // Move more pills into caps row (after cloud dropdown)
    if (morePills && capsRow && morePills.parentElement !== capsRow)
      capsRow.appendChild(morePills);
    // Move arch-content into its own section
    if (archContent && archTarget && archContent.parentElement !== archTarget)
      archTarget.appendChild(archContent);
    if (archSection) archSection.classList.add('active');
    // Move tpl-content into its own section
    if (tplContent && tplTarget && tplContent.parentElement !== tplTarget)
      tplTarget.appendChild(tplContent);
    if (tplSection) tplSection.classList.add('active');
    // Hide More section
    if (moreSection) moreSection.style.display = 'none';
  } else {
    // Move sort back to top-row (inline with pills)
    if (sortContainer && topRow && sortContainer.parentElement !== topRow)
      topRow.appendChild(sortContainer);
    // Move more-pills back into More popup (first child of more-content)
    if (morePills && moreContent && morePills.parentElement !== moreContent)
      moreContent.insertBefore(morePills, moreContent.firstChild);
    // Move arch-content back into More popup
    if (archContent && moreContent && archContent.parentElement !== moreContent)
      moreContent.appendChild(archContent);
    // Move tpl-content back into More popup
    if (tplContent && moreContent && tplContent.parentElement !== moreContent)
      moreContent.appendChild(tplContent);
    // Hide arch/tpl sections, show More section
    if (archSection) archSection.classList.remove('active');
    if (tplSection) tplSection.classList.remove('active');
    if (moreSection) moreSection.style.display = '';
  }
}

function initApp() {
  var desktopSort = document.getElementById('desktop-sort-select');
  var mobileSort = document.getElementById('mobile-sort-select');
  if (desktopSort && mobileSort) {
    syncSort(desktopSort, mobileSort);
    syncSort(mobileSort, desktopSort);
  }

  // Layout filters for narrow/wide mode (moves elements between containers)
  layoutFilters();
  window.addEventListener('resize', layoutFilters);

  if (document.getElementById('card-list')) {
    var formInput = document.getElementById('form-input');
    var navInput = document.getElementById('navbar-input');
    if (formInput) formInput.addEventListener('input', function() {
      if (navInput) navInput.value = formInput.value;
      applyFilters();
    });
    if (navInput) navInput.addEventListener('input', function() {
      if (formInput) formInput.value = navInput.value;
      applyFilters();
    });
    document.querySelectorAll('.cap-filter').forEach(function(cb) { cb.addEventListener('change', applyFilters); });
    var cloudFilter = document.getElementById('cloud-filter');
    if (cloudFilter) cloudFilter.addEventListener('change', applyFilters);

    var sizeBtn = document.getElementById('size-filter-btn');
    var sizePanel = document.getElementById('size-filter-panel');
    var contextBtn = document.getElementById('context-filter-btn');
    var contextPanel = document.getElementById('context-filter-panel');
    var moreBtn = document.getElementById('more-filter-btn');
    var morePanel = document.getElementById('more-filter-panel');

    // On mobile the panels are position:fixed relative to the viewport (see CSS).
    // Set a viewport-relative top so the panel sits just under its button, and
    // never off the bottom of the screen. Recomputed each time a panel opens and
    // on resize. On desktop (>=768px) the CSS overrides `top` back to the
    // button-relative calc(100% + 6px), so this var is harmless there.
    function isMobile() { return window.innerWidth < 768; }
    function placePanel(panel, btn) {
      if (!panel || !btn) return;
      if (!isMobile()) { panel.style.removeProperty('--panel-top'); return; }
      var top = btn.getBoundingClientRect().bottom + 6;
      var max = window.innerHeight - 80; // leave room so it can't render fully off-screen
      if (top > max) top = Math.max(8, max);
      panel.style.setProperty('--panel-top', top + 'px');
    }

    if (sizeBtn && sizePanel) {
      sizeBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        sizePanel.classList.toggle('hidden');
        if (!sizePanel.classList.contains('hidden')) {
          if (morePanel) morePanel.classList.add('hidden');
          if (contextPanel) contextPanel.classList.add('hidden');
          placePanel(sizePanel, sizeBtn);
        }
      });
      document.addEventListener('click', function(e) {
        if (!sizePanel.contains(e.target) && e.target !== sizeBtn) {
          sizePanel.classList.add('hidden');
        }
      });
    }

    if (contextBtn && contextPanel) {
      contextBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        contextPanel.classList.toggle('hidden');
        if (!contextPanel.classList.contains('hidden')) {
          if (sizePanel) sizePanel.classList.add('hidden');
          if (morePanel) morePanel.classList.add('hidden');
          placePanel(contextPanel, contextBtn);
        }
      });
      document.addEventListener('click', function(e) {
        if (!contextPanel.contains(e.target) && e.target !== contextBtn) {
          contextPanel.classList.add('hidden');
        }
      });
    }

    if (moreBtn && morePanel) {
      moreBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        morePanel.classList.toggle('hidden');
        if (!morePanel.classList.contains('hidden')) {
          if (sizePanel) sizePanel.classList.add('hidden');
          if (contextPanel) contextPanel.classList.add('hidden');
          placePanel(morePanel, moreBtn);
        }
      });
      document.addEventListener('click', function(e) {
        if (!morePanel.contains(e.target) && e.target !== moreBtn) {
          morePanel.classList.add('hidden');
        }
      });
    }

    // Re-place any open panel when the viewport size/orientation changes.
    window.addEventListener('resize', function() {
      if (sizePanel && !sizePanel.classList.contains('hidden')) placePanel(sizePanel, sizeBtn);
      if (contextPanel && !contextPanel.classList.contains('hidden')) placePanel(contextPanel, contextBtn);
      if (morePanel && !morePanel.classList.contains('hidden')) placePanel(morePanel, moreBtn);
    });
    document.querySelectorAll('.more-filter').forEach(function(cb) { cb.addEventListener('change', applyFilters); });
    document.querySelectorAll('.moe-radio').forEach(function(r) { r.addEventListener('change', applyFilters); });
    document.querySelectorAll('.tpl-radio').forEach(function(r) { r.addEventListener('change', applyFilters); });

    // Dual-handle slider (HuggingFace-style)
    var sizeMin = document.getElementById('size-min');
    var sizeMax = document.getElementById('size-max');
    var sizeFill = document.getElementById('size-slider-fill');
    var sizeHandleMin = document.getElementById('size-handle-min');
    var sizeHandleMax = document.getElementById('size-handle-max');
    var sizeMinTip = document.getElementById('size-min-tooltip');
    var sizeMaxTip = document.getElementById('size-max-tooltip');
    var sizeBtnLabel = document.getElementById('size-filter-btn');

    function sizeLabel(v) {
      if (v === 0) return '<1b';
      if (v >= 500) return '>500b';
      return v + 'b';
    }

    function updateSizeUI() {
      updateSizeVisuals();
      applyFilters();
    }

    // Drag handles
    var draggingSize = null;
    function onSizeDrag(e) {
      if (!draggingSize) return;
      e.preventDefault();
      var track = document.getElementById('size-slider-track');
      if (!track) return;
      var rect = track.getBoundingClientRect();
      var x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
      var pct = Math.max(0, Math.min(1, x / rect.width)) * 100;
      var val = pctToVal(pct);
      if (draggingSize === 'min') {
        val = Math.min(val, parseInt(sizeMax.value));
        sizeMin.value = val;
      } else {
        val = Math.max(val, parseInt(sizeMin.value));
        sizeMax.value = val;
      }
      updateSizeUI();
    }

    if (sizeHandleMin) {
      sizeHandleMin.addEventListener('mousedown', function(e) { draggingSize = 'min'; e.preventDefault(); });
      sizeHandleMin.addEventListener('touchstart', function(e) { draggingSize = 'min'; e.preventDefault(); });
    }
    if (sizeHandleMax) {
      sizeHandleMax.addEventListener('mousedown', function(e) { draggingSize = 'max'; e.preventDefault(); });
      sizeHandleMax.addEventListener('touchstart', function(e) { draggingSize = 'max'; e.preventDefault(); });
    }
    document.addEventListener('mousemove', onSizeDrag);
    document.addEventListener('mouseup', function() { draggingSize = null; });
    document.addEventListener('touchmove', onSizeDrag);
    document.addEventListener('touchend', function() { draggingSize = null; });

    // Tick buttons jump the min handle
    document.querySelectorAll('[data-tick]').forEach(function(btn) {
      btn.addEventListener('click', function(e) {
        e.stopPropagation();
        var v = parseInt(btn.getAttribute('data-tick'));
        if (v === 0) { sizeMin.value = 0; }
        else if (v >= 500) { sizeMax.value = 500; }
        else { sizeMin.value = v; }
        updateSizeUI();
      });
    });

    // Show tooltips on hover/focus
    if (sizeHandleMin) {
      sizeHandleMin.addEventListener('mouseenter', function() { sizeMinTip.classList.remove('hidden'); });
      sizeHandleMin.addEventListener('mouseleave', function() { sizeMinTip.classList.add('hidden'); });
    }
    if (sizeHandleMax) {
      sizeHandleMax.addEventListener('mouseenter', function() { sizeMaxTip.classList.remove('hidden'); });
      sizeHandleMax.addEventListener('mouseleave', function() { sizeMaxTip.classList.add('hidden'); });
    }

    var sizeReset = document.getElementById('size-filter-reset');
    if (sizeReset) sizeReset.addEventListener('click', function() {
      sizeMin.value = 0;
      sizeMax.value = 500;
      updateSizeUI();
    });

    // Context dual-handle slider
    var contextMin = document.getElementById('context-min');
    var contextMax = document.getElementById('context-max');
    var contextFill = document.getElementById('context-slider-fill');
    var contextHandleMin = document.getElementById('context-handle-min');
    var contextHandleMax = document.getElementById('context-handle-max');
    var contextMinTip = document.getElementById('context-min-tooltip');
    var contextMaxTip = document.getElementById('context-max-tooltip');
    var contextBtnLabel = document.getElementById('context-filter-btn');

    function updateContextUI() {
      updateContextVisuals();
      applyFilters();
    }

    var draggingContext = null;
    function onContextDrag(e) {
      if (!draggingContext) return;
      e.preventDefault();
      var track = document.getElementById('context-slider-track');
      if (!track) return;
      var rect = track.getBoundingClientRect();
      var x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
      var pct = Math.max(0, Math.min(1, x / rect.width)) * 100;
      var val = ctxPctToVal(pct);
      if (draggingContext === 'min') {
        val = Math.min(val, parseInt(contextMax.value));
        contextMin.value = val;
      } else {
        val = Math.max(val, parseInt(contextMin.value));
        contextMax.value = val;
      }
      updateContextUI();
    }

    if (contextHandleMin) {
      contextHandleMin.addEventListener('mousedown', function(e) { draggingContext = 'min'; e.preventDefault(); });
      contextHandleMin.addEventListener('touchstart', function(e) { draggingContext = 'min'; e.preventDefault(); });
    }
    if (contextHandleMax) {
      contextHandleMax.addEventListener('mousedown', function(e) { draggingContext = 'max'; e.preventDefault(); });
      contextHandleMax.addEventListener('touchstart', function(e) { draggingContext = 'max'; e.preventDefault(); });
    }
    document.addEventListener('mousemove', onContextDrag);
    document.addEventListener('mouseup', function() { draggingContext = null; });
    document.addEventListener('touchmove', onContextDrag);
    document.addEventListener('touchend', function() { draggingContext = null; });

    // Context tick buttons jump the min handle
    document.querySelectorAll('[data-ctx-tick]').forEach(function(btn) {
      btn.addEventListener('click', function(e) {
        e.stopPropagation();
        var v = parseInt(btn.getAttribute('data-ctx-tick'));
        if (v === 0) { contextMin.value = 0; }
        else if (v >= 1048576) { contextMax.value = 1048576; }
        else { contextMin.value = v; }
        updateContextUI();
      });
    });

    // Context tooltips on hover
    if (contextHandleMin) {
      contextHandleMin.addEventListener('mouseenter', function() { contextMinTip.classList.remove('hidden'); });
      contextHandleMin.addEventListener('mouseleave', function() { contextMinTip.classList.add('hidden'); });
    }
    if (contextHandleMax) {
      contextHandleMax.addEventListener('mouseenter', function() { contextMaxTip.classList.remove('hidden'); });
      contextHandleMax.addEventListener('mouseleave', function() { contextMaxTip.classList.add('hidden'); });
    }

    var contextReset = document.getElementById('context-filter-reset');
    if (contextReset) contextReset.addEventListener('click', function() {
      contextMin.value = 0;
      contextMax.value = 1048576;
      updateContextUI();
    });

    // Restore all filter state from the URL (query, capabilities, sort,
    // cloud, size, more/audio/mlx/mtp/image, moe, template), then apply.
    applyUrlToFilters();
    applyFilters();
    document.documentElement.classList.remove('js-init');
  }

  // --- Navbar search preview dropdown (non-search pages only) ---
  if (!document.getElementById('card-list')) {
    initNavSuggest();
  }
  initFmtFilters();
  initGraph();
}

// --- Graph panel: KV cache memory vs context length ---
var GRAPH_PALETTE = [
  '#bf6584', '#52f185', '#02a9f7', '#659818', '#fab1f4', '#f7a22a', '#8968f4', '#5be3f9', '#24b6a2', '#e068d9', '#b47828', '#d5d64a', '#fb9798', '#ff5c83', '#e34b00', '#9e91e4', '#8dca81', '#3480fc', '#b0a03e', '#eb7c36', '#af62c1', '#1292c0', '#4dba32', '#69c1fd', '#d080b6', '#8bcf27', '#e493f7', '#d84497', '#797dcd', '#82e6b4', '#109d7b', '#f7c56f', '#df7e80', '#3ed0c9', '#b6aaff', '#c9b959', '#75b169', '#c474fa', '#a1e647', '#31d96f', '#e3406b', '#f45fb0', '#eb98d0', '#8b8c29', '#729fe9', '#b4de8a', '#25afd2', '#92ac1d'
];
// Light-mode palette: darker variants with adequate contrast on white.
var GRAPH_PALETTE_LIGHT = [
  '#a05770', '#00d98f', '#e98dfe', '#f8a054', '#0089ed', '#258900', '#79c0f1', '#f86686', '#9e8339', '#a761d6', '#209994', '#666aab', '#db5521', '#78b31f', '#8e93f4', '#afc178', '#ed9cb5', '#d14e96', '#d48970', '#62b28a', '#c287bd', '#a840a3', '#ab5900', '#c1a9ef', '#c53444', '#00ade4', '#b39f1d', '#7474ef', '#017e9a', '#6a7a32', '#66ccba', '#399d57', '#f3751a', '#e56bc2', '#bb6e63', '#a172ac', '#6287c3', '#a9c71f', '#c07af1', '#1e8469', '#7457d1', '#296cd8', '#dab33d', '#7c921a', '#d58d29', '#8b5d96', '#40b1b7', '#c26e17'
];
function graphPalette() {
  return document.documentElement.classList.contains('dark') ? GRAPH_PALETTE : GRAPH_PALETTE_LIGHT;
}
var GRAPH_CTX_TICKS = [0, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608];
var GRAPH_MAX_MODELS = 10;
var GRAPH_MAX_CURVES = 40;
var GRAPH_RENDER_TIMER = null;

function fmtGiB(v) {
  if (v < 0.01) return '0';
  if (v < 1) return v.toFixed(2);
  if (v < 10) return v.toFixed(2);
  return v.toFixed(1);
}

function ctxLabel(v) {
  if (v <= 0) return '0';
  if (v >= 1024*1024) {
    var m = v / 1048576;
    return (m === Math.floor(m) ? String(m) : m.toFixed(1)) + 'M';
  }
  if (v >= 1024) return Math.round(v/1024) + 'K';
  return String(v);
}

function graphTagOrder(tagsObj) {
  var names = Object.keys(tagsObj);
  function parseSize(name) {
    var m = name.match(/^(\d+(?:\.\d+)?)([bmk])$/i);
    if (!m) return null;
    var n = parseFloat(m[1]);
    var unit = m[2].toLowerCase();
    if (unit === 'b') return n * 1e9;
    if (unit === 'm') return n * 1e6;
    return n * 1e3;  // 'k'
  }
  // Separate parsable and unparsable, sort each, then interleave unparsable in the middle.
  var parsed = [], unparsed = [];
  for (var i = 0; i < names.length; i++) {
    var sz = parseSize(names[i]);
    if (sz !== null) parsed.push({name: names[i], size: sz});
    else unparsed.push(names[i]);
  }
  parsed.sort(function(x, y) {
    if (x.size !== y.size) return x.size - y.size;
    return x.name < y.name ? -1 : (x.name > y.name ? 1 : 0);
  });
  unparsed.sort();
  var parsedNames = parsed.map(function(p) { return p.name; });
  // Unparsable tags go in the middle
  var mid = Math.floor(parsedNames.length / 2);
  var result = parsedNames.slice(0, mid).concat(unparsed).concat(parsedNames.slice(mid));
  return result;
}

var graphData = null;
var graphPanel = null;
var graphSvg = null;
var graphLegend = null;
var graphTooltip = null;
var graphObserver = null;
var visibleModels = [];
var disabledGraphCurves = {};
var graphHoverKey = null;
var graphFocusModel = null;
var graphNormalSubtitle = '';
var graphLogY = false;
var graphCtxCap = 0;  // 0 = no cap (show full range); set by the context slider
var graphModelOverrideList = null;
var graphOverrideEntry = null;

function getModelEntry(key) {
  if (graphOverrideEntry) return graphOverrideEntry;
  if (!graphData) return null;
  // Prefer the per-path entry (not merged across namespaces) so that a
  // community model like /frob/ds-flash doesn't pollute the graph for the
  // library /library/ds-flash. Fall back to the legacy name-keyed dict.
  if (graphData.by_path && graphData.by_path[key]) return graphData.by_path[key];
  if (graphData.models && graphData.models[key]) return graphData.models[key];
  return null;
}

// Map a path key (e.g. "library/ds-flash") back to the short display name
// (e.g. "ds-flash") using the card list.
function pathDisplayName(pathKey) {
  var cardList = document.getElementById('card-list');
  if (cardList) {
    var cards = cardList.querySelectorAll('li[x-test-model]');
    for (var i = 0; i < cards.length; i++) {
      if ((cards[i].getAttribute('data-path') || '') === pathKey) {
        return cards[i].getAttribute('data-name') || pathKey;
      }
    }
  }
  var slash = pathKey.indexOf('/');
  return slash >= 0 ? pathKey.slice(slash + 1) : pathKey;
}

function scheduleGraphRender() {
  if (GRAPH_RENDER_TIMER) clearTimeout(GRAPH_RENDER_TIMER);
  GRAPH_RENDER_TIMER = setTimeout(function() {
    GRAPH_RENDER_TIMER = null;
    renderGraph();
  }, 150);
}

function applyHoverDim() {
  if (!graphSvg) return;
  var els = graphSvg.querySelectorAll('[data-model]');
  for (var i = 0; i < els.length; i++) {
    var el = els[i];
    var m = el.getAttribute('data-model');
    var t = el.getAttribute('data-tag');
    var key = m + '|' + t;
    var dim = false;
    if (graphHoverKey) {
      dim = (key !== graphHoverKey);
    } else if (graphFocusModel) {
      dim = (m !== graphFocusModel);
    }
    el.style.opacity = dim ? '0.12' : '';
  }
}

function getVisibleModelsInOrder() {
  if (graphModelOverrideList) return graphModelOverrideList;
  var cardList = document.getElementById('card-list');
  if (!cardList) return [];
  var cards = cardList.querySelectorAll('li[x-test-model]');
  var result = [];
  var seen = {};
  for (var i = 0; i < cards.length; i++) {
    var card = cards[i];
    if (card.style.display === 'none') continue;
    var path = card.getAttribute('data-path');
    if (!path) continue;
    if (seen[path]) continue;            // dedupe: skip cards sharing a data-path
    if (visibleModels.indexOf(path) !== -1) {
      seen[path] = true;
      result.push(path);
    }
  }
  return result;
}

function renderGraph() {
  if (!graphData || !graphSvg) return;
  var ticks = graphData.ticks || GRAPH_CTX_TICKS;
  var models = graphData.models || {};
  var palette = graphPalette();

  var visInOrder = getVisibleModelsInOrder();

  // Filter to models present in data with >= 1 tag
  var shownModels = [];
  for (var i = 0; i < visInOrder.length; i++) {
    var name = visInOrder[i];
    var m = getModelEntry(name);
    if (!m || !m.tags) continue;
    var tagNames = Object.keys(m.tags);
    if (tagNames.length === 0) continue;
    shownModels.push(name);
    if (shownModels.length >= GRAPH_MAX_MODELS) break;
  }
  if (graphFocusModel && shownModels.indexOf(graphFocusModel) === -1) {
    graphFocusModel = null;
  }

  var totalInViewWithData = 0;
  for (var i = 0; i < visInOrder.length; i++) {
    var m = getModelEntry(visInOrder[i]);
    if (m && m.tags && Object.keys(m.tags).length > 0) totalInViewWithData++;
  }

  // Build curves: [model, tagName, points] where points = [{ctx, gib}]
  var curves = [];
  var droppedModels = 0;
  for (var mi = 0; mi < shownModels.length; mi++) {
    var name = shownModels[mi];
    var m = getModelEntry(name);
    var tagNames = Object.keys(m.tags).sort();
    var modelCurves = [];
    for (var tj = 0; tj < tagNames.length; tj++) {
      var tagName = tagNames[tj];
      var tag = m.tags[tagName];
      if (!tag || !tag.v || tag.v.length === 0) continue;
      var c = tag.c;
      var pts = [];
      for (var ti = 0; ti < ticks.length; ti++) {
        if (ticks[ti] < c) {
          if (ti < tag.v.length) pts.push({ctx: ticks[ti], gib: tag.v[ti]});
        }
      }
      // Endpoint at ctx = c (only if within cap)
      if (tag.v.length > 0 && (!graphCtxCap || c <= graphCtxCap)) {
        pts.push({ctx: c, gib: tag.v[tag.v.length - 1]});
      }
      // Apply context cap: drop points beyond the slider's selected max
      if (graphCtxCap) {
        var capped = [];
        for (var pi = 0; pi < pts.length; pi++) {
          if (pts[pi].ctx <= graphCtxCap) capped.push(pts[pi]);
        }
        pts = capped;
      }
      if (pts.length > 0) modelCurves.push([name, tagName, pts]);
    }
    // Check if adding this model's curves would exceed GRAPH_MAX_CURVES
    if (curves.length + modelCurves.length > GRAPH_MAX_CURVES) {
      // Drop entire trailing models where possible
      droppedModels = shownModels.length - mi;
      break;
    }
    curves = curves.concat(modelCurves);
  }
  // Hard cap on curves (safety)
  if (curves.length > GRAPH_MAX_CURVES) {
    curves = curves.slice(0, GRAPH_MAX_CURVES);
  }

  // Assign a distinct palette color to each curve sequentially.
  // Glasbey greedy ordering => consecutive entries are maximally distant,
  // so adjacent curves in this flattened list get well-separated colors.
  var curveColorMap = {};
  for (var ci_c = 0; ci_c < curves.length; ci_c++) {
    curveColorMap[curves[ci_c][0] + '|' + curves[ci_c][1]] = palette[ci_c % palette.length];
  }

  // Curves actually drawn/scaled: exclude legend-disabled ones. The legend,
  // caps, and color indices keep using the full `curves` list so toggling
  // never re-colors other curves and disabled entries stay visible (dimmed)
  // in the legend for re-enabling.
  var activeCurves = [];
  for (var ac = 0; ac < curves.length; ac++) {
    if (!disabledGraphCurves[curves[ac][0] + '|' + curves[ac][1]]) {
      activeCurves.push(curves[ac]);
    }
  }

  var W = graphSvg.clientWidth || 560;
  var H = graphSvg.clientHeight || 360;
  if (W < 100) W = 560;
  if (H < 100) H = 360;
  graphSvg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var padL = 56, padR = 16, padT = 16, padB = 44;
  var plotW = W - padL - padR;
  var plotH = H - padT - padB;

  // X scale: max ctx across drawable (non-disabled) curves
  var maxCtx = 0;
  for (var i = 0; i < activeCurves.length; i++) {
    var pts = activeCurves[i][2];
    for (var k = 0; k < pts.length; k++) {
      if (pts[k].ctx > maxCtx) maxCtx = pts[k].ctx;
    }
  }
  if (maxCtx <= 0) maxCtx = ticks[ticks.length - 1]; // degenerate/empty fallback

  // Y scale: the axis max IS the highest data point (no headroom/rounding),
  // so the peak curve's endpoint lands exactly on the top-right corner of the
  // plot and the top label reads its true value.
  var maxGiB = 0;
  for (var i = 0; i < activeCurves.length; i++) {
    var pts = activeCurves[i][2];
    for (var k = 0; k < pts.length; k++) {
      if (pts[k].gib > maxGiB) maxGiB = pts[k].gib;
    }
  }
  if (maxGiB <= 0) maxGiB = 1;

  // Y axis: linear (default) or log (toggle). Log mode starts at 0.01 GiB so
  // every curve — from 2 GiB models to 960 GiB behemoths — gets visible
  // vertical resolution. Each power-of-2 step gets equal pixel height.
  var YLOG_MIN = 0.01;
  var yLogMin, yLogMax;
  if (graphLogY) {
    yLogMin = YLOG_MIN;
    yLogMax = maxGiB;
    if (yLogMax <= yLogMin) yLogMax = yLogMin * 2;
  }
  function yPx(gib) {
    if (graphLogY) {
      if (gib <= 0) gib = yLogMin;
      var lMin = Math.log(yLogMin), lMax = Math.log(yLogMax);
      var lG = Math.log(gib < yLogMin ? yLogMin : gib);
      return padT + plotH - ((lG - lMin) / (lMax - lMin)) * plotH;
    }
    return padT + plotH - (gib / maxGiB) * plotH;
  }

  // Hybrid x scale: linear for ctx 0..XBREAK (small contexts stay readable),
  // then log-spaced — each doubling past XBREAK occupies the same pixel width
  // as the linear 16K→32K segment (the last linear segment). Pure linear when
  // every curve ends <= XBREAK.
  var XBREAK = 32768;
  var hasXBreak = maxCtx > XBREAK;
  var xLoW = 0, xSegW = 0;
  if (hasXBreak) {
    // Solve: xLoW + log2(maxCtx/XBREAK) * (xLoW/2) = plotW
    // (the log segment width per doubling = half the linear region, matching
    // the 32K→64K segment which spans the second half of 0..64K)
    var nSeg = Math.log(maxCtx / XBREAK) / Math.LN2;
    xLoW = plotW / (1 + 0.5 * nSeg);
    xSegW = xLoW / 2;
  }
  function xPx(ctx) {
    if (!hasXBreak) return padL + (ctx / maxCtx) * plotW;
    if (ctx <= XBREAK) return padL + (ctx / XBREAK) * xLoW;
    return padL + xLoW + (Math.log(ctx / XBREAK) / Math.LN2) * xSegW;
  }

  var svgContent = '';

  if (curves.length === 0) {
    // Empty state: show centered message, clear legend
    svgContent += '<text class="graph-label" x="' + (W / 2) + '" y="' + (H / 2) + '" text-anchor="middle">Scroll to bring models into view</text>';
    if (graphLegend) graphLegend.innerHTML = '';
    var sub = document.getElementById('graph-subtitle');
    if (sub) sub.textContent = 'Models in view';
  } else {
    // Y-axis gridlines + labels
    if (graphLogY) {
      // Log: place a gridline at each power of 2 from YLOG_MIN to maxGiB
      var pwr = Math.floor(Math.log(yLogMin) / Math.LN2);
      var endPwr = Math.ceil(Math.log(yLogMax) / Math.LN2);
      for (var pw = pwr; pw <= endPwr; pw++) {
        var val = Math.pow(2, pw);
        if (val < yLogMin * 0.99) continue;
        if (val > yLogMax * 1.01) continue;
        var y = yPx(val);
        svgContent += '<line class="graph-grid" x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (W - padR) + '" y2="' + y.toFixed(1) + '"/>';
        svgContent += '<text class="graph-label" x="' + (padL - 8) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end">' + fmtGiB(val) + '</text>';
      }
    } else {
      // Linear: 5 evenly-spaced steps
      var ySteps = 5;
      for (var i = 0; i <= ySteps; i++) {
        var val = (maxGiB / ySteps) * i;
        var y = yPx(val);
        svgContent += '<line class="graph-grid" x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (W - padR) + '" y2="' + y.toFixed(1) + '"/>';
        svgContent += '<text class="graph-label" x="' + (padL - 8) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end">' + fmtGiB(val) + '</text>';
      }
    }
    // Y axis title
    svgContent += '<text class="graph-label" x="' + (padL - 42) + '" y="' + (padT + plotH/2) + '" text-anchor="middle" transform="rotate(-90 ' + (padL - 42) + ' ' + (padT + plotH/2) + ')">' + (graphLogY ? 'GiB (log)' : 'GiB') + '</text>';

    // X-axis labels at context ticks (skip tick 0). Gridlines are drawn for
    // every candidate; text labels are greedily pruned so that close labels
    // (small ticks crammed on the left of a linear axis) don't collide.
    // Past the fixed tick list, generate power-of-2 log candidates up to
    // maxCtx so long runs (e.g. llama4 10M) don't render as a bare diagonal
    // with a lone '>1M'-style endpoint and no intermediate reference lines.
    var MIN_LABEL_GAP = 20;
    var longTicks = ticks.slice();
    if (maxCtx > ticks[ticks.length - 1]) {
      var t2 = ticks[ticks.length - 1];
      while (t2 * 2 <= maxCtx) {
        t2 = t2 * 2;
        longTicks.push(t2);
      }
    }
    var xCandidates = [];
    for (var i = 0; i < longTicks.length; i++) {
      var tick = longTicks[i];
      if (tick === 0) continue;
      if (tick > maxCtx) continue;
      var cx = xPx(tick);
      xCandidates.push({ x: cx, label: ctxLabel(tick) });
    }
    // Endpoint gridline + label when a model's context extends past the last tick
    if (maxCtx > longTicks[longTicks.length - 1]) {
      xCandidates.push({ x: xPx(maxCtx), label: ctxLabel(maxCtx) });
    }
    // Also add a gridline + label at each visible curve's own endpoint when it
    // falls between standard ticks (e.g. 125K, 327K) so the model's true max
    // context is labelled on the axis, not just implied by where the line stops.
    var tickSet = {};
    for (var ti_s = 0; ti_s < longTicks.length; ti_s++) tickSet[longTicks[ti_s]] = true;
    for (var ec = 0; ec < activeCurves.length; ec++) {
      var ecName = activeCurves[ec][0];
      var ecTag = activeCurves[ec][1];
      var ecEntry = getModelEntry(ecName);
      if (!ecEntry || !ecEntry.tags || !ecEntry.tags[ecTag]) continue;
      var ecCtx = ecEntry.tags[ecTag].c;
      if (ecCtx <= 0 || tickSet[ecCtx]) continue;
      tickSet[ecCtx] = true;
      xCandidates.push({ x: xPx(ecCtx), label: ctxLabel(ecCtx) });
    }
    // Draw ALL gridlines for every candidate (text pruning happens below)
    for (var ci = 0; ci < xCandidates.length; ci++) {
      var gx = xCandidates[ci].x;
      svgContent += '<line class="graph-grid" x1="' + gx.toFixed(1) + '" y1="' + padT + '" x2="' + gx.toFixed(1) + '" y2="' + (padT + plotH) + '"/>';
    }
    // Greedily select which labels to draw so they stay >= MIN_LABEL_GAP apart.
    // Always keep the first; keep a later one only if it clears the gap; always
    // keep the last — if it collides with the previously kept one, drop that
    // previously kept one so the endpoint/max label always shows.
    var kept = [];
    for (var ki = 0; ki < xCandidates.length; ki++) {
      var cand = xCandidates[ki];
      var isLast = (ki === xCandidates.length - 1);
      if (kept.length === 0) {
        kept.push(ki);
      } else {
        var lastKept = kept[kept.length - 1];
        var gap = cand.x - xCandidates[lastKept].x;
        if (gap >= MIN_LABEL_GAP) {
          kept.push(ki);
        } else if (isLast) {
          // Drop the previously kept one and keep the last instead
          kept[kept.length - 1] = ki;
        }
      }
    }
    for (var li = 0; li < kept.length; li++) {
      var kc = xCandidates[kept[li]];
      svgContent += '<text class="graph-label" x="' + kc.x.toFixed(1) + '" y="' + (H - padB + 18) + '" text-anchor="middle">' + kc.label + '</text>';
    }
    // X axis title
    svgContent += '<text class="graph-label" x="' + (padL + plotW/2) + '" y="' + (H - 6) + '" text-anchor="middle">Context length</text>';

    // Axes
    svgContent += '<line class="graph-axis" x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (padT + plotH) + '"/>';
    svgContent += '<line class="graph-axis" x1="' + padL + '" y1="' + (padT + plotH) + '" x2="' + (W - padR) + '" y2="' + (padT + plotH) + '"/>';
    if (activeCurves.length === 0) {
      svgContent += '<text class="graph-label" x="' + (W / 2) + '" y="' + (H / 2) + '" text-anchor="middle">All curves hidden — click a legend entry to re-enable</text>';
    }

    // Curves
    for (var ci = 0; ci < activeCurves.length; ci++) {
      var modelName = activeCurves[ci][0];
      var tagName = activeCurves[ci][1];
      var pts = activeCurves[ci][2];
      var color = curveColorMap[modelName + '|' + tagName] || palette[0];

      // Build the polyline path — in log-Y mode skip the gib=0 origin point
      // (ctx=0 maps to the bottom-left corner in log space, creating a spurious
      // dive-and-rise artifact). The curve starts at its first real tick.
      var pathD = '';
      var firstPt = true;
      for (var k = 0; k < pts.length; k++) {
        if (graphLogY && pts[k].gib <= 0) continue;
        var px = xPx(pts[k].ctx);
        var py = yPx(pts[k].gib);
        pathD += (firstPt ? 'M' : 'L') + px.toFixed(1) + ' ' + py.toFixed(1) + ' ';
        firstPt = false;
      }
      svgContent += '<path class="graph-line" d="' + pathD + '" stroke="' + color + '" data-model="' + escHtml(modelName) + '" data-tag="' + escHtml(tagName) + '"/>';

      // Dots at every point (skip gib=0 origin in log mode too)
      for (var k = 0; k < pts.length; k++) {
        if (graphLogY && pts[k].gib <= 0) continue;
        var dx = xPx(pts[k].ctx);
        var dy = yPx(pts[k].gib);
        svgContent += '<circle class="graph-dot" cx="' + dx.toFixed(1) + '" cy="' + dy.toFixed(1) + '" r="2.5" fill="' + color + '" data-model="' + escHtml(modelName) + '" data-tag="' + escHtml(tagName) + '" data-ctx="' + pts[k].ctx + '" data-gib="' + pts[k].gib.toFixed(3) + '"/>';
      }
    }

    // Legend: grouped per model
    if (graphLegend) {
      var curveCountByModel = {};
      for (var cc = 0; cc < curves.length; cc++) {
        var ccName = curves[cc][0];
        curveCountByModel[ccName] = (curveCountByModel[ccName] || 0) + 1;
      }
      var legendHtml = '';
      for (var mi3 = 0; mi3 < shownModels.length; mi3++) {
        var modelName = shownModels[mi3];
        // Check if this model has any curves in the final set
        if (!curveCountByModel[modelName]) {
          droppedModels = shownModels.length - mi3;
          break;
        }
        var mTags = getModelEntry(modelName).tags;
        var tagOrder = graphTagOrder(mTags);
        if (curveCountByModel[modelName] === 1) {
          // Single-curve model: swatch + model name, no redundant tag chip
          var singleTag = null;
          for (var ci4 = 0; ci4 < curves.length; ci4++) {
            if (curves[ci4][0] === modelName) { singleTag = curves[ci4][1]; break; }
          }
          var singleKey = modelName + '|' + singleTag;
          var singleColor = curveColorMap[singleKey] || palette[0];
          var singleDisp = pathDisplayName(modelName);
          legendHtml += '<button type="button" class="legend-model' + (disabledGraphCurves[singleKey] ? ' legend-off' : '') + '" data-key="' + escHtml(singleKey) + '" title="Toggle curve"><b>' + escHtml(singleDisp) + '</b><span class="legend-swatch" style="background:' + singleColor + '"></span></button>';
          continue;
        }
        var tagSpans = '';
        for (var tj2 = 0; tj2 < tagOrder.length; tj2++) {
          var tagName = tagOrder[tj2];
          // Check this tag has a curve
          var tagHasCurve = false;
          for (var ci3 = 0; ci3 < curves.length; ci3++) {
            if (curves[ci3][0] === modelName && curves[ci3][1] === tagName) { tagHasCurve = true; break; }
          }
          if (!tagHasCurve) continue;
          var itemKey = modelName + '|' + tagName;
          var tagColor = curveColorMap[itemKey] || palette[0];
          tagSpans += '<button type="button" class="legend-item' + (disabledGraphCurves[itemKey] ? ' legend-off' : '') + '" data-key="' + escHtml(itemKey) + '" title="Toggle curve">' + escHtml(tagName) + '<span class="legend-swatch" style="background:' + tagColor + '"></span></button>';
        }
        if (tagSpans) {
          var multiDisp = pathDisplayName(modelName);
          var focusedClass = (graphFocusModel === modelName) ? ' legend-focused' : '';
          legendHtml += '<span class="legend-model"><b data-focus-model="' + escHtml(modelName) + '" class="legend-focus' + focusedClass + '">' + escHtml(multiDisp) + ':</b>' + tagSpans + '</span>';
        }
      }
      if (droppedModels > 0) {
        legendHtml += '<span class="legend-more">+' + droppedModels + ' more model' + (droppedModels > 1 ? 's' : '') + '</span>';
      }
      graphLegend.innerHTML = legendHtml;
    }

    // Subtitle
    var sub = document.getElementById('graph-subtitle');
    if (sub) {
      if (graphFocusModel) {
        sub.textContent = 'Focused: ' + pathDisplayName(graphFocusModel);
      } else if (graphModelOverrideList) {
        sub.textContent = (window.GRAPH_MODEL_TITLE || shownModels[0]);
      } else if (shownModels.length < totalInViewWithData) {
        sub.textContent = 'Showing ' + shownModels.length + ' of ' + totalInViewWithData + ' models in view';
      } else {
        sub.textContent = shownModels.length + ' models in view';
      }
      graphNormalSubtitle = sub.textContent;
    }
  }

  // Inject SVG content via namespace-safe DOMParser
  var svgNS = 'http://www.w3.org/2000/svg';
  var wrapper = '<svg xmlns="' + svgNS + '">' + svgContent + '</svg>';
  var parser = new DOMParser();
  var doc = parser.parseFromString(wrapper, 'image/svg+xml');
  while (graphSvg.firstChild) graphSvg.removeChild(graphSvg.firstChild);
  var node;
  while ((node = doc.documentElement.firstChild)) {
    graphSvg.appendChild(node);
  }

  // Re-attach tooltip handlers
  if (graphTooltip) {
    var dotEls = graphSvg.querySelectorAll('.graph-dot');
    for (var i = 0; i < dotEls.length; i++) {
      dotEls[i].addEventListener('mouseenter', function(e) {
        var modelName = this.getAttribute('data-model');
        var tagName = this.getAttribute('data-tag');
        var ctx = parseInt(this.getAttribute('data-ctx'));
        var gib = parseFloat(this.getAttribute('data-gib'));
        graphTooltip.innerHTML = '<span class="font-medium text-neutral-800 dark:text-neutral-200">' + escHtml(pathDisplayName(modelName)) + ':' + escHtml(tagName) + '</span> <span class="text-neutral-500 dark:text-neutral-400">@ ' + ctxLabel(ctx) + ' ctx</span><br><span class="text-neutral-700 dark:text-neutral-300">' + fmtGiB(gib) + ' GiB KV cache</span>';
        graphTooltip.classList.remove('hidden');
      });
      dotEls[i].addEventListener('mousemove', function(e) {
        // Place the popup to the right of the cursor, unless the cursor is in
        // the right part of the viewport — then flip it to the left so it never
        // gets clipped by the screen edge.
        var tipW = 240;  // rough tooltip width
        var flip = (e.clientX + 12 + tipW) > window.innerWidth - 8;
        graphTooltip.style.left = (flip ? (e.clientX - 12 - tipW) : (e.clientX + 12)) + 'px';
        graphTooltip.style.top = (e.clientY - 10) + 'px';
      });
      dotEls[i].addEventListener('mouseleave', function(e) {
        graphTooltip.classList.add('hidden');
      });
      dotEls[i].addEventListener('mouseenter', function(e) {
        graphHoverKey = this.getAttribute('data-model') + '|' + this.getAttribute('data-tag');
        applyHoverDim();
      });
      dotEls[i].addEventListener('mouseleave', function(e) {
        graphHoverKey = null;
        applyHoverDim();
      });
    }
  }
  applyHoverDim();
}

function initGraph() {
  graphPanel = document.getElementById('graph-panel');
  graphSvg = document.getElementById('graph-svg');
  graphLegend = document.getElementById('graph-legend');
  graphTooltip = document.getElementById('graph-tooltip');
  if (!graphPanel || !graphSvg) return;

  var graphModelOverride = window.GRAPH_MODEL;
  var modelPageMode = (typeof graphModelOverride === 'string' && graphModelOverride.length > 0);

  // Legend toggles: one delegated listener survives every innerHTML rebuild.
  if (graphLegend) {
    graphLegend.addEventListener('click', function(e) {
      // Model focus toggle (clicking the model name header)
      var focusEl = (e.target && e.target.closest) ? e.target.closest('[data-focus-model]') : null;
      if (focusEl && graphLegend.contains(focusEl)) {
        var mname = focusEl.getAttribute('data-focus-model');
        graphFocusModel = (graphFocusModel === mname) ? null : mname;
        applyHoverDim();
        var sub = document.getElementById('graph-subtitle');
        if (sub) sub.textContent = graphFocusModel ? ('Focused: ' + graphFocusModel) : graphNormalSubtitle;
        return;
      }
      // Curve toggle
      var el = (e.target && e.target.closest) ? e.target.closest('[data-key]') : null;
      if (!el || !graphLegend.contains(el)) return;
      var key = el.getAttribute('data-key');
      if (disabledGraphCurves[key]) {
        delete disabledGraphCurves[key];
      } else {
        disabledGraphCurves[key] = true;
      }
      renderGraph();
    });
    graphLegend.addEventListener('mouseover', function(e) {
      var el = (e.target && e.target.closest) ? e.target.closest('[data-key]') : null;
      if (!el || !graphLegend.contains(el)) return;
      graphHoverKey = el.getAttribute('data-key');
      applyHoverDim();
    });
    graphLegend.addEventListener('mouseout', function(e) {
      var el = (e.target && e.target.closest) ? e.target.closest('[data-key]') : null;
      if (!el || !graphLegend.contains(el)) return;
      graphHoverKey = null;
      applyHoverDim();
    });
  }

  // All / None toggle buttons — mutually exclusive active states.
  // "All" is active by default (all curves visible). "None" active means
  // all curves disabled. Active style uses bg fill only (no font-weight
  // change, so button width stays constant).
  var btnAll = document.getElementById('graph-all');
  var btnNone = document.getElementById('graph-none');
  var btnLogY = document.getElementById('graph-logy');
  function setToggleActive(btn, active) {
    if (!btn) return;
    btn.classList.toggle('bg-neutral-100', active);
    btn.classList.toggle('dark:bg-neutral-800', active);
  }
  // "All" starts active
  setToggleActive(btnAll, true);
  if (btnAll) btnAll.addEventListener('click', function() {
    disabledGraphCurves = {};
    setToggleActive(btnAll, true);
    setToggleActive(btnNone, false);
    renderGraph();
  });
  if (btnLogY) btnLogY.addEventListener('click', function() {
    graphLogY = !graphLogY;
    setToggleActive(btnLogY, graphLogY);
    renderGraph();
  });

  // Context-range slider: custom single-handle slider copied from the filter
  // panel's style. Drag the handle to cap the visible x-range so cramped
  // charts can be zoomed into the region the user cares about.
  var graphRangeHandle = document.getElementById('graph-range-handle');
  var graphRangeFill = document.getElementById('graph-range-fill');
  var graphRangeTrack = document.getElementById('graph-range-track');

  // Breakpoints: 13 log2 steps (4K..8M) + Full. Each gets equal slider width.
  var GRAPH_RANGE_BP = [
    {pct: 0,       val: 0},
    {pct: 7.692,   val: 4096},
    {pct: 15.385,  val: 8192},
    {pct: 23.077,  val: 16384},
    {pct: 30.769,  val: 32768},
    {pct: 38.462,  val: 65536},
    {pct: 46.154,  val: 131072},
    {pct: 53.846,  val: 262144},
    {pct: 61.538,  val: 524288},
    {pct: 69.231,  val: 1048576},
    {pct: 76.923,  val: 2097152},
    {pct: 84.615,  val: 4194304},
    {pct: 92.308,  val: 8388608},
    {pct: 100,     val: 0}  // 0 = "Full" (no cap)
  ];
  function graphRangeValToPct(v) {
    for (var i = 0; i < GRAPH_RANGE_BP.length - 1; i++) {
      var a = GRAPH_RANGE_BP[i], b = GRAPH_RANGE_BP[i + 1];
      if (v >= a.val && v <= b.val) return a.pct + (v - a.val) / (b.val - a.val) * (a.pct === b.pct ? 1 : (b.pct - a.pct));
    }
    return v <= 0 ? 0 : 100;
  }
  function graphRangePctToVal(pct) {
    for (var i = 0; i < GRAPH_RANGE_BP.length - 1; i++) {
      var a = GRAPH_RANGE_BP[i], b = GRAPH_RANGE_BP[i + 1];
      if (pct >= a.pct && pct <= b.pct) {
        if (a.pct === b.pct) return a.val;
        return Math.round(a.val + (pct - a.pct) / (b.pct - a.pct) * (b.val - a.val));
      }
    }
    return pct <= 0 ? 0 : 0;
  }
  function updateGraphRangeUI() {
    var pct = graphCtxCap === 0 ? 100 : graphRangeValToPct(graphCtxCap);
    if (graphRangeFill) graphRangeFill.style.width = pct + '%';
    if (graphRangeHandle) graphRangeHandle.style.left = pct + '%';
    // Toggle active dots
    var container = document.getElementById('graph-range-container');
    if (container) {
      container.querySelectorAll('.slider-dot').forEach(function(dot) {
        var dotPct = parseFloat(dot.style.left);
        dot.classList.toggle('active', dotPct <= pct);
      });
    }
  }
  if (graphRangeHandle) {
    var draggingGraphRange = false;
    function onGraphRangeDrag(e) {
      if (!draggingGraphRange || !graphRangeTrack) return;
      e.preventDefault();
      var rect = graphRangeTrack.getBoundingClientRect();
      var x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
      var pct = Math.max(0, Math.min(1, x / rect.width)) * 100;
      // Snap to nearest breakpoint
      var bestIdx = 0, bestDist = Infinity;
      for (var i = 0; i < GRAPH_RANGE_BP.length; i++) {
        var d = Math.abs(GRAPH_RANGE_BP[i].pct - pct);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      graphCtxCap = GRAPH_RANGE_BP[bestIdx].val;
      updateGraphRangeUI();
      renderGraph();
    }
    graphRangeHandle.addEventListener('mousedown', function(e) { draggingGraphRange = true; e.preventDefault(); });
    graphRangeHandle.addEventListener('touchstart', function(e) { draggingGraphRange = true; e.preventDefault(); });
    document.addEventListener('mousemove', onGraphRangeDrag);
    document.addEventListener('mouseup', function() { draggingGraphRange = false; });
    document.addEventListener('touchmove', onGraphRangeDrag);
    document.addEventListener('touchend', function() { draggingGraphRange = false; });
    // Click on tick labels to jump
    var rangeContainer = document.getElementById('graph-range-container');
    if (rangeContainer) {
      rangeContainer.querySelectorAll('[data-ctx-tick], span').forEach(function(el) {
        // Tick label clicks handled below via the text spans
      });
    }
    updateGraphRangeUI();
  }
  if (btnNone) btnNone.addEventListener('click', function() {
    // Disable every curve for models currently in view
    var vis = getVisibleModelsInOrder();
    for (var vi = 0; vi < vis.length; vi++) {
      var m = getModelEntry(vis[vi]);
      if (!m || !m.tags) continue;
      var tns = Object.keys(m.tags);
      for (var ti = 0; ti < tns.length; ti++) {
        disabledGraphCurves[vis[vi] + '|' + tns[ti]] = true;
      }
    }
    setToggleActive(btnAll, false);
    setToggleActive(btnNone, true);
    renderGraph();
  });

  // Re-render graph curves when the theme (dark class) changes, so
  // curve/legend colors stay legible in both light and dark mode.
  if (typeof MutationObserver !== 'undefined') {
    var themeObserver = new MutationObserver(function(mutations) {
      for (var mi = 0; mi < mutations.length; mi++) {
        if (mutations[mi].attributeName === 'class') {
          scheduleGraphRender();
          return;
        }
      }
    });
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  }

  // Re-render when the panel/SVG resizes (full-height flex layout)
  if (typeof ResizeObserver !== 'undefined' && graphSvg) {
    var graphResizeObserver = new ResizeObserver(function() {
      scheduleGraphRender();
    });
    graphResizeObserver.observe(graphSvg);
  }

  // --- Model-page mode: render a single model's tags, no observer/scroll ---
  if (modelPageMode) {
    var modelUrl = window.GRAPH_DATA_URL;
    if (!modelUrl) return; // panel stays display:none via CSS
    fetch(modelUrl).then(function(r) { return r.json(); }).then(function(data) {
      graphData = data;
      if (!graphData || !graphData.models) return; // stays hidden
      var key = graphModelOverride.toLowerCase();
      // Prefer path-keyed entry (per-path tags, not merged) for detail pages;
      // fall back to name-keyed models for older graph-data.json without by_path.
      var entry = null;
      if (graphData.by_path && graphData.by_path[key]) {
        entry = graphData.by_path[key];
      } else if (graphData.models[key]) {
        entry = graphData.models[key];
      }
      if (!entry || !entry.tags || Object.keys(entry.tags).length === 0) return; // stays hidden
      graphOverrideEntry = entry;
      graphModelOverrideList = [key];
      graphPanel.classList.add('graph-ready');
      renderGraph();
    }).catch(function() {
      // stays hidden — no data, no flash
    });
    return;
  }

  // Scroll-based layout shift: when the filter sidebar has scrolled fully out
  // of view, relax the results list leftward and widen the graph panel into
  // the freed space (relevant in the >=1500px tier; no-op otherwise).
  var filtersOffscreen = false;
  var layoutTicking = false;
  function updateFiltersOffscreen() {
    if (document.body.classList.contains('filters-hidden')) return; // manual toggle owns the layout
    var topRow = document.getElementById('top-row');
    if (!topRow) return;
    var gone = topRow.getBoundingClientRect().bottom <= 0;
    if (gone !== filtersOffscreen) {
      filtersOffscreen = gone;
      document.body.classList.toggle('filters-offscreen', gone);
    }
  }
  // Manual hide/show: hides the sidebar and applies the expanded graph layout
  // regardless of scroll position.
  var filtersToggle = document.getElementById('graph-filters-toggle');
  if (filtersToggle) filtersToggle.addEventListener('click', function() {
    var hidden = document.body.classList.toggle('filters-hidden');
    filtersToggle.textContent = hidden ? 'Show filters' : 'Hide filters';
    if (hidden) {
      filtersOffscreen = false;
      document.body.classList.remove('filters-offscreen');
    } else {
      updateFiltersOffscreen();
    }
  });
  // Hide/show the graph panel. When hidden, the filter sidebar re-appears
  // in its normal position.
  var graphHideToggle = document.getElementById('graph-hide-toggle');
  if (graphHideToggle) graphHideToggle.addEventListener('click', function() {
    var hidden = document.body.classList.toggle('graph-hidden');
    graphHideToggle.textContent = hidden ? 'Show graph' : 'Hide graph';
    if (hidden) {
      // Restore filters if they were hidden
      document.body.classList.remove('filters-hidden');
      document.body.classList.remove('filters-offscreen');
      filtersOffscreen = false;
      if (filtersToggle) filtersToggle.textContent = 'Hide filters';
    } else {
      updateFiltersOffscreen();
    }
    if (typeof scheduleGraphRender === 'function') scheduleGraphRender();
  });
  window.addEventListener('scroll', function() {
    if (!layoutTicking) {
      layoutTicking = true;
      requestAnimationFrame(function() {
        layoutTicking = false;
        updateFiltersOffscreen();
      });
    }
  }, { passive: true });
  window.addEventListener('resize', updateFiltersOffscreen);
  updateFiltersOffscreen();

  var url = window.GRAPH_DATA_URL;
  if (!url) {
    graphPanel.style.display = 'none';
    return;
  }

  // Skip the graph-data fetch (<100KB) and observer setup entirely when the
  // panel can never be visible on this viewport. The graph only renders at
  // >=1080px (graph tier + >=1500px tiers); mobile/tablet users never see it.
  // If the viewport is widened later, load lazily on first resize past 1080px.
  var GRAPH_MIN_VW = 1080;
  var indexGraphLoaded = false;
  function maybeLoadIndexGraph() {
    if (indexGraphLoaded || window.innerWidth < GRAPH_MIN_VW) return;
    indexGraphLoaded = true;
    window.removeEventListener('resize', maybeLoadIndexGraph);
    loadIndexGraphData();
  }
  function loadIndexGraphData() {
  fetch(url).then(function(r) { return r.json(); }).then(function(data) {
    graphData = data;
    if (!graphData || !graphData.models || Object.keys(graphData.models).length === 0) {
      graphPanel.style.display = 'none';
      return;
    }
    // Set up IntersectionObserver over all model cards
    var cardList = document.getElementById('card-list');
    if (!cardList) {
      graphPanel.style.display = 'none';
      return;
    }
    var cards = cardList.querySelectorAll('li[x-test-model]');
    graphObserver = new IntersectionObserver(function(entries) {
      var changed = false;
      for (var i = 0; i < entries.length; i++) {
        var entry = entries[i];
        var pathKey = entry.target.getAttribute('data-path');
        if (!pathKey) continue;
        var isVis = entry.isIntersecting && entry.target.style.display !== 'none';
        var idx = visibleModels.indexOf(pathKey);
        if (isVis && idx === -1) {
          visibleModels.push(pathKey);
          changed = true;
        } else if (!isVis && idx !== -1) {
          visibleModels.splice(idx, 1);
          // Reset disabled curves for this model so they re-appear on return
          var prefix = pathKey + '|';
          for (var dk in disabledGraphCurves) {
            if (dk.indexOf(prefix) === 0) delete disabledGraphCurves[dk];
          }
          changed = true;
        }
      }
      if (changed) scheduleGraphRender();
    }, { threshold: 0 });
    for (var i = 0; i < cards.length; i++) {
      graphObserver.observe(cards[i]);
    }
    // Initial render
    renderGraph();
  }).catch(function() {
    graphPanel.style.display = 'none';
  });
  }
  window.addEventListener('resize', maybeLoadIndexGraph);
  maybeLoadIndexGraph();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


# Models to exclude from the build entirely. Each entry is a model path
# (as stored in models.json, e.g. "/x/canary" or "/frob/mixtao"). This is a
# single unified ignore list used by load_models(), the profile-page builder,
# and the main detail/tag build loop. Add "username/model" or "x/model" paths
# here to suppress a model from every page (search, profile, detail, tags).
IGNORELIST = {
    "/x/canary",  # ollama.com/x canary/test model — not a real model
    "/frob/whyhow-ai_PatientSeek",
    "/frob/mixtao",
    "/frob/NireeskshanAI_Niri",
}


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

    # Avatar image — use downloaded file if available
    avatar_file = None
    for ext in [".png", ".jpg"]:
        candidate = f"{username}-profile{ext}"
        if (PUBLIC / "assets" / candidate).exists():
            avatar_file = candidate
            break
    avatar_src = url(f"/assets/{avatar_file}") if avatar_file else ""

    # Model data: use profile's embedded card data, fall back to models.json
    all_models = load_models()
    models_by_path = {m["path"]: m for m in all_models}
    profile_models = []
    for m in model_paths:
        if isinstance(m, dict):
            if m.get("path") in IGNORELIST:
                continue
            profile_models.append(m)
        elif isinstance(m, str) and m in models_by_path:
            if m not in IGNORELIST:
                profile_models.append(models_by_path[m])

    # Build model cards (reuse render_card)
    # Compute profile-specific ranks: popular = pulls desc, newest = updated_title desc.
    # Merge with global ranks so models present there keep their global ranks; the
    # rest get local ranks so the sort dropdown works on the profile page too.
    global_ranks = load_ranks()
    # Profile ranks keyed by path for non-official models
    profile_ranks = {}

    # Popular rank from profile page ordering (pulls descending)
    popular_order = sorted(
        profile_models,
        key=lambda m: m.get("pulls", 0),
        reverse=True,
    )
    for rank, m in enumerate(popular_order):
        profile_ranks.setdefault(m["path"], {})["popular_rank"] = rank

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
        profile_ranks.setdefault(m["path"], {})["newest_rank"] = rank

    # Updated rank
    for rank, m in enumerate(newest_order):
        profile_ranks.setdefault(m["path"], {})["updated_rank"] = rank

    # Oldest rank (reverse of newest)
    for rank, m in enumerate(reversed(newest_order)):
        profile_ranks.setdefault(m["path"], {})["oldest_rank"] = rank

    # Default server-side order: popular (pulls descending) — matches ?sort=popular default
    sorted_models = sorted(
        profile_models,
        key=lambda m: profile_ranks.get(m["path"], {}).get("popular_rank", 9999),
    )
    cards_html = ""
    for m in sorted_models:
        tags = load_tags(m["path"], m)
        cards_html += render_card(m, tags, global_ranks, profile_ranks)

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
    <script>window.IS_PROFILE_PAGE = true;</script>
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
          <img src="{avatar_src}" alt="profile" class="absolute inset-0 h-full w-full border border-neutral-300 object-cover rounded-full" />
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


def build_x_page() -> None:
    """Build the /x page mirroring ollama.com/x — Ollama's experimental models.

    Layout matches ollama.com/x: a namespace header (avatar + name "x" +
    "Experimental Ollama models"), a sort dropdown (Popular / Newest), and the
    list of /x model cards. The canary model is excluded via IGNORELIST.
    """
    all_models = load_models()
    x_models = [m for m in all_models if m["path"].strip("/").startswith("x/")]

    # Read bio/avatar from the scraped profile_x.json (saved by the scraper
    # alongside the /x model cards). Falls back to defaults if absent.
    bio = "Experimental Ollama models"
    profile_x = HERE / "scraper" / "profile_x.json"
    if profile_x.exists():
        try:
            _px = json.loads(profile_x.read_text())
            if _px.get("bio"):
                bio = _px["bio"]
        except Exception:
            pass
    bio_esc = esc(bio)

    global_ranks = load_ranks()
    # Local ranks for the /x page ordering (by pulls for popular, by
    # updated_title for newest), so the sort dropdown works here too.
    x_ranks: dict[str, dict] = {}

    from datetime import datetime as _dt

    def _parse_updated(s: str) -> _dt:
        try:
            return _dt.strptime(s, "%b %d, %Y %I:%M %p UTC")
        except Exception:
            return _dt.min

    popular_order = sorted(x_models, key=lambda m: m.get("pulls", 0), reverse=True)
    for rank, m in enumerate(popular_order):
        x_ranks.setdefault(m["path"], {})["popular_rank"] = rank
    newest_order = sorted(
        x_models,
        key=lambda m: _parse_updated(m.get("updated_title") or ""),
        reverse=True,
    )
    for rank, m in enumerate(newest_order):
        x_ranks.setdefault(m["path"], {})["newest_rank"] = rank
        x_ranks.setdefault(m["path"], {})["updated_rank"] = rank
    for rank, m in enumerate(reversed(newest_order)):
        x_ranks.setdefault(m["path"], {})["oldest_rank"] = rank

    sorted_models = sorted(
        x_models,
        key=lambda m: x_ranks.get(m["path"], {}).get("popular_rank", 9999),
    )
    cards_html = ""
    for m in sorted_models:
        tags = load_tags(m["path"], m)
        cards_html += render_card(m, tags, global_ranks, x_ranks)

    if not cards_html:
        cards_html = '<p class="text-neutral-500 dark:text-neutral-400 py-8">No models found.</p>'

    sort_options = [("popular", "Popular"), ("newest", "Newest")]
    opt_html = "\n".join(
        f'        <option value="{v}">{l}</option>' for v, l in sort_options
    )

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html("x", bio)}
    <script>window.IS_PROFILE_PAGE = true;</script>
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html("")}

<main class="mx-auto flex w-full max-w-2xl flex-col px-6 py-5 md:py-12 lg:px-8">
  <div class="grid grid-cols-4 gap-4 md:gap-0">
    <div class="col-span-1">
      <div class="flex w-20 flex-col items-center md:w-28">
        <div class="group relative h-20 w-20 overflow-hidden rounded-full md:h-28 md:w-28">
          <img src="{url("/assets/x-profile.png")}" alt="profile" class="absolute inset-0 h-full w-full border border-neutral-300 object-cover rounded-full" />
        </div>
      </div>
    </div>
    <div class="col-span-3">
      <div class="flex flex-grow flex-col">
        <div class="flex flex-row items-center justify-between">
          <span class="text-[28px] font-medium tracking-tight">x</span>
        </div>
        <div class="space-y-1">
          <div class="my-2">
            <h2 class="break-words sm:max-w-lg">
              <span>{bio_esc}</span>
            </h2>
          </div>
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

    out_dir = PUBLIC / "x"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page)
    print(f"  /x page: {len(x_models)} models")


# --------------------------------------------------------------------------- #
# Download page (/download)
# --------------------------------------------------------------------------- #


def _absolutize(href: str) -> str:
    """Make a relative ollama.com URL absolute. Leaves http(s)/mailto links as-is."""
    if not href:
        return href
    if href.startswith(("http://", "https://", "mailto:", "#")):
        return href
    if href.startswith("/"):
        return "https://ollama.com" + href
    return "https://ollama.com/" + href


# OS icon SVGs matching ollama.com
_OS_SVGS = {
    "mac": (
        '<svg fill="currentColor" stroke-width="0" viewBox="0 0 1024 1024" '
        'class="h-8 w-8 p-1" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M747.4 535.7c-.4-68.2 30.5-119.6 92.9-157.5-34.9-50-87.7-77.5-157.3-82.8-65.9-5.2-138 38.4-164.4 38.4-27.9 0-91.7-36.6-141.9-36.6C273.1 298.8 163 379.8 163 544.6c0 48.7 8.9 99 26.7 150.8 23.8 68.2 109.6 235.3 199.1 232.6 46.8-1.1 79.9-33.2 140.8-33.2 59.1 0 89.7 33.2 141.9 33.2 90.3-1.3 167.9-153.2 190.5-221.6-121.1-57.1-114.6-167.2-114.6-170.7zm-105.1-305c50.7-60.2 46.1-115 44.6-134.7-44.8 2.6-96.6 30.5-126.1 64.8-32.5 36.8-51.6 82.3-47.5 133.6 48.4 3.7 92.6-21.2 129-63.7z"></path>'
        "</svg>"
    ),
    "linux": (
        '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 448 512" '
        'class="h-8 w-8 p-0.5" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M220.8 123.3c1 .5 1.8 1.7 3 1.7 1.1 0 2.8-.4 2.9-1.5.2-1.4-1.9-2.3-3.2-2.9-1.7-.7-3.9-1-5.5-.1-.4.2-.8.7-.6 1.1.3 1.3 2.3 1.1 3.4 1.7zm-21.9 1.7c1.2 0 2-1.2 3-1.7 1.1-.6 3.1-.4 3.5-1.6.2-.4-.2-.9-.6-1.1-1.6-.9-3.8-.6-5.5.1-1.3.6-3.4 1.5-3.2 2.9.1 1 1.8 1.5 2.8 1.4zM420 403.8c-3.6-4-5.3-11.6-7.2-19.7-1.8-8.1-3.9-16.8-10.5-22.4-1.3-1.1-2.6-2.1-4-2.9-1.3-.8-2.7-1.5-4.1-2 9.2-27.3 5.6-54.5-3.7-79.1-11.4-30.1-31.3-56.4-46.5-74.4-17.1-21.5-33.7-41.9-33.4-72C311.1 85.4 315.7.1 234.8 0 132.4-.2 158 103.4 156.9 135.2c-1.7 23.4-6.4 41.8-22.5 64.7-18.9 22.5-45.5 58.8-58.1 96.7-6 17.9-8.8 36.1-6.2 53.3-6.5 5.8-11.4 14.7-16.6 20.2-4.2 4.3-10.3 5.9-17 8.3s-14 6-18.5 14.5c-2.1 3.9-2.8 8.1-2.8 12.4 0 3.9.6 7.9 1.2 11.8 1.2 8.1 2.5 15.7.8 20.8-5.2 14.4-5.9 24.4-2.2 31.7 3.8 7.3 11.4 10.5 20.1 12.3 17.3 3.6 40.8 2.7 59.3 12.5 19.8 10.4 39.9 14.1 55.9 10.4 11.6-2.6 21.1-9.6 25.9-20.2 12.5-.1 26.3-5.4 48.3-6.6 14.9-1.2 33.6 5.3 55.1 4.1.6 2.3 1.4 4.6 2.5 6.7v.1c8.3 16.7 23.8 24.3 40.3 23 16.6-1.3 34.1-11 48.3-27.9 13.6-16.4 36-23.2 50.9-32.2 7.4-4.5 13.4-10.1 13.9-18.3.4-8.2-4.4-17.3-15.5-29.7zM223.7 87.3c9.8-22.2 34.2-21.8 44-.4 6.5 14.2 3.6 30.9-4.3 40.4-1.6-.8-5.9-2.6-12.6-4.9 1.1-1.2 3.1-2.7 3.9-4.6 4.8-11.8-.2-27-9.1-27.3-7.3-.5-13.9 10.8-11.8 23-4.1-2-9.4-3.5-13-4.4-1-6.9-.3-14.6 2.9-21.8zM183 75.8c10.1 0 20.8 14.2 19.1 33.5-3.5 1-7.1 2.5-10.2 4.6 1.2-8.9-3.3-20.1-9.6-19.6-8.4.7-9.8 21.2-1.8 28.1 1 .8 1.9-.2-5.9 5.5-15.6-14.6-10.5-52.1 8.4-52.1zm-13.6 60.7c6.2-4.6 13.6-10 14.1-10.5 4.7-4.4 13.5-14.2 27.9-14.2 7.1 0 15.6 2.3 25.9 8.9 6.3 4.1 11.3 4.4 22.6 9.3 8.4 3.5 13.7 9.7 10.5 18.2-2.6 7.1-11 14.4-22.7 18.1-11.1 3.6-19.8 16-38.2 14.9-3.9-.2-7-1-9.6-2.1-8-3.5-12.2-10.4-20-15-8.6-4.8-13.2-10.4-14.7-15.3-1.4-4.9 0-9 4.2-12.3zm3.3 334c-2.7 35.1-43.9 34.4-75.3 18-29.9-15.8-68.6-6.5-76.5-21.9-2.4-4.7-2.4-12.7 2.6-26.4v-.2c2.4-7.6.6-16-.6-23.9-1.2-7.8-1.8-15 .9-20 3.5-6.7 8.5-9.1 14.8-11.3 10.3-3.7 11.8-3.4 19.6-9.9 5.5-5.7 9.5-12.9 14.3-18 5.1-5.5 10-8.1 17.7-6.9 8.1 1.2 15.1 6.8 21.9 16l19.6 35.6c9.5 19.9 43.1 48.4 41 68.9zm-1.4-25.9c-4.1-6.6-9.6-13.6-14.4-19.6 7.1 0 14.2-2.2 16.7-8.9 2.3-6.2 0-14.9-7.4-24.9-13.5-18.2-38.3-32.5-38.3-32.5-13.5-8.4-21.1-18.7-24.6-29.9s-3-23.3-.3-35.2c5.2-22.9 18.6-45.2 27.2-59.2 2.3-1.7.8 3.2-8.7 20.8-8.5 16.1-24.4 53.3-2.6 82.4.6-20.7 5.5-41.8 13.8-61.5 12-27.4 37.3-74.9 39.3-112.7 1.1.8 4.6 3.2 6.2 4.1 4.6 2.7 8.1 6.7 12.6 10.3 12.4 10 28.5 9.2 42.4 1.2 6.2-3.5 11.2-7.5 15.9-9 9.9-3.1 17.8-8.6 22.3-15 7.7 30.4 25.7 74.3 37.2 95.7 6.1 11.4 18.3 35.5 23.6 64.6 3.3-.1 7 .4 10.9 1.4 13.8-35.7-11.7-74.2-23.3-84.9-4.7-4.6-4.9-6.6-2.6-6.5 12.6 11.2 29.2 33.7 35.2 59 2.8 11.6 3.3 23.7.4 35.7 16.4 6.8 35.9 17.9 30.7 34.8-2.2-.1-3.2 0-4.2 0 3.2-10.1-3.9-17.6-22.8-26.1-19.6-8.6-36-8.6-38.3 12.5-12.1 4.2-18.3 14.7-21.4 27.3-2.8 11.2-3.6 24.7-4.4 39.9-.5 7.7-3.6 18-6.8 29-32.1 22.9-76.7 32.9-114.3 7.2zm257.4-11.5c-.9 16.8-41.2 19.9-63.2 46.5-13.2 15.7-29.4 24.4-43.6 25.5s-26.5-4.8-33.7-19.3c-4.7-11.1-2.4-23.1 1.1-36.3 3.7-14.2 9.2-28.8 9.9-40.6.8-15.2 1.7-28.5 4.2-38.7 2.6-10.3 6.6-17.2 13.7-21.1.3-.2.7-.3 1-.5.8 13.2 7.3 26.6 18.8 29.5 12.6 3.3 30.7-7.5 38.4-16.3 9-.3 15.7-.9 22.6 5.1 9.9 8.5 7.1 30.3 17.1 41.6 10.6 11.6 14 19.5 13.7 24.6zM173.3 148.7c2 1.9 4.7 4.5 8 7.1 6.6 5.2 15.8 10.6 27.3 10.6 11.6 0 22.5-5.9 31.8-10.8 4.9-2.6 10.9-7 14.8-10.4s5.9-6.3 3.1-6.6-2.6 2.6-6 5.1c-4.4 3.2-9.7 7.4-13.9 9.8-7.4 4.2-19.5 10.2-29.9 10.2s-18.7-4.8-24.9-9.7c-3.1-2.5-5.7-5-7.7-6.9-1.5-1.4-1.9-4.6-4.3-4.9-1.4-.1-1.8 3.7 1.7 6.5z"></path>'
        "</svg>"
    ),
    "windows": (
        '<svg fill="currentColor" stroke-width="0" viewBox="0 0 448 512" '
        'class="h-8 w-8 p-1" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M0 93.7l183.6-25.3v177.4H0V93.7zm0 324.6l183.6 25.3V268.4H0v149.9zm203.8 28L448 480V268.4H203.8v177.9zm0-380.6v180.1H448V32L203.8 65.7z"></path>'
        "</svg>"
    ),
}

_CHECK_SVG_DL = (
    '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
    'stroke-width="1.5" stroke="currentColor" class="check-icon hidden h-5 w-5">'
    '<path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5"/>'
    "</svg>"
)
_COPY_SVG_DL = (
    '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
    'stroke-width="1.5" stroke="currentColor" class="copy-icon h-5 w-5">'
    '<path stroke-linecap="round" stroke-linejoin="round" '
    'd="M16.5 8.25V6a2.25 2.25 0 00-2.25-2.25H6A2.25 2.25 0 003.75 6v8.25'
    "A2.25 2.25 0 006 16.5h2.25m8.25-8.25H18a2.25 2.25 0 012.25 2.25V18A2.25 "
    "2.25 0 0118 20.25h-7.5A2.25 2.25 0 018.25 18v-1.5m8.25-8.25h-6a2.25 2.25 "
    '0 00-2.25 2.25v6"/>'
    "</svg>"
)


def build_download_page() -> None:
    """Build the /download page from scraper/download.json.

    Uses OS tab buttons (like ollama.com) with JS autodetect + click to switch.
    Only the active OS section is shown at a time.
    """
    data_path = SCRAPER / "download.json"
    if not data_path.exists():
        print("  download: no data, skipping")
        return
    data = json.loads(data_path.read_text())
    tabs = data.get("tabs", [])
    if not tabs:
        print("  download: no tabs, skipping")
        return

    # Build tab buttons + content sections
    tab_buttons = []
    content_sections = []
    os_slugs = []
    for tab in tabs:
        os_slug = tab.get("os", "")
        os_slugs.append(os_slug)
        label = esc(tab.get("label", os_slug))
        icon = _OS_SVGS.get(os_slug, "")

        # Tab button — data-os attribute for JS switching
        tab_buttons.append(
            f'<a data-os="{esc(os_slug)}" '
            f'class="dl-tab flex cursor-pointer flex-col items-center rounded-lg px-6 py-2 '
            f'hover:bg-neutral-100 dark:hover:bg-neutral-900 transition-colors" '
            f'href="#{esc(os_slug)}">'
            f"{icon}"
            f"{label}</a>"
        )

        # Content section
        command = tab.get("command", "")
        helper = tab.get("helper", "")
        or_sep = tab.get("or_separator", False)
        dl_url = tab.get("download_url", "")
        dl_label = tab.get("download_label", "")
        footnote = tab.get("footnote", "")

        cmd_esc = esc(command)
        cmd_block = ""
        if command:
            cmd_block = (
                '<pre class="language-none mb-2 flex flex-1 justify-center '
                "whitespace-nowrap rounded-lg bg-neutral-100 dark:bg-neutral-900 "
                'font-mono text-sm">'
                f'<code class="command flex-1 py-3 pl-4 pr-4 overflow-auto text-neutral-900 dark:text-neutral-100">{cmd_esc}</code>'
                '<button class="block py-1 px-3 leading-[0] w-12 text-neutral-500 '
                'hover:text-black dark:hover:text-white focus:outline-none" '
                'onclick="copyToClipboard(this)">'
                f"{_COPY_SVG_DL}"
                f"{_CHECK_SVG_DL}"
                "</button></pre>"
            )

        helper_html = (
            f'<p class="text-xs text-neutral-800 dark:text-neutral-300 mt-1">{esc(helper)}</p>'
            if helper
            else ""
        )
        or_html = (
            '<p class="my-2 text-xs text-neutral-800 dark:text-neutral-300">or</p>'
            if or_sep
            else ""
        )
        dl_html = ""
        if dl_url and dl_label:
            dl_href = esc(_absolutize(dl_url))
            dl_html = (
                f'<a class="w-full max-w-[16rem] rounded-3xl bg-neutral-800 dark:bg-neutral-200 '
                f"px-2 py-2 text-lg text-white dark:text-neutral-900 hover:bg-black "
                f'dark:hover:bg-white inline-block text-center" href="{dl_href}">'
                f"{esc(dl_label)}</a>"
            )
        footnote_html = (
            f'<p class="mt-4 text-xs text-neutral-800 dark:text-neutral-300">{esc(footnote)}</p>'
            if footnote
            else ""
        )

        content_sections.append(
            f'<div id="dl-{esc(os_slug)}" class="dl-section mx-auto mb-16 mt-12 flex w-full min-w-0 flex-col items-center text-center self-center hidden">\n'
            f'  <div class="min-w-0 max-w-full flex-1 self-center text-center">\n'
            f"    {cmd_block}\n"
            f"    {helper_html}\n"
            f"    {or_html}\n"
            f"    {dl_html}\n"
            f"  </div>\n"
            f"  {footnote_html}\n"
            f"</div>"
        )

    tabs_html = "\n".join(tab_buttons)
    sections_html = "\n".join(content_sections)
    os_array = ",".join(f'"{esc(s)}"' for s in os_slugs)

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html("Download Ollama", "Download Ollama on macOS, Linux, and Windows.")}
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html("")}
<main class="mx-auto flex w-full max-w-6xl flex-col items-center px-6 py-10 md:py-24">
  <h1 class="mb-12 text-3xl tracking-tight text-neutral-900 dark:text-neutral-100">Download Ollama</h1>
  <nav class="grid grid-cols-3 gap-4 text-sm">
{tabs_html}
  </nav>
{sections_html}
</main>
{footer_html()}
{theme_script()}
<script src="{url("/assets/app.js")}"></script>
<script>
var dlOSes = [{os_array}];
function showOS(os) {{
  document.querySelectorAll('.dl-tab').forEach(function(t) {{
    var active = t.getAttribute('data-os') === os;
    if (active) {{
      t.classList.add('bg-neutral-100','dark:bg-neutral-900');
    }} else {{
      t.classList.remove('bg-neutral-100','dark:bg-neutral-900');
    }}
  }});
  document.querySelectorAll('.dl-section').forEach(function(s) {{
    s.classList.toggle('hidden', s.id !== 'dl-' + os);
  }});
}}
document.querySelectorAll('.dl-tab').forEach(function(t) {{
  t.addEventListener('click', function(e) {{
    e.preventDefault();
    showOS(t.getAttribute('data-os'));
  }});
}});
// Autodetect OS — default to Linux on mobile, matching ollama.com behavior
var ua = navigator.userAgent;
var detected = 'linux';
if (/Mac|iPhone|iPad|iPod/.test(ua)) detected = 'mac';
else if (/Windows/.test(ua)) detected = 'windows';
// On mobile, ollama.com shows Linux regardless
if (/Android|iPhone|iPad|iPod|Mobile|webOS|BlackBerry/.test(ua)) detected = 'linux';
if (dlOSes.indexOf(detected) === -1) detected = dlOSes[0] || 'linux';
showOS(detected);
function copyToClipboard(btn) {{
  var code = btn.parentElement.querySelector('code.command');
  if (!code) return;
  var text = code.textContent;
  navigator.clipboard.writeText(text).then(function() {{
    var copyIcon = btn.querySelector('.copy-icon');
    var checkIcon = btn.querySelector('.check-icon');
    if (copyIcon && checkIcon) {{
      copyIcon.classList.add('hidden');
      checkIcon.classList.remove('hidden');
      setTimeout(function() {{
        copyIcon.classList.remove('hidden');
        checkIcon.classList.add('hidden');
      }}, 1500);
    }}
  }});
}}
</script>
</body>
</html>"""

    out = PUBLIC / "download"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)
    print(f"  download: {len(tabs)} OS sections")


# --------------------------------------------------------------------------- #
# Pricing page (/pricing)
# --------------------------------------------------------------------------- #

_CHECK_SVG = (
    '<svg class="w-4 h-4 text-neutral-400 dark:text-neutral-500 flex-shrink-0 mt-0.5" '
    'viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" '
    'd="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414'
    'L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>'
)


def _pricing_linkify(html_frag: str) -> str:
    """Rewrite internal ollama.com links in a scraped HTML fragment.

    /search?c=cloud -> site url(); /download -> site url(); /settings and
    /upgrade?... -> absolute https://ollama.com prefix; leaves mailto/external
    links untouched.
    """

    def repl(m):
        href = m.group(1)
        if (
            href.startswith("/search")
            or href == "/download"
            or href.startswith("/download?")
        ):
            return f'href="{url(href)}"'
        if href.startswith("/") and not href.startswith("//"):
            return f'href="https://ollama.com{href}"'
        return m.group(0)

    return re.sub(r'href="([^"]+)"', lambda m: repl(m), html_frag)


def build_pricing_page() -> None:
    """Build the /pricing page from scraper/pricing.json.

    Replicates ollama.com/pricing: tabbed layout (Individuals / Team & Enterprise),
    tier cards with images, badges, price subtext, paused buttons, and FAQ.
    """
    data_path = SCRAPER / "pricing.json"
    if not data_path.exists():
        print("  pricing: no data, skipping")
        return
    data = json.loads(data_path.read_text())
    tiers = data.get("tiers", [])
    faq = data.get("faq", [])
    if not tiers:
        print("  pricing: no tiers, skipping")
        return

    def _build_card(tier: dict, is_teams: bool) -> str:
        name = esc(tier.get("name", ""))
        price_raw = tier.get("price", "")
        # Wrap the suffix (everything after the first space following a $price)
        # in a smaller span. e.g. "$25 / seat / mo" -> "$25" + " / seat / mo"
        price_html = ""
        if price_raw:
            suffix_m = re.match(r"^(\$\d+(?:\.\d+)?)(.*)$", price_raw)
            if suffix_m:
                price_main = suffix_m.group(1)
                price_suffix = suffix_m.group(2).strip()
                suffix_span = (
                    f' <span class="text-base font-normal">{esc(price_suffix)}</span>'
                    if price_suffix
                    else ""
                )
                price_html = f'<div class="text-2xl font-semibold font-rounded text-neutral-900 dark:text-neutral-100">{esc(price_main)}{suffix_span}</div>'
            else:
                price_html = f'<div class="text-2xl font-semibold font-rounded text-neutral-900 dark:text-neutral-100">{esc(price_raw)}</div>'
        else:
            price_html = '<div class="text-2xl font-semibold font-rounded text-neutral-600 dark:text-neutral-400">—</div>'

        price_subtext_raw = tier.get("price_subtext", "")
        price_subtext_html = ""
        if price_subtext_raw:
            subtext = sanitize_readme_html(_pricing_linkify(price_subtext_raw))
            price_subtext_html = f'<p class="text-xs text-neutral-600 dark:text-neutral-400 m-0">{subtext}</p>'

        desc = esc(tier.get("description", ""))
        desc_html = (
            f'<p class="mb-4 text-neutral-700 dark:text-neutral-300">{desc}</p>'
            if desc
            else ""
        )

        image_url = tier.get("image_url", "")
        img_html = ""
        if image_url:
            img_cls = "h-20 pricing-tier-img" if is_teams else "h-16 pricing-tier-img"
            img_src = esc(_absolutize(image_url))
            img_html = (
                f'<img src="{img_src}" alt="Ollama" class="{img_cls} self-start mb-4">'
            )

        badge = esc(tier.get("badge", ""))
        badge_html = ""
        if badge:
            badge_html = f'<span class="inline-flex items-center whitespace-nowrap rounded-full border border-neutral-200 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-900 px-2.5 py-1 text-xs font-medium text-neutral-600 dark:text-neutral-400">{badge}</span>'

        btn_url_raw = tier.get("button_url", "")
        if btn_url_raw == "/download":
            btn_url = esc(url("/download"))
        else:
            btn_url = esc(_absolutize(btn_url_raw))
        btn_label = esc(tier.get("button_label", ""))
        button_paused = tier.get("button_paused", False)
        notice_raw = tier.get("notice", "")

        if button_paused:
            btn_mb = "mb-2" if notice_raw else "mb-6"
            btn_cls = (
                f"block w-full text-center border border-neutral-200 dark:border-neutral-800 "
                f"bg-neutral-100 dark:bg-neutral-900 text-neutral-500 dark:text-neutral-500 "
                f"font-medium py-2 px-6 rounded-full {btn_mb} cursor-not-allowed"
            )
            btn_html = (
                f'<div class="{btn_cls}" aria-disabled="true">{btn_label}</div>'
                if btn_label
                else ""
            )
            if notice_raw:
                notice_html = sanitize_readme_html(_pricing_linkify(notice_raw))
                btn_html += f'\n      <p class="text-xs text-neutral-500 dark:text-neutral-500 mb-6">{notice_html}</p>'
        elif tier.get("name") in ("Free", "Enterprise"):
            btn_cls = (
                "block w-full text-center border border-neutral-300 dark:border-neutral-700 "
                "hover:bg-neutral-50 dark:hover:bg-neutral-900 text-black dark:text-neutral-100 "
                "font-medium py-2 px-6 rounded-full mb-6"
            )
            btn_html = (
                f'<a href="{btn_url}" class="{btn_cls}">{btn_label}</a>'
                if btn_url
                else ""
            )
        else:
            btn_cls = (
                "block w-full text-center bg-neutral-800 dark:bg-neutral-200 "
                "hover:bg-black dark:hover:bg-white text-white dark:text-neutral-900 "
                "font-medium py-2 px-6 rounded-full mb-6"
            )
            btn_html = (
                f'<a href="{btn_url}" class="{btn_cls}">{btn_label}</a>'
                if btn_url
                else ""
            )

        features = tier.get("features", [])
        feat_items = "\n".join(
            f'        <li class="text-sm flex items-start gap-2 text-neutral-700 dark:text-neutral-300">\n'
            f"          {_CHECK_SVG}\n"
            f"          <span>{esc(f)}</span>\n"
            f"        </li>"
            for f in features
        )
        feats_subtitle = esc(tier.get("features_subtitle", ""))
        feats_subtitle_html = (
            f'<div class="text-sm font-medium mb-3 text-neutral-900 dark:text-neutral-100">{feats_subtitle}</div>'
            if feats_subtitle
            else ""
        )
        feats_html = (
            f'<ul class="space-y-3">\n{feat_items}\n      </ul>' if feat_items else ""
        )

        # Coming soon features (Team card: plain text items, no check icons)
        cs_features = tier.get("coming_soon_features", [])
        if cs_features:
            cs_items = "\n".join(
                f'        <li class="text-sm text-neutral-600 dark:text-neutral-400">{esc(f)}</li>'
                for f in cs_features
            )
            feats_html += (
                f'\n      <div class="text-sm font-medium mt-6 mb-3 text-neutral-900 dark:text-neutral-100">Coming soon:</div>\n'
                f'      <ul class="space-y-3">\n{cs_items}\n      </ul>'
            )

        name_html = f'<h2 class="text-3xl font-medium mb-2 text-neutral-900 dark:text-neutral-100">{name}</h2>'
        if badge_html:
            name_html = f'<div class="flex items-center gap-3 mb-2"><h2 class="text-3xl font-medium text-neutral-900 dark:text-neutral-100">{name}</h2>{badge_html}</div>'

        price_block = f'<div class="mb-3 min-h-[3rem]">{price_html}\n        {price_subtext_html}</div>'
        if is_teams:
            price_block = f'<div class="mb-5 min-h-[3rem]">{price_html}\n        {price_subtext_html}</div>'

        card_cls = (
            "md:col-span-2 flex flex-col border border-neutral-200 dark:border-neutral-800 rounded-3xl p-8"
            if not is_teams
            else "flex flex-col border border-neutral-200 dark:border-neutral-800 rounded-3xl p-8"
        )
        return f"""    <div class="{card_cls}">
      {img_html}
      {name_html}
      {desc_html}
      {price_block}
      {btn_html}
      {feats_subtitle_html}
      {feats_html}
    </div>"""

    # Split tiers into individuals (Free/Pro/Max) and teams (Team/Enterprise).
    individuals = [t for t in tiers if t.get("name") in ("Free", "Pro", "Max")]
    teams = [t for t in tiers if t.get("name") in ("Team", "Enterprise")]

    individuals_cards = "\n".join(_build_card(t, False) for t in individuals)
    teams_cards = "\n".join(_build_card(t, True) for t in teams)

    # FAQ groups.
    faq_groups = []
    faq_groups_teams = []
    for grp in faq:
        gname = esc(grp.get("group", ""))
        items = grp.get("items", [])
        item_html = []
        for it in items:
            q = esc(it.get("q", ""))
            a_raw = it.get("a", "")
            a_html = _pricing_linkify(a_raw) if a_raw else ""
            item_html.append(
                f"""        <li>
           <h4 class="text-base font-semibold mb-1 text-neutral-900 dark:text-neutral-100">{q}</h4>
           <div class="text-sm text-neutral-700 dark:text-neutral-300">{a_html}</div>
         </li>"""
            )
        items_block = "\n".join(item_html)
        block = f"""    <div>
      <h3 class="text-xl font-semibold font-rounded mb-4 text-neutral-900 dark:text-neutral-100">{gname}</h3>
      <ul class="space-y-5">
{items_block}
      </ul>
    </div>"""
        faq_groups.append(block)
        # Team FAQ group goes in the teams section
        if gname == "Team":
            faq_groups_teams.append(block)
    faq_html = "\n".join(faq_groups)
    faq_teams_html = "\n".join(faq_groups_teams)

    # Tab switcher JS (inline, matches ollama.com)
    tab_js = """  <script>
    (function () {
      var buttons = Array.from(document.querySelectorAll("[data-pricing-tab]"));
      var panels = {
        individuals: document.getElementById("pricing-individuals"),
        teams: document.getElementById("pricing-teams"),
      };
      var teamsFAQ = document.getElementById("pricing-teams-faq");
      var faqMain = document.getElementById("pricing-faq");
      var activeClasses = ["border-neutral-800", "bg-neutral-800", "text-white"];
      var inactiveClasses = ["border-transparent", "text-neutral-700", "dark:text-neutral-300", "hover:bg-white", "dark:hover:bg-neutral-900"];

      function selectPricingTab(name, updateHash) {
        buttons.forEach(function (button) {
          var active = button.dataset.pricingTab === name;
          button.setAttribute("aria-selected", active ? "true" : "false");
          button.tabIndex = active ? 0 : -1;
          activeClasses.forEach(function (className) {
            button.classList.toggle(className, active);
          });
          inactiveClasses.forEach(function (className) {
            button.classList.toggle(className, !active);
          });
        });
        Object.keys(panels).forEach(function (panelName) {
          panels[panelName].hidden = panelName !== name;
        });
        if (teamsFAQ) teamsFAQ.hidden = name !== "teams";
        if (faqMain) faqMain.hidden = name === "teams";
        // Sync html.tab-* class so CSS visibility rules stay in sync on click
        var htmlEl = document.documentElement;
        htmlEl.classList.toggle("tab-teams", name === "teams");
        htmlEl.classList.toggle("tab-individuals", name !== "teams");
        if (updateHash) {
          history.replaceState(null, "", name === "teams" ? "#teams" : "#individuals");
        }
      }

      buttons.forEach(function (button, index) {
        button.addEventListener("click", function () {
          selectPricingTab(button.dataset.pricingTab, true);
        });
        button.addEventListener("keydown", function (event) {
          var nextIndex = index;
          if (event.key === "ArrowRight") nextIndex = (index + 1) % buttons.length;
          if (event.key === "ArrowLeft") nextIndex = (index - 1 + buttons.length) % buttons.length;
          if (event.key === "Home") nextIndex = 0;
          if (event.key === "End") nextIndex = buttons.length - 1;
          if (nextIndex === index) return;
          event.preventDefault();
          buttons[nextIndex].focus();
          selectPricingTab(buttons[nextIndex].dataset.pricingTab, true);
        });
      });

      // Run immediately (before paint) to avoid FOUC
      selectPricingTab(window.location.hash === "#teams" ? "teams" : "individuals", false);
    })();
  </script>"""

    page = f"""<!DOCTYPE html>
<html lang="en" class="">
<head>
{head_html("Pricing", "Ollama pricing plans — Free, Pro, Max, Team, and Enterprise.")}
    {pricing_tab_script_head()}
</head>
<body class="antialiased min-h-screen w-full m-0 flex flex-col bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
{nav_html("")}
<main class="mx-auto flex w-full max-w-6xl flex-col px-6 py-12 md:py-20">
  <section class="text-center mb-8">
    <h1 class="text-4xl md:text-5xl font-semibold font-rounded tracking-tight mb-4 text-neutral-900 dark:text-neutral-100">Pricing</h1>
  </section>

  <div class="mb-10 flex justify-center" role="tablist" aria-label="Pricing for">
    <div class="inline-flex gap-1 rounded-full border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900 p-1">
      <button type="button" id="pricing-individuals-tab" role="tab" aria-selected="true" aria-controls="pricing-individuals" data-pricing-tab="individuals" class="rounded-full border border-neutral-800 bg-neutral-800 px-5 py-2 text-sm font-medium text-white focus:outline-none">Individuals</button>
      <button type="button" id="pricing-teams-tab" role="tab" aria-selected="false" aria-controls="pricing-teams" data-pricing-tab="teams" tabindex="-1" class="rounded-full border border-transparent px-5 py-2 text-sm font-medium text-neutral-700 dark:text-neutral-300 hover:bg-white dark:hover:bg-neutral-900 focus:outline-none">Team &amp; Enterprise</button>
    </div>
  </div>

  <section id="pricing-individuals" role="tabpanel" aria-labelledby="pricing-individuals-tab" tabindex="0" class="mb-12">
    <div class="grid grid-cols-1 md:grid-cols-6 gap-4">
{individuals_cards}
    </div>
  </section>

  <section id="pricing-teams" role="tabpanel" aria-labelledby="pricing-teams-tab" tabindex="0" class="mb-12" hidden>
    <div class="grid grid-cols-1 gap-4 md:grid-cols-2">
{teams_cards}
    </div>
  </section>

{tab_js}

  <section id="pricing-faq" class="flex flex-col w-full max-w-xl mx-auto mt-8">
    <h2 class="text-2xl font-semibold font-rounded mb-6 text-neutral-900 dark:text-neutral-100">Frequently asked questions</h2>
    <div class="space-y-10">
{faq_html}
    </div>
  </section>

  <div id="pricing-teams-faq" hidden>
    <div class="space-y-10">
{faq_teams_html}
    </div>
  </div>
</main>
{footer_html()}
{theme_script()}
<script src="{url("/assets/app.js")}"></script>
</body>
</html>"""

    out = PUBLIC / "pricing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)
    print(f"  pricing: {len(tiers)} tiers, {len(faq)} faq groups")


def main() -> int:
    global BASE
    if "--base" in sys.argv:
        idx = sys.argv.index("--base")
        BASE = sys.argv[idx + 1].rstrip("/") if idx + 1 < len(sys.argv) else ""
    models = load_models()
    print(f"loaded {len(models)} models from scraper/models.json")
    global _NEW_MODELS
    _NEW_MODELS = load_new_model_paths()
    if _NEW_MODELS:
        print(f"  {len(_NEW_MODELS)} new models this run")

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
    build_profile_page("frob")
    build_profile_page("huihui_ai")

    # /x page (Ollama's experimental image models)
    print("building /x page ...")
    build_x_page()

    # static standalone pages (/download + /pricing)
    print("building download + pricing pages ...")
    build_download_page()
    build_pricing_page()

    # Load profile models and add them to the build list
    _all_models = list(models)
    for _username in ["maternion", "frob", "huihui_ai"]:
        _pf = HERE / "scraper" / f"profile_{_username}.json"
        if _pf.exists():
            _pdata = json.loads(_pf.read_text())
            _existing_paths = {m["path"] for m in _all_models}
            for _m in _pdata.get("models", []):
                if isinstance(_m, dict) and _m["path"] not in _existing_paths:
                    if _m["path"] not in IGNORELIST:
                        _all_models.append(_m)

    # model detail + tags + per-tag pages
    print(f"building model detail + tags + tag pages ({len(_all_models)} models) ...")
    tag_pages_built = 0
    blob_pages_built = 0
    for i, m in enumerate(_all_models, 1):
        tags = load_tags(m["path"], m)
        build_detail(m, tags)
        build_tags_page(m, tags)
        # Build per-tag pages. Two cases produce blob pages:
        #  1. This tag has its own scraped tag page (tp) -> build blobs at this
        #     tag's path from its own file list.
        #  2. This tag shares a digest with a sibling tag (alias tags, e.g.
        #     :8b and :latest) but has no scraped tag page of its own. In that
        #     case reuse a sibling tag's file list, rewriting each blob URL to
        #     point at the current tag's path, so its blob pages exist too.
        #     Without this the alias tag's blob links 404 (the canonical blob
        #     data is stored once per digest under whichever tag was scraped
        #     first).
        tag_page_data: dict[str, dict | None] = {}
        for t in tags:
            tag_page_data[t["name"]] = load_tag_page(m["path"], t["name"])

        # /x model pages don't have per-tag detail pages (ollama.com's /x
        # models render tags as plain text — they're not clickable links and
        # have no tag pages). Skip building tag pages and blob pages for them;
        # only the model detail page and the /tags listing page are built.
        is_x_model = m["path"].strip("/").startswith("x/")
        if is_x_model:
            if i % 50 == 0:
                print(f"  {i}/{len(_all_models)}")
            continue

        for t in tags:
            tp = tag_page_data[t["name"]]
            build_tag_page(m, t, tp)
            tag_pages_built += 1

            files = None
            src_tag = t["name"]
            if tp and (tp.get("files") or tp.get("blobs")):
                files = tp.get("files") or tp.get("blobs") or []
            elif t.get("digest"):
                # No tag page for this tag — try an alias sibling (same digest)
                # that does have a tag page, and reuse its file list.
                dig = t["digest"]
                for sib in tags:
                    if sib.get("name") == t["name"]:
                        continue
                    if sib.get("digest") != dig:
                        continue
                    sib_tp = tag_page_data.get(sib["name"])
                    if sib_tp and (sib_tp.get("files") or sib_tp.get("blobs")):
                        files = sib_tp.get("files") or sib_tp.get("blobs") or []
                        src_tag = sib["name"]
                        break

            if files:
                # Current tag's path prefix, e.g. /library/lfm2.5:8b
                cur_prefix = (m["path"].strip("/") + ":" + t["name"]).rstrip("/")
                for f in files:
                    blob_url = f.get("blob_url") or f.get("url") or ""
                    if not blob_url:
                        continue
                    # If the file came from a sibling (alias) tag, rewrite the
                    # blob URL's tag segment to the current tag so the page is
                    # built at the current tag's path.
                    if src_tag != t["name"] and "/blobs/" in blob_url:
                        digest_seg = (
                            blob_url.rstrip("/")
                            .rsplit("/blobs/", 1)[-1]
                            .split("?", 1)[0]
                        )
                        blob_url = "/" + cur_prefix + "/blobs/" + digest_seg
                    bp = load_blob_page(blob_url)
                    if bp:
                        # Build at the current tag's blob path (target_blob_url)
                        # so alias tags that share a digest with another tag
                        # also get their own blob pages instead of only the
                        # canonical tag's copy (which would 404 otherwise).
                        build_blob_page(bp, target_blob_url=blob_url)
                        blob_pages_built += 1
        if i % 50 == 0:
            print(f"  {i}/{len(_all_models)}")
    print(
        f"built {len(_all_models)} model pages + tags pages + {tag_pages_built} tag pages + {blob_pages_built} blob pages"
    )

    # write the catalog JSON for client-side use (minimal slice for nav dropdown)
    nav_models = [
        {
            "name": m["name"],
            "description": m.get("description", ""),
            "path": m.get("path", f"/library/{m['name']}"),
            "pulls": m.get("pulls", 0),
        }
        for m in models
    ]
    (PUBLIC / "assets" / "models.json").write_text(
        json.dumps(nav_models, ensure_ascii=False, separators=(",", ":"))
    )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

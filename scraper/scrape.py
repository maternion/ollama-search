#!/usr/bin/env python3
"""Scraper for ollama.com model catalog.

Uses only stdlib + `requests` (already installed on this system) — no bs4/httpx.
Parsing relies on the stable `x-test-*` hooks and known class fragments in the
ollama.com markup.

Strategy (verified against ollama.com markup):
  - /library?sort=popular|newest  -> full OFFICIAL catalog, single page, 236
    models. Cards use x-test-model-title (not x-test-search-response-title).
  - /search?q=<term>               -> 20 mixed official+user models per query,
    paginated but pagination is broken (every ?page=N returns the same 20).
    Cards use x-test-search-response-title. Sweep a-z, 0-9, and common
    substrings to enumerate user models broadly. Each query is de-duplicated.
  - /<path>/tags                   -> per-model tag table. MLX tags use the
    `-mlx` suffix; GGUF is the default (no suffix). Registry manifests are
    NOT used (return 412/401 for MLX).

Output:
  scraper/models.json            catalog (list of model cards)
  scraper/tags/<slug>.json       per-model tag table
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

BASE = "https://ollama.com"
HERE = Path(__file__).resolve().parent
DATA = HERE
TAGS_DIR = HERE / "tags"
PAGES_DIR = HERE / "pages"
TAG_PAGES_DIR = HERE / "tag_pages"
BLOBS_DIR = HERE / "blobs"

# Polite crawling: small delay between requests, generous timeout.
DELAY = 0.5
TIMEOUT = 30.0
HARD_DEADLINE = 60.0  # wall-clock seconds per request (SIGALRM)
MAX_CONSECUTIVE_FAILURES = 10  # stop scraping after this many in a row
CHECKPOINT_EVERY = 5  # git-push checkpoint every N models
UA = "ollama-search-scraper/0.1 (+https://github.com/anomalyco/opencode)"

log = logging.getLogger("scraper")

_START_TIME = 0.0
_MAX_RUNTIME = 0.0  # 0 = unlimited


def _time_up() -> bool:
    """Check if --max-runtime has been exceeded."""
    if _MAX_RUNTIME <= 0:
        return False
    elapsed = time.time() - _START_TIME
    return elapsed >= _MAX_RUNTIME


# --------------------------------------------------------------------------- #
# Hard wall-clock timeout via SIGALRM (POSIX, main thread only)
# ---------------------------------------------------------------------------


class _HardTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _HardTimeoutError("wall-clock deadline exceeded")


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #


class Client:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
        )
        self.requests = 0
        self.consecutive_failures = 0
        self.bail_out = False

    def get(self, url: str) -> str | None:
        if self.bail_out:
            return None
        for attempt in range(3):
            try:
                # Hard wall-clock deadline via SIGALRM — catches slow-trickle
                # hangs that requests' per-byte read timeout misses.
                old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(int(HARD_DEADLINE))
                try:
                    r = self.session.get(url, timeout=TIMEOUT)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                self.requests += 1
                if r.status_code == 200:
                    self.consecutive_failures = 0
                    return r.text
                log.warning("GET %s -> %s", url, r.status_code)
                if r.status_code in (404, 410):
                    self.consecutive_failures += 1
                    return None
                if r.status_code in (429, 502, 503, 504):
                    time.sleep(5 * (attempt + 1))
                    continue
                self.consecutive_failures += 1
                return None
            except _HardTimeoutError:
                log.warning("HARD TIMEOUT after %ss: %s", HARD_DEADLINE, url)
                self.consecutive_failures += 1
                break
            except requests.RequestException as e:
                log.warning("error %s: %s", url, e)
                self.consecutive_failures += 1
                time.sleep(2 * (attempt + 1))
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log.error(
                "ABORTING: %d consecutive failures — site appears blocked/down",
                self.consecutive_failures,
            )
            self.bail_out = True
        return None

    def close(self) -> None:
        self.session.close()


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

_COUNT_RE = re.compile(r"([\d.]+)\s*([KMB]?)", re.IGNORECASE)


def parse_count(text: str) -> int:
    """Parse '236.5K', '1.2M', '117M', '4' -> int."""
    if not text:
        return 0
    m = _COUNT_RE.search(text.strip())
    if not m:
        return 0
    num = float(m.group(1))
    suffix = m.group(2).upper()
    mult = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return int(num * mult)


def parse_size_bytes(text: str) -> int | None:
    """Parse a tag-row size like '7.6GB', '43GB', '4.9GB', '243GB' -> bytes."""
    if not text:
        return None
    t = text.strip().upper()
    m = re.match(r"([\d.]+)\s*([KMGTP]?B?)", t)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2)
    mult = {
        "": 1,
        "B": 1,
        "KB": 1_000,
        "K": 1_000,
        "MB": 1_000_000,
        "M": 1_000_000,
        "GB": 1_000_000_000,
        "G": 1_000_000_000,
        "TB": 1_000_000_000_000,
        "T": 1_000_000_000_000,
    }.get(unit, 1)
    return int(num * mult)


def slugify(path: str) -> str:
    """Turn a /library/foo or /user/foo path into a safe filename slug."""
    return path.strip("/").replace("/", "__")


def strip_tags(s: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    s = re.sub(r"<[^>]+>", "", s)
    s = html_mod.unescape(s)
    return s.strip()


def first_attr_value(html_fragment: str, attr: str) -> str | None:
    """First value of `attr` in the fragment (handles ' and " delimiters)."""
    m = re.search(attr + r'\s*=\s*"([^"]*)"', html_fragment)
    if not m:
        m = re.search(attr + r"\s*=\s*'([^']*)'", html_fragment)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class Tag:
    name: str
    size_bytes: int | None
    size_text: str
    context: str  # e.g. "256K"
    input_type: str  # e.g. "Text", "Text, Image"
    digest: str  # short hash
    updated: str  # relative text e.g. "1 month ago"
    format: str  # "gguf" | "mlx"
    usage_level: str = ""  # "low", "medium", "high", "max" — for cloud tags only
    usage_active_slots: int = 0  # 0-4


@dataclass
class Model:
    name: str
    path: str  # site-relative path e.g. /library/llama3.1 or /user/model
    description: str
    capabilities: list[str]
    cloud: bool
    sizes: list[str]
    pulls: int
    tag_count: int
    updated: str  # relative text e.g. "1 week ago"
    updated_title: str  # absolute tooltip e.g. "Jul 2, 2026 2:58 PM UTC"
    official: bool
    owner: str | None  # username for user models, None for official
    source_url: str
    cloud_only: bool = False  # True if all tags are cloud (no downloadable local tags)
    tags: list[Tag] = field(default_factory=list)


@dataclass
class FileEntry:
    type: str  # "model", "license", "params", "template"
    blob_url: str  # full path like /library/gpt-oss:120b/blobs/6be6d66a3f54
    content_preview: str  # truncated text or JSON preview
    size: str  # "65GB", "11kB", etc.
    arch: str  # only for "model" type, else ""
    parameters: str  # only for "model" type, else ""
    quantization: str  # only for "model" type, else ""


@dataclass
class AppEntry:
    name: str  # "Claude Code"
    icon_url: str  # "/public/claude.png"
    command: str  # "ollama launch claude --model kimi-k2.6:cloud"


@dataclass
class ModelPage:
    readme_html: str  # raw HTML inside <div id="display">
    manifest_updated: str  # "9 months ago"
    manifest_digest: str  # "a951a23b46a1"
    manifest_size: str  # "65GB"
    files: list[FileEntry]
    # Cloud metrics (only present for cloud models)
    cloud_usage_level: str = ""  # "low", "medium", "high", "max" — "" if not cloud
    cloud_usage_active_slots: int = 0  # 0-4
    cloud_context: str = ""  # e.g. "256K"
    cloud_context_unit: str = ""  # e.g. "tokens"
    cloud_size: str = ""  # e.g. "1.04T"
    cloud_size_unit: str = ""  # e.g. "parameters"
    applications: list[AppEntry] = field(default_factory=list)


@dataclass
class TagPage:
    tag_name: str  # e.g. "latest", "e2b-mlx"
    full_path: str  # e.g. "/library/gemma4:latest"
    readme_html: str
    manifest_updated: str
    manifest_digest: str
    manifest_size: str
    files: list[FileEntry]
    # Cloud metrics (only present for cloud tag pages)
    cloud_usage_level: str = ""  # "low", "medium", "high", "max" — "" if not cloud
    cloud_usage_active_slots: int = 0  # 0-4
    cloud_context: str = ""  # e.g. "256K"
    cloud_context_unit: str = ""  # e.g. "tokens"
    cloud_size: str = ""  # e.g. "1.04T"
    cloud_size_unit: str = ""  # e.g. "parameters"
    applications: list[AppEntry] = field(default_factory=list)


@dataclass
class MetadataEntry:
    key: str  # e.g. "general.architecture"
    value: str  # e.g. "gptoss"


@dataclass
class TensorEntry:
    name: str  # e.g. "token_embd.weight", "blk.0.attn_k.bias"
    dtype: str  # e.g. "BF16", "F32", "Q4_K_M"
    shape: str  # e.g. "[2880, 201088]", "[512]"
    group: str = ""  # e.g. "blk.0", or "" for ungrouped tensors


@dataclass
class BlobPage:
    blob_url: str  # full path like /library/gpt-oss:120b/blobs/6be6d66a3f54
    tag_full: str  # e.g. "library/gpt-oss:120b"
    blob_type: str  # "model", "license", "params", "template", "json"
    digest: str  # "6be6d66a3f54"
    size: str  # "65GB"
    metadata: list[MetadataEntry]  # for model type blobs
    content: str  # raw text for license/template/params/json blobs
    tensors: list[TensorEntry] = field(default_factory=list)
    tensor_groups: list[str] = field(
        default_factory=list
    )  # e.g. ["blk.0", "blk.1", ...]


# --------------------------------------------------------------------------- #
# Card parsing (two markup variants: /library and /search)
# --------------------------------------------------------------------------- #

# A card <li x-test-model ...> ... </li>. We split on the opening tag of each
# card; the card ends at the next "<li x-test-model" or at a known footer.
_CARD_OPEN_RE = re.compile(r"<li\s+x-test-model[^>]*>", re.IGNORECASE)


def _extract_cards(html: str) -> list[str]:
    """Return the inner HTML of every <li x-test-model> card."""
    starts = [m.start() for m in _CARD_OPEN_RE.finditer(html)]
    cards: list[str] = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(html)
        # The card's own <li> opening tag position; find its matching </li>
        # naively by slicing to the next card (good enough given flat structure).
        inner = html[s:end]
        # Trim trailing to the last </li> before the next card.
        close = inner.rfind("</li>")
        if close != -1 and i + 1 < len(starts):
            inner = inner[: close + len("</li>")]
        cards.append(inner)
    return cards


def _find_spans_with_attr(fragment: str, attr: str) -> list[str]:
    """Return inner text of every <span ... attr ...>...</span> in fragment."""
    out: list[str] = []
    for m in re.finditer(
        r"<span\b[^>]*\b" + re.escape(attr) + r"\b[^>]*>(.*?)</span>",
        fragment,
        re.IGNORECASE | re.DOTALL,
    ):
        out.append(strip_tags(m.group(1)))
    return out


def parse_card(card_html: str, source_url: str) -> Model | None:
    # --- path from first <a href="..."> ---
    am = re.search(r'<a\s+href="([^"]+)"', card_html, re.IGNORECASE)
    if not am:
        return None
    path = am.group(1)
    if not path or path == "/library" or path.startswith("/search"):
        return None

    # --- name: x-test-search-response-title (search) or x-test-model-title
    # attribute (library) ---
    name = ""
    tm = re.search(
        r"x-test-search-response-title[^>]*>(.*?)</span>", card_html, re.DOTALL
    )
    if tm:
        name = strip_tags(tm.group(1))
    if not name:
        tm = re.search(r'x-test-model-title\s+title="([^"]+)"', card_html)
        if tm:
            name = tm.group(1)
    if not name:
        name = path.strip("/").split("/")[-1]

    # --- description: <p class="... break-words ...">...</p> ---
    desc = ""
    dm = re.search(r"<p\b[^>]*\bbreak-words\b[^>]*>(.*?)</p>", card_html, re.DOTALL)
    if dm:
        desc = strip_tags(dm.group(1))

    # --- capabilities (indigo x-test-capability) ---
    capabilities = []
    for cap in _find_spans_with_attr(card_html, "x-test-capability"):
        if cap and cap not in capabilities:
            capabilities.append(cap)

    # --- cloud: a <span ...>cloud</span> with the cyan badge classes ---
    cloud = False
    for m in re.finditer(r"<span\b[^>]*>(.*?)</span>", card_html, re.DOTALL):
        if strip_tags(m.group(1)).lower() == "cloud":
            # only count the cyan badge variant
            if "text-cyan" in m.group(0) or "bg-cyan" in m.group(0):
                cloud = True
            else:
                # fallback: any cloud text counts
                cloud = True
            break

    # --- sizes (blue x-test-size) ---
    sizes = []
    for s in _find_spans_with_attr(card_html, "x-test-size"):
        if s and s not in sizes:
            sizes.append(s)

    # --- pulls / tags / updated ---
    pulls_spans = _find_spans_with_attr(card_html, "x-test-pull-count")
    pulls = parse_count(pulls_spans[0]) if pulls_spans else 0
    tag_spans = _find_spans_with_attr(card_html, "x-test-tag-count")
    tag_count = parse_count(tag_spans[0]) if tag_spans else 0
    upd_spans = _find_spans_with_attr(card_html, "x-test-updated")
    updated = upd_spans[0] if upd_spans else ""
    # The updated timestamp tooltip (when present) is a title= on the <span>
    # wrapping the clock SVG + the x-test-updated span. Match only that span,
    # not the model-title div's title (which holds the model name).
    updated_title = ""
    if upd_spans:
        pm = re.search(
            r'<span[^>]*\btitle="([^"]+)"[^>]*>\s*<svg[^>]*>.*?'
            r'd="M12 6v6h4\.5m4\.5 0a9 9 0.*?"[^>]*>.*?'
            r"x-test-updated",
            card_html,
            re.DOTALL,
        )
        if pm:
            updated_title = pm.group(1)

    # --- official vs user ---
    parts = path.strip("/").split("/")
    official = parts[0] == "library"
    owner = None if official else parts[0]
    real_name = parts[1] if official and len(parts) > 1 else name

    return Model(
        name=real_name,
        path=path,
        description=desc,
        capabilities=capabilities,
        cloud=cloud,
        sizes=sizes,
        pulls=pulls,
        tag_count=tag_count,
        updated=updated,
        updated_title=updated_title,
        official=official,
        owner=owner,
        source_url=BASE + path,
    )


def parse_cards(html_str: str, source_url: str) -> list[Model]:
    cards: list[Model] = []
    for inner in _extract_cards(html_str):
        m = parse_card(inner, source_url)
        if m is not None:
            cards.append(m)
    return cards


# --------------------------------------------------------------------------- #
# Tag page parsing
# --------------------------------------------------------------------------- #


_MLX_BADGE_RE = re.compile(
    r'<span class="ml-2 inline-flex[^"]*\bborder-neutral-600\b[^"]*">\s*MLX\s*</span>',
    re.IGNORECASE,
)


def detect_format(tag_name: str, row_html: str = "") -> str:
    """Detect MLX vs GGUF. Prefers the MLX badge in the row HTML;
    falls back to name-based detection for edge cases."""
    if row_html and _MLX_BADGE_RE.search(row_html):
        return "mlx"
    # Fallback: name-based heuristic
    if re.search(r"(?:^|[-_])mlx(?:$|[-_])", tag_name, re.IGNORECASE):
        return "mlx"
    return "gguf"


_TAG_ROW_RE = re.compile(
    r'<div class="group px-4 py-3">(.*?)(?=<div class="group px-4 py-3">|<!--|$)',
    re.DOTALL,
)


def parse_tags_page(html: str) -> list[Tag]:
    """Parse /<path>/tags -> list of Tag.

    Each row has a mobile anchor and a desktop grid. We parse the desktop
    grid's columns (size/context/input) plus the footer (digest + date).
    """
    tags: list[Tag] = []
    seen: set[str] = set()

    for row_m in _TAG_ROW_RE.finditer(html):
        row = row_m.group(1)

        # Tag full name from the desktop anchor <a href="...:tagname">tagname</a>
        # or the mobile one. We prefer the desktop anchor inside the grid.
        anchor = None
        am = re.search(r'<a\s+href="[^"]*"[^>]*>([^<]+:[^<]+)</a>', row)
        if not am:
            # fallback: any anchor text with a colon
            am = re.search(r"<a\s[^>]*>([^<]*:[^<]*)</a>", row)
        if not am:
            continue
        full_name = strip_tags(am.group(1))
        if ":" not in full_name:
            continue
        tag_name = full_name.split(":", 1)[1].strip()
        if tag_name in seen:
            continue

        # digest: <span class="font-mono ...">xxxx</span> in the footer
        digest = ""
        dm = re.search(r"<span\b[^>]*\bfont-mono\b[^>]*>(.*?)</span>", row, re.DOTALL)
        if dm:
            digest = strip_tags(dm.group(1))

        # updated: footer text after the digest. The footer is:
        #   <div class="flex text-neutral-500 text-xs ...">
        #     <span class="font-mono text-[11px]">digest</span>&nbsp;·&nbsp;date
        #   </div>
        updated = ""
        fm = re.search(
            r"<div\b[^>]*\btext-neutral-500\s+text-xs\b[^>]*>(.*?)</div>",
            row,
            re.DOTALL,
        )
        if fm:
            footer = fm.group(1)
            footer_text = strip_tags(re.sub(r"<[^>]+>", " ", footer))
            # footer text is like "<digest> · 1 month ago"; strip the digest
            # and any leading separator dot, then the residual "·" prefix.
            if digest and digest in footer_text:
                updated = footer_text.replace(digest, "", 1)
            else:
                updated = footer_text
            updated = updated.lstrip(" \u00b7\xa0·").strip()

        # Size / context / input from the desktop grid columns. The grid has
        # col-span-6 (name) + col-span-2 (size) + col-span-2 (context) +
        # col-span-2 (input). We grab every col-span-2 element's text in
        # document order; the first three are size/context/input.
        size_text = ""
        context = ""
        input_type = ""
        cols = re.findall(
            r"<(?:p|div)\b[^>]*\bcol-span-2\b[^>]*>(.*?)</(?:p|div)>",
            row,
            re.DOTALL,
        )
        texts = [strip_tags(c) for c in cols]
        if len(texts) >= 3:
            size_text, context, input_type = texts[0], texts[1], texts[2]

        # Mobile fallback for size/context/input.
        if not size_text:
            mm = re.search(r"<a\b[^>]*\bmd:hidden\b[^>]*>(.*?)</a>", row, re.DOTALL)
            if mm:
                mt = strip_tags(re.sub(r"<[^>]+>", " ", mm.group(1)))
                sm = re.search(r"([\d.]+\s*[KMGTP]?B)\b", mt, re.IGNORECASE)
                if sm:
                    size_text = sm.group(1).strip()
                cm = re.search(r"([\d.]+\s*[KMG]?)\s*context window", mt, re.IGNORECASE)
                if cm:
                    context = cm.group(1).strip()
                im = re.search(r"(Text(?:,\s*\w+)*|Image|Audio)\s*input", mt)
                if im:
                    input_type = im.group(1).strip()

        seen.add(tag_name)

        # Usage tier for cloud tags
        usage_active = len(re.findall(r"x-test-model-tag-usage-slot-active", row))
        usage_level = ""
        if usage_active > 0 or "usage-slot" in row:
            # Try mobile text
            usage_text_m = re.search(
                r"(Low|Medium|High|Max)\s+Usage", row, re.IGNORECASE
            )
            if usage_text_m:
                usage_level = usage_text_m.group(1).lower()
            elif usage_active > 0:
                levels = {1: "low", 2: "medium", 3: "high", 4: "max"}
                usage_level = levels.get(usage_active, "")

        tags.append(
            Tag(
                name=tag_name,
                size_bytes=parse_size_bytes(size_text),
                size_text=size_text,
                context=context,
                input_type=input_type,
                digest=digest,
                updated=updated,
                format=detect_format(tag_name, row),
                usage_level=usage_level,
                usage_active_slots=usage_active,
            )
        )

    return tags


# --------------------------------------------------------------------------- #
# Crawl orchestration
# --------------------------------------------------------------------------- #


def crawl_official(client: Client) -> tuple[dict[str, Model], dict[str, list[str]]]:
    """Crawl /search (trending), /library?sort=popular and /library?sort=newest.

    Returns (models_by_path, sort_orders) where sort_orders maps
    sort name → list of model paths in document order.
    The 'popular' order comes from /search (trending, 20 models) with
    remaining models appended in /library?sort=popular (pulls) order.
    """
    found: dict[str, Model] = {}
    orders: dict[str, list[str]] = {}

    # 1. Fetch /search for the trending list
    log.info("crawling %s/search", BASE)
    search_html = client.get(f"{BASE}/search")
    search_cards = []
    if search_html:
        search_cards = parse_cards(search_html, f"{BASE}/search")
        log.info("  search (trending): %d cards", len(search_cards))
        for m in search_cards:
            if m.path not in found:
                found[m.path] = m
    time.sleep(DELAY)

    # 2. Fetch /library?sort=popular and /library?sort=newest
    lib_orders: dict[str, list[str]] = {}
    for sort in ("popular", "newest"):
        url = f"{BASE}/library?sort={sort}"
        log.info("crawling %s", url)
        html = client.get(url)
        if html is None:
            log.error("failed to fetch %s", url)
            continue
        cards = parse_cards(html, url)
        log.info("  %s: %d cards", sort, len(cards))
        lib_orders[sort] = [m.path for m in cards]
        for m in cards:
            if m.path not in found:
                found[m.path] = m
        time.sleep(DELAY)

    # Build 'popular' order: trending from /search first, then rest by pulls
    trending_paths = [m.path for m in search_cards]
    trending_set = set(trending_paths)
    remaining = [p for p in lib_orders.get("popular", []) if p not in trending_set]
    orders["popular"] = trending_paths + remaining
    orders["newest"] = lib_orders.get("newest", [])

    return found, orders


# Search terms to enumerate user (and extra official) models.
SEARCH_TERMS: list[str] = list("abcdefghijklmnopqrstuvwxyz0123456789") + [
    "llama",
    "qwen",
    "gemma",
    "mistral",
    "deepseek",
    "phi",
    "gpt",
    "code",
    "vision",
    "chat",
    "instruct",
    "mlx",
    "gguf",
    "uncensored",
    "lora",
    "embed",
    "math",
    "reason",
    "agent",
    "tool",
    "think",
    "audio",
    "image",
    "kimi",
    "glm",
    "minimax",
    "nemotron",
    "gpt-oss",
    "starcoder",
    "tulu",
    "aya",
    "command",
    "olmo",
    "smol",
    "tiny",
    "mini",
    "large",
    "medium",
]


def crawl_search(client: Client, models: dict[str, Model]) -> None:
    """Crawl /search?q=<term> for many terms; merge new models into `models`."""
    for term in SEARCH_TERMS:
        url = f"{BASE}/search?q={term}"
        html = client.get(url)
        if html is None:
            log.error("failed to fetch %s", url)
            continue
        cards = parse_cards(html, url)
        new = 0
        for m in cards:
            if m.path not in models:
                models[m.path] = m
                new += 1
        log.info("search q=%-12s -> %2d cards (%d new)", term, len(cards), new)
        time.sleep(DELAY)


def fetch_tags(client: Client, model: Model) -> list[Tag]:
    url = BASE + model.path + "/tags"
    html = client.get(url)
    if html is None:
        log.error("tags fetch failed: %s", url)
        return []
    tags = parse_tags_page(html)
    if not tags:
        log.warning("no tags parsed for %s", url)
    return tags


# --------------------------------------------------------------------------- #
# Model page parsing
# --------------------------------------------------------------------------- #


def _extract_div_by_id(html: str, div_id: str) -> str:
    """Extract the full inner HTML of <div id="...">...</div>, accounting for
    nested divs by tracking depth from the opening tag."""
    import re as _re

    m = _re.search(r'<div\s+id="' + _re.escape(div_id) + r'"', html)
    if not m:
        return ""
    start = m.start()
    # Begin scanning from the end of the opening <div ...> tag.
    open_end = html.find(">", start)
    if open_end == -1:
        return ""
    depth = 1
    i = open_end + 1
    n = len(html)
    while i < n and depth > 0:
        next_open = html.find("<div", i)
        next_close = html.find("</div>", i)
        if next_close == -1:
            return ""
        if next_open != -1 and next_open < next_close:
            depth += 1
            i = next_open + len("<div")
            # Advance past the rest of the opening tag's attributes
            tag_end = html.find(">", i)
            if tag_end == -1:
                return ""
            i = tag_end + 1
        else:
            depth -= 1
            close_start = next_close
            i = next_close + len("</div>")
            if depth == 0:
                return html[open_end + 1 : close_start]
    return ""


def _extract_section_by_id(html: str, section_id: str) -> str:
    """Extract the full inner HTML of <section id="...">...</section>."""
    import re as _re

    m = _re.search(r'<section\s+[^>]*\bid="' + _re.escape(section_id) + r'"', html)
    if not m:
        return ""
    start = m.start()
    open_end = html.find(">", start)
    if open_end == -1:
        return ""
    close = html.find("</section>", open_end)
    if close == -1:
        return ""
    return html[open_end + 1 : close]


def _parse_file_explorer(
    file_explorer: str,
) -> tuple[str, str, str, list[FileEntry]]:
    """Parse the file-explorer section HTML.

    Returns (manifest_updated, manifest_digest, manifest_size, files).
    Shared by parse_model_page() and parse_tag_page().
    """
    # --- manifest header row: <div class="flex items-center justify-between
    #     bg-neutral-50 px-4 py-3 ..."> ... </div> ---
    manifest_updated = ""
    manifest_digest = ""
    manifest_size = ""
    hm = re.search(
        r'<div class="flex items-center justify-between bg-neutral-50[^"]*">'
        r"(.*?)</div>",
        file_explorer,
        re.DOTALL,
    )
    if hm:
        header = hm.group(1)
        # Desktop: <p class="hidden sm:block">Updated 9 months ago</p>
        um = re.search(r'<p class="hidden sm:block">\s*(Updated[^<]*)</p>', header)
        if um:
            manifest_updated = strip_tags(um.group(1)).replace("Updated", "").strip()
        else:
            # Mobile: text after the SVG
            mm = re.search(
                r'<p class="flex items-center sm:hidden">.*?<svg[^>]*>.*?</svg>(.*?)</p>',
                header,
                re.DOTALL,
            )
            if mm:
                manifest_updated = strip_tags(mm.group(1)).strip()
        # Digest + size: <p>a951a23b46a1 · 65GB ·</p>
        dm = re.search(r"<p\b[^>]*>([^<]*·[^<]*)</p>", header)
        if dm:
            blob_text = strip_tags(dm.group(1))
            parts = [p.strip() for p in blob_text.split("·") if p.strip()]
            if parts:
                manifest_digest = parts[0]
            if len(parts) > 1:
                manifest_size = parts[1]

    # --- file rows: <div class="group block grid-cols-12 ..."> ... </div> ---
    files: list[FileEntry] = []
    for rm in re.finditer(
        r'<div class="group block grid-cols-12[^"]*">(.*?)(?=<div class="group block grid-cols-12|<!--|$)',
        file_explorer,
        re.DOTALL,
    ):
        row = rm.group(1)
        # type + blob_url from the <a> inside sm:col-span-2
        tm = re.search(
            r'<div class="[^"]*sm:col-span-2[^"]*">\s*<a\s+href="([^"]+)"[^>]*>(.*?)</a>',
            row,
            re.DOTALL,
        )
        if not tm:
            continue
        blob_url = tm.group(1)
        ftype = strip_tags(tm.group(2))

        # content_preview: text content of the sm:col-span-8 div
        cm = re.search(
            r'<div class="[^"]*sm:col-span-8[^"]*">(.*?)</div>',
            row,
            re.DOTALL,
        )
        content_preview = strip_tags(re.sub(r"<[^>]+>", " ", cm.group(1))) if cm else ""

        # size: sm:col-start-12 div
        sz = ""
        sm = re.search(
            r'<div class="[^"]*sm:col-start-12[^"]*"[^>]*>(.*?)</div>',
            row,
            re.DOTALL,
        )
        if sm:
            sz = strip_tags(sm.group(1))

        # For model-type rows, extract arch/parameters/quantization from the
        # nested <span class="text-neutral-800 ..."> values following their
        # labels.
        arch = ""
        parameters = ""
        quantization = ""
        if ftype == "model":
            for m in re.finditer(
                r'<span class="hidden sm:block">(.*?)</span>'
                r'\s*<span class="[^"]*text-neutral-800[^"]*">(.*?)</span>',
                row,
                re.DOTALL,
            ):
                label = strip_tags(m.group(1)).lower()
                value = strip_tags(m.group(2))
                if label == "arch":
                    arch = value
                elif label == "parameters":
                    parameters = value
                elif label == "quantization":
                    quantization = value

        files.append(
            FileEntry(
                type=ftype,
                blob_url=blob_url,
                content_preview=content_preview,
                size=sz,
                arch=arch,
                parameters=parameters,
                quantization=quantization,
            )
        )

    return manifest_updated, manifest_digest, manifest_size, files


def _parse_applications(html: str) -> list[AppEntry]:
    """Parse the Applications section (id="external-tools-section") HTML.

    Each app row is a <div class="group flex items-center justify-between
    px-4 py-3"> with an icon, name span, code command, and a hidden input
    holding the same command.
    """
    applications: list[AppEntry] = []
    ext_section = _extract_section_by_id(html, "external-tools-section")
    if not ext_section:
        return applications
    for row_m in re.finditer(
        r'<div class="group flex items-center justify-between px-4 py-3">(.*?)(?=<div class="group flex items-center|</div>\s*</div>\s*</div>|$)',
        ext_section,
        re.DOTALL,
    ):
        row = row_m.group(1)
        nm = re.search(r'<span class="text-sm font-medium[^"]*">([^<]+)</span>', row)
        im = re.search(r'<img\s+src="([^"]+)"', row)
        cm = re.search(r"<code[^>]*>([^<]+)</code>", row)
        if not cm:
            cm = re.search(r'<input class="command hidden" value="([^"]+)"', row)
        if nm:
            applications.append(
                AppEntry(
                    name=strip_tags(nm.group(1)),
                    icon_url=im.group(1) if im else "",
                    command=strip_tags(cm.group(1)) if cm else "",
                )
            )
    return applications


def _parse_cloud_metrics(
    html: str,
) -> tuple[str, int, str, str, str, str]:
    """Extract cloud metrics (usage/context/size) from a model or tag page.

    Returns (cloud_usage_level, cloud_usage_active_slots, cloud_context,
    cloud_context_unit, cloud_size, cloud_size_unit).

    These metrics are present on tag pages (/library/model:tag) for cloud
    tags, identified by `x-test-model-metric="usage|context|size"` markers.
    They are absent on base model pages, so this returns empty defaults there.
    """
    cloud_usage_level = ""
    cloud_usage_active_slots = 0
    cloud_context = ""
    cloud_context_unit = ""
    cloud_size = ""
    cloud_size_unit = ""

    # Usage metric
    usage_m = re.search(
        r'x-test-model-metric="usage".*?(?=x-test-model-metric="context"|$)',
        html,
        re.DOTALL,
    )
    if usage_m:
        usage_section = usage_m.group(0)
        active_count = len(re.findall(r"x-test-model-cost-slot-active", usage_section))
        cloud_usage_active_slots = active_count
        # Level text: <span class="...">high</span> or <span class="...">low</span>
        level_m = re.search(r"break-words[^>]*>\s*(\w+)\s*</span>", usage_section)
        if level_m:
            cloud_usage_level = level_m.group(1).strip().lower()

    # Context metric
    ctx_m = re.search(
        r'x-test-model-metric="context".*?(?=x-test-model-metric="size"|$)',
        html,
        re.DOTALL,
    )
    if ctx_m:
        ctx_section = ctx_m.group(0)
        val_m = re.search(
            r"text-xl font-medium leading-none[^>]*>\s*([^<]+)</span>", ctx_section
        )
        if val_m:
            cloud_context = val_m.group(1).strip()
        unit_m = re.search(r"break-words[^>]*>\s*([^<]+)</span>", ctx_section)
        if unit_m:
            cloud_context_unit = unit_m.group(1).strip()

    # Size metric
    size_m = re.search(
        r'x-test-model-metric="size".*?(?=</div>\s*</div>|$)', html, re.DOTALL
    )
    if size_m:
        size_section = size_m.group(0)
        val_m = re.search(
            r"text-xl font-medium leading-none[^>]*>\s*([^<]+)</span>", size_section
        )
        if val_m:
            cloud_size = val_m.group(1).strip()
        unit_m = re.search(r"break-words[^>]*>\s*([^<]+)</span>", size_section)
        if unit_m:
            cloud_size_unit = unit_m.group(1).strip()

    return (
        cloud_usage_level,
        cloud_usage_active_slots,
        cloud_context,
        cloud_context_unit,
        cloud_size,
        cloud_size_unit,
    )


def parse_model_page(html: str) -> ModelPage | None:
    """Parse a model page (/<path>) -> ModelPage.

    Extracts the readme HTML (inside <div id="display">) and the file
    explorer section (manifest header + per-file rows).
    """
    readme_html = _extract_div_by_id(html, "display")
    file_explorer = _extract_section_by_id(html, "file-explorer")
    if not readme_html and not file_explorer:
        return None

    manifest_updated, manifest_digest, manifest_size, files = _parse_file_explorer(
        file_explorer
    )

    # Cloud metrics (present on tag pages; base pages currently lack them but
    # we parse anyway in case ollama.com adds them in the future).
    (
        cloud_usage_level,
        cloud_usage_active_slots,
        cloud_context,
        cloud_context_unit,
        cloud_size,
        cloud_size_unit,
    ) = _parse_cloud_metrics(html)

    return ModelPage(
        readme_html=readme_html,
        manifest_updated=manifest_updated,
        manifest_digest=manifest_digest,
        manifest_size=manifest_size,
        files=files,
        cloud_usage_level=cloud_usage_level,
        cloud_usage_active_slots=cloud_usage_active_slots,
        cloud_context=cloud_context,
        cloud_context_unit=cloud_context_unit,
        cloud_size=cloud_size,
        cloud_size_unit=cloud_size_unit,
        applications=_parse_applications(html),
    )


def parse_tag_page(html: str, full_path: str) -> TagPage | None:
    """Parse a per-tag page (/<path>:<tag>) -> TagPage.

    Same structure as parse_model_page() but the path includes the tag.
    The file-explorer is present on tag pages (unlike base model pages where
    it is often missing).
    """
    readme_html = _extract_div_by_id(html, "display")
    file_explorer = _extract_section_by_id(html, "file-explorer")
    if not readme_html and not file_explorer:
        return None

    # tag_name is the part after the last ":" in the full_path
    tag_name = full_path.rsplit(":", 1)[-1] if ":" in full_path else ""

    manifest_updated, manifest_digest, manifest_size, files = _parse_file_explorer(
        file_explorer
    )

    # Cloud metrics (present on tag pages for cloud tags)
    (
        cloud_usage_level,
        cloud_usage_active_slots,
        cloud_context,
        cloud_context_unit,
        cloud_size,
        cloud_size_unit,
    ) = _parse_cloud_metrics(html)

    return TagPage(
        tag_name=tag_name,
        full_path=full_path,
        readme_html=readme_html,
        manifest_updated=manifest_updated,
        manifest_digest=manifest_digest,
        manifest_size=manifest_size,
        files=files,
        cloud_usage_level=cloud_usage_level,
        cloud_usage_active_slots=cloud_usage_active_slots,
        cloud_context=cloud_context,
        cloud_context_unit=cloud_context_unit,
        cloud_size=cloud_size,
        cloud_size_unit=cloud_size_unit,
        applications=_parse_applications(html),
    )


def fetch_model_page(client: Client, model: Model) -> ModelPage | None:
    url = BASE + model.path
    html = client.get(url)
    if html is None:
        log.error("model page fetch failed: %s", url)
        return None
    return parse_model_page(html)


def save_model_page(model: Model, page: ModelPage) -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(model.path)
    out = {
        "path": model.path,
        "name": model.name,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "readme_html": page.readme_html,
        "manifest_updated": page.manifest_updated,
        "manifest_digest": page.manifest_digest,
        "manifest_size": page.manifest_size,
        "files": [asdict(f) for f in page.files],
        "cloud_usage_level": page.cloud_usage_level,
        "cloud_usage_active_slots": page.cloud_usage_active_slots,
        "cloud_context": page.cloud_context,
        "cloud_context_unit": page.cloud_context_unit,
        "cloud_size": page.cloud_size,
        "cloud_size_unit": page.cloud_size_unit,
        "applications": [asdict(a) for a in page.applications],
    }
    fp = PAGES_DIR / f"{slug}.json"
    _atomic_write(fp, json.dumps(out, indent=2))
    log.debug("saved %s", fp)


def fetch_tag_page(client: Client, model: Model, tag_name: str) -> TagPage | None:
    full_path = f"{model.path}:{tag_name}"
    url = BASE + full_path
    html = client.get(url)
    if html is None:
        log.error("tag page fetch failed: %s", url)
        return None
    return parse_tag_page(html, full_path)


def save_tag_page(model: Model, tag_name: str, page: TagPage) -> None:
    TAG_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(model.path)
    out = {
        "path": model.path,
        "tag_name": tag_name,
        "full_path": page.full_path,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "readme_html": page.readme_html,
        "manifest_updated": page.manifest_updated,
        "manifest_digest": page.manifest_digest,
        "manifest_size": page.manifest_size,
        "files": [asdict(f) for f in page.files],
        "cloud_usage_level": page.cloud_usage_level,
        "cloud_usage_active_slots": page.cloud_usage_active_slots,
        "cloud_context": page.cloud_context,
        "cloud_context_unit": page.cloud_context_unit,
        "cloud_size": page.cloud_size,
        "cloud_size_unit": page.cloud_size_unit,
        "applications": [asdict(a) for a in page.applications],
    }
    fp = TAG_PAGES_DIR / f"{slug}__{tag_name}.json"
    _atomic_write(fp, json.dumps(out, indent=2))
    log.debug("saved %s", fp)


# --------------------------------------------------------------------------- #
# Blob page parsing
# --------------------------------------------------------------------------- #


def _extract_div_by_class_substring(html: str, class_substr: str) -> str:
    """Extract the inner HTML of the first <div ...> whose class attribute
    contains `class_substr`, accounting for nested divs (depth-aware)."""
    import re as _re

    m = _re.search(
        r'<div\b[^>]*\bclass\s*=\s*"[^"]*\b'
        + _re.escape(class_substr)
        + r'\b[^"]*"[^>]*>',
        html,
        _re.DOTALL,
    )
    if not m:
        return ""
    depth = 1
    i = m.end()
    n = len(html)
    while i < n and depth > 0:
        next_open = html.find("<div", i)
        next_close = html.find("</div>", i)
        if next_close == -1:
            return ""
        if next_open != -1 and next_open < next_close:
            depth += 1
            i = next_open + len("<div")
            tag_end = html.find(">", i)
            if tag_end == -1:
                return ""
            i = tag_end + 1
        else:
            depth -= 1
            close_start = next_close
            i = next_close + len("</div>")
            if depth == 0:
                return html[m.end() : close_start]
    return ""


def parse_blob_page(html: str, blob_url: str) -> BlobPage | None:
    """Parse a blob page (/<path>:<tag>/blobs/<digest>) -> BlobPage.

    Blob pages have a `<div id="file-explorer">` block with a header (tag full
    name, blob type, digest + size) and content (metadata table for model-type
    blobs, raw text for license/template/params/json blobs).
    """
    file_explorer = _extract_div_by_id(html, "file-explorer")
    if not file_explorer:
        return None

    # --- header: tag full name (from the <a>), blob type (div after the "/" span),
    #     digest + size (last div in the header) ---
    tag_full = ""
    blob_type = ""
    digest = ""
    size = ""
    header = _extract_div_by_class_substring(
        file_explorer, "flex items-center justify-between bg-neutral-50"
    )
    if header:
        # tag full name from <a href="...">name</a> (prefer the desktop span
        # hidden sm:block, which holds the full "name:tag"; fallback to whole
        # <a> text).
        am = re.search(
            r"<a\b[^>]*>\s*<span\b[^>]*\bhidden sm:block\b[^>]*>(.*?)</span>",
            header,
            re.DOTALL,
        )
        if am:
            tag_full = strip_tags(am.group(1))
        else:
            am = re.search(r"<a\b[^>]*>(.*?)</a>", header, re.DOTALL)
            if am:
                tag_full = strip_tags(am.group(1))
        # blob type: the <div> following the "/" separator span.
        # Header structure: <a>...</a><span>/</span><div>type</div>
        sm = re.search(
            r"<span\b[^>]*>[^<]*</span>\s*<div\b[^>]*>(.*?)</div>",
            header,
            re.DOTALL,
        )
        if sm:
            blob_type = strip_tags(sm.group(1))
        # digest + size: last leaf <div> in the header (right-aligned).
        divs = re.findall(r"<div\b[^>]*>(.*?)</div>", header, re.DOTALL)
        if divs:
            blob_text = strip_tags(divs[-1])
            parts = [p.strip() for p in re.split(r"[·\u00b7]", blob_text) if p.strip()]
            if parts:
                digest = parts[0]
            if len(parts) > 1:
                size = parts[1]

    # --- content: metadata table (model-type) or raw text (others) ---
    # ollama.com blob pages render two sections inside a single <ul role="list">:
    #   1. Metadata  — sticky header div text "Metadata"; <li> rows use
    #      `px-2 sm:px-4 pt-2 sm:pb-2` with 2 columns (key, value).
    #   2. Tensor    — sticky header div text "Tensor"; <li> rows use
    #      `px-4 py-2` with 3 columns (name, type, shape). Block group
    #      dividers (`blk.0`, `blk.1`, ...) are sticky divs whose text is the
    #      group name.
    # The two section headers share the same border/bg classes, so we split by
    # locating the "Tensor" sticky header by its text content.
    metadata: list[MetadataEntry] = []
    content = ""
    tensors: list[TensorEntry] = []
    tensor_groups: list[str] = []

    has_metadata = re.search(r"<li\b[^>]*\bgrid grid-cols-8\b", file_explorer)
    if has_metadata:
        # Locate the start of the Tensor section: the first sticky div whose
        # inner text is "Tensor". Everything before it is metadata; everything
        # from it onward is tensor data (header + column header + entries +
        # group dividers).
        tensor_start = -1
        for sm in re.finditer(
            r'<div class="sticky top-0 border-y[^"]*">\s*'
            r'<div class="py-2 px-4 text-xs[^"]*">\s*(.*?)\s*</div>',
            file_explorer,
            re.DOTALL,
        ):
            if strip_tags(sm.group(1)) == "Tensor":
                tensor_start = sm.start()
                break

        if tensor_start == -1:
            metadata_html = file_explorer
            tensor_html = ""
        else:
            metadata_html = file_explorer[:tensor_start]
            tensor_html = file_explorer[tensor_start:]

        # --- metadata entries (2-column rows) ---
        for lm in re.finditer(
            r"<li\b[^>]*\bgrid grid-cols-8\b[^>]*>(.*?)(?=<li\b[^>]*\bgrid grid-cols-8|</ul>|<!--|$)",
            metadata_html,
            re.DOTALL,
        ):
            row = lm.group(1)
            # key: the text-neutral-600 div (or the sm:text-black div)
            km = re.search(
                r"<div\b[^>]*\btext-neutral-600\b[^>]*>(.*?)</div>",
                row,
                re.DOTALL,
            )
            key = strip_tags(km.group(1)) if km else ""
            # value: the font-mono div holding the value. There may be two
            # font-mono divs (mobile + desktop); prefer the one with the
            # `hidden sm:block` (desktop) marker.
            vm = re.search(
                r"<div\b[^>]*\bhidden sm:block\b[^>]*\bcol-span-4\b[^>]*\bfont-mono\b[^>]*>(.*?)</div>",
                row,
                re.DOTALL,
            )
            if not vm:
                vm = re.search(
                    r"<div\b[^>]*\bcol-span-4\b[^>]*\bfont-mono\b[^>]*>(.*?)</div>",
                    row,
                    re.DOTALL,
                )
            if not vm:
                vm = re.search(
                    r"<div\b[^>]*\bfont-mono\b[^>]*>(.*?)</div>",
                    row,
                    re.DOTALL,
                )
            value = strip_tags(vm.group(1)) if vm else ""
            if key or value:
                metadata.append(MetadataEntry(key=key, value=value))

        # --- tensor entries (3-column rows) + block group dividers ---
        # The column-header <li> uses `grid-cols-8 ... hidden sm:grid` (no
        # standalone `grid` class), so it is naturally skipped by the
        # `grid grid-cols-8` match below. Walk tensor_html in document order,
        # interleaving tensor rows and sticky group dividers so each tensor
        # records the group that was active when it appeared.
        current_group = ""
        token_re = re.compile(
            r"<li\b[^>]*\bgrid grid-cols-8\b[^>]*>"
            r"|"
            r'<div class="sticky top-0 border-y[^"]*">\s*'
            r'<div class="py-2 px-4 text-xs[^"]*">\s*(.*?)\s*</div>',
            re.DOTALL,
        )
        for em in token_re.finditer(tensor_html):
            if em.group(1) is not None:
                # sticky group divider (also matches the leading "Tensor"
                # header, which we skip because gname == "Tensor")
                gname = strip_tags(em.group(1))
                if gname and gname != "Tensor":
                    current_group = gname
                    if gname not in tensor_groups:
                        tensor_groups.append(gname)
                continue
            # tensor row <li> — capture up to the next row/divider/</ul>
            start = em.end()
            nxt = token_re.search(tensor_html, start)
            end = nxt.start() if nxt else len(tensor_html)
            row = tensor_html[start:end]
            # name: the text-neutral-600 div
            nm = re.search(
                r"<div\b[^>]*\btext-neutral-600\b[^>]*>(.*?)</div>",
                row,
                re.DOTALL,
            )
            name = strip_tags(nm.group(1)) if nm else ""
            # dtype: prefer the desktop `col-span-1` font-mono div, then
            # fall back to the mobile `hidden sm:block` font-mono div.
            dm = re.search(
                r"<div\b[^>]*\bcol-span-1\b[^>]*\bfont-mono\b[^>]*>(.*?)</div>",
                row,
                re.DOTALL,
            )
            if not dm:
                dm = re.search(
                    r"<div\b[^>]*\bhidden sm:block\b[^>]*\bfont-mono\b[^>]*>(.*?)</div>",
                    row,
                    re.DOTALL,
                )
            dtype = strip_tags(dm.group(1)) if dm else ""
            # shape: the col-span-3 font-mono div
            shp = re.search(
                r"<div\b[^>]*\bcol-span-3\b[^>]*\bfont-mono\b[^>]*>(.*?)</div>",
                row,
                re.DOTALL,
            )
            shape = strip_tags(shp.group(1)) if shp else ""
            if name or dtype or shape:
                tensors.append(
                    TensorEntry(
                        name=name, dtype=dtype, shape=shape, group=current_group
                    )
                )
    else:
        # Raw text content: find the whitespace-pre-wrap div (depth-aware, since
        # its class spans multiple lines and it contains child <div> line
        # elements) and extract each child line's text.
        block = _extract_div_by_class_substring(file_explorer, "whitespace-pre-wrap")
        if block:
            lines = re.findall(r"<div\b[^>]*>(.*?)</div>", block, re.DOTALL)
            if lines:
                content = "\n".join(strip_tags(ln) for ln in lines)
            else:
                content = strip_tags(block)

    return BlobPage(
        blob_url=blob_url,
        tag_full=tag_full,
        blob_type=blob_type,
        digest=digest,
        size=size,
        metadata=metadata,
        content=content,
        tensors=tensors,
        tensor_groups=tensor_groups,
    )


def fetch_blob_page(client: Client, blob_url: str) -> BlobPage | None:
    url = BASE + blob_url
    html = client.get(url)
    if html is None:
        log.error("blob page fetch failed: %s", url)
        return None
    return parse_blob_page(html, blob_url)


def save_blob_page(blob_url: str, page: BlobPage) -> None:
    BLOBS_DIR.mkdir(parents=True, exist_ok=True)
    safe = blob_url.strip("/").replace("/", "__").replace(":", "_")
    out = {
        "blob_url": page.blob_url,
        "tag_full": page.tag_full,
        "blob_type": page.blob_type,
        "digest": page.digest,
        "size": page.size,
        "metadata": [asdict(m) for m in page.metadata],
        "content": page.content,
        "tensors": [asdict(t) for t in page.tensors],
        "tensor_groups": page.tensor_groups,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    fp = BLOBS_DIR / f"{safe}.json"
    _atomic_write(fp, json.dumps(out, indent=2))
    log.debug("saved %s", fp)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _atomic_write(filepath: Path, content: str) -> None:
    """Write content to filepath atomically using a temp file + rename."""
    tmp = filepath.with_suffix(filepath.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, filepath)


def save_models(models: Iterable[Model]) -> None:
    data: dict = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "count": 0,
        "models": [],
    }
    ms = sorted(models, key=lambda m: (not m.official, m.name.lower()))
    data["models"] = [asdict(m) for m in ms]
    data["count"] = len(data["models"])
    fp = DATA / "models.json"
    _atomic_write(fp, json.dumps(data, indent=2))
    log.info("wrote models.json (%d models)", data["count"])
    log.debug("saved %s", fp)


def save_sort_data(sort_orders: dict, models: dict) -> None:
    """Save sort_orders.json and sort_ranks.json for build.py."""
    _atomic_write(DATA / "sort_orders.json", json.dumps(sort_orders, indent=2))
    ranks: dict[str, dict] = {}
    all_names = {m.name for m in models.values()}
    for sort_name, paths in sort_orders.items():
        for rank, path in enumerate(paths):
            slug = path.strip("/").split("/")[-1]
            model = models.get(path)
            name = model.name if model else slug
            if name not in ranks:
                ranks[name] = {}
            ranks[name][f"{sort_name}_rank"] = rank
    for name in all_names:
        if name not in ranks:
            ranks[name] = {}
        for sort_name in sort_orders:
            key = f"{sort_name}_rank"
            ranks[name].setdefault(key, 9999)
    _atomic_write(DATA / "sort_ranks.json", json.dumps(ranks, indent=2))
    log.info("wrote sort_orders.json + sort_ranks.json")


def save_tags(model: Model, tags: list[Tag]) -> None:
    TAGS_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(model.path)
    out = {
        "path": model.path,
        "name": model.name,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "tags": [asdict(t) for t in tags],
    }
    fp = TAGS_DIR / f"{slug}.json"
    _atomic_write(fp, json.dumps(out, indent=2))
    log.debug("saved %s", fp)


# --------------------------------------------------------------------------- #
# Git checkpoint — push scraped data to scraped-data branch
# --------------------------------------------------------------------------- #

_GIT_CHECKPOINT_COUNT = 0


def git_checkpoint(label: str = "") -> None:
    """Commit + push scraped data to the 'scraped-data' branch.

    Only acts if the GIT_CHECKPOINT env var is set (CI mode).  Uses a git
    worktree at GIT_CHECKPOINT_WORKTREE to avoid branch switching.
    """
    global _GIT_CHECKPOINT_COUNT
    if not os.environ.get("GIT_CHECKPOINT"):
        return
    worktree = os.environ.get("GIT_CHECKPOINT_WORKTREE", "")
    if not worktree or not os.path.isdir(worktree):
        return
    _GIT_CHECKPOINT_COUNT += 1
    tag = f"[{_GIT_CHECKPOINT_COUNT}] " if label else ""
    msg = f"checkpoint {tag}{label}".strip()
    try:
        # Copy scraper data to the worktree
        os.makedirs(f"{worktree}/scraper", exist_ok=True)
        for item in os.listdir("scraper"):
            src = f"scraper/{item}"
            dst = f"{worktree}/scraper/{item}"
            if os.path.isdir(src):
                subprocess.run(["cp", "-r", src, dst], check=True, cwd=HERE.parent)
            else:
                subprocess.run(["cp", src, dst], check=True, cwd=HERE.parent)
        subprocess.run(["git", "add", "-A", "scraper/"], check=True, cwd=worktree)
        r = subprocess.run(
            ["git", "commit", "-m", f"{msg} [skip ci]"],
            capture_output=True,
            text=True,
            cwd=worktree,
        )
        if r.returncode != 0:
            return  # nothing to commit
        subprocess.run(
            ["git", "push", "origin", "scraped-data"],
            check=True,
            capture_output=True,
            text=True,
            cwd=worktree,
        )
        log.info("  git checkpoint pushed: %s", msg)
    except subprocess.CalledProcessError as e:
        log.warning("  git checkpoint failed: %s", e)
    except Exception as e:
        log.warning("  git checkpoint error: %s", e)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def check_only() -> int:
    """Fetch /search + /library?sort=newest (2 requests), hash card data, compare to cache.

    Exit 0 = no change, 1 = changed (or first run).
    Writes the hash to scraper/.catalog-hash.
    """
    import hashlib

    client = Client()
    try:
        # Fetch /search (trending)
        search_html = client.get(f"{BASE}/search")
        cards = []
        if search_html:
            cards = parse_cards(search_html, f"{BASE}/search")
            log.info("fetched %d cards from /search", len(cards))

        # Fetch /library?sort=newest (all official models)
        lib_html = client.get(f"{BASE}/library?sort=newest")
        if lib_html:
            lib_cards = parse_cards(lib_html, f"{BASE}/library?sort=newest")
            log.info("fetched %d cards from /library?sort=newest", len(lib_cards))
            # Merge, deduplicating by path
            seen = {c.path for c in cards}
            for c in lib_cards:
                if c.path not in seen:
                    cards.append(c)
                    seen.add(c.path)

        log.info("total unique cards: %d", len(cards))
    finally:
        client.close()

    # Hash: name + pulls + tag_count + updated + path
    sig = "|".join(
        f"{m.name}:{m.pulls}:{m.tag_count}:{m.updated}:{m.path}" for m in cards
    )
    current = hashlib.sha256(sig.encode()).hexdigest()[:16]

    cache_file = DATA / ".catalog-hash"
    prev = ""
    if cache_file.exists():
        prev = cache_file.read_text().strip()

    log.info("previous hash: %s", prev or "(none)")
    log.info("current hash:  %s", current)

    if current == prev:
        log.info("no changes — exiting 0")
        return 0
    else:
        _atomic_write(cache_file, current)
        log.info("changes detected — exiting 1")
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape ollama.com model catalog.")
    ap.add_argument(
        "--refresh-model",
        help="re-scrape tags for a single model path (e.g. /library/gemma4)",
    )
    ap.add_argument(
        "--skip-tags",
        action="store_true",
        help="only scrape the catalog, not per-model tags",
    )
    ap.add_argument(
        "--skip-pages",
        action="store_true",
        help="skip fetching per-model pages (official only)",
    )
    ap.add_argument(
        "--skip-tag-pages",
        action="store_true",
        help="skip fetching per-tag detail pages (official only: latest + MLX tags)",
    )
    ap.add_argument(
        "--skip-blobs",
        action="store_true",
        help="skip fetching per-blob detail pages (official only)",
    )
    ap.add_argument(
        "--skip-search",
        action="store_true",
        help="skip /search user-model sweep (official only)",
    )
    ap.add_argument(
        "--smart",
        action="store_true",
        help="only fetch tags for new/changed models; reuse cached tags for unchanged",
    )
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="validate existing data and exit",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="fetch /search + /library?sort=newest (2 requests), hash card data, "
        "compare to scraper/.catalog-hash, exit 0=unchanged 1=changed",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--max-runtime",
        type=int,
        default=0,
        help="stop scraping gracefully after N seconds (0 = unlimited). "
        "Useful for CI: scrape in time-bounded bursts, deploy between bursts.",
    )
    args = ap.parse_args(argv)

    global _START_TIME, _MAX_RUNTIME
    _START_TIME = time.time()
    _MAX_RUNTIME = args.max_runtime if args.max_runtime > 0 else 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    TAGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.self_check:
        return self_check()

    if args.check_only:
        return check_only()

    client = Client()
    try:
        if args.refresh_model:
            path = args.refresh_model
            if not path.startswith("/"):
                path = "/" + path
            m = Model(
                name=path.rsplit("/", 1)[-1],
                path=path,
                description="",
                capabilities=[],
                cloud=False,
                sizes=[],
                pulls=0,
                tag_count=0,
                updated="",
                updated_title="",
                official=path.startswith("/library/"),
                owner=(
                    None
                    if path.startswith("/library/")
                    else path.strip("/").split("/")[0]
                ),
                source_url=BASE + path,
            )
            log.info("refreshing tags for %s", path)
            m.tags = fetch_tags(client, m)
            save_tags(m, m.tags)
            log.info("  %d tags", len(m.tags))
            if m.official:
                page_slug = slugify(m.path)
                page_file = PAGES_DIR / f"{page_slug}.json"
                if page_file.exists():
                    log.info("  page: cached")
                else:
                    page = fetch_model_page(client, m)
                    if page:
                        save_model_page(m, page)
                        log.info(
                            "  page: %d files, readme %d chars",
                            len(page.files),
                            len(page.readme_html),
                        )
                    else:
                        log.warning("  page: fetch failed")
                # Fetch tag pages for ALL tags (skip cached)
                tags_to_fetch = list(m.tags)
                cached_tp = 0
                fetched_tp = 0
                for t in tags_to_fetch:
                    if client.bail_out:
                        break
                    slug = slugify(m.path)
                    tp_file = TAG_PAGES_DIR / f"{slug}__{t.name}.json"
                    if tp_file.exists():
                        try:
                            existing = json.loads(tp_file.read_text())
                            cached_digest = existing.get("manifest_digest", "")
                            if cached_digest == t.digest:
                                cached_tp += 1
                                continue
                            else:
                                log.info(
                                    "  tag page digest changed: %s:%s (%s -> %s)",
                                    m.path,
                                    t.name,
                                    cached_digest,
                                    t.digest,
                                )
                        except Exception:
                            pass
                    log.info("  tag page: %s:%s", m.path, t.name)
                    tp = fetch_tag_page(client, m, t.name)
                    if tp:
                        save_tag_page(m, t.name, tp)
                        fetched_tp += 1
                        log.info(
                            "    %d files, readme %d chars",
                            len(tp.files),
                            len(tp.readme_html),
                        )
                    time.sleep(DELAY)
                log.info("  tag pages: %d cached, %d fetched", cached_tp, fetched_tp)
                # Fetch blob pages for every file in every tag page
                if not args.skip_blobs:
                    log.info("  fetching blob pages for %s", m.path)
                    BLOBS_DIR.mkdir(parents=True, exist_ok=True)
                    blob_count = 0
                    for t in m.tags:
                        if client.bail_out or _time_up():
                            break
                        slug = slugify(m.path)
                        tp_file = TAG_PAGES_DIR / f"{slug}__{t.name}.json"
                        if not tp_file.exists():
                            continue
                        try:
                            tp_data = json.loads(tp_file.read_text())
                        except Exception:
                            continue
                        for f in tp_data.get("files", []):
                            burl = f.get("blob_url", "")
                            if not burl:
                                continue
                            safe = burl.strip("/").replace("/", "__").replace(":", "_")
                            bf = BLOBS_DIR / f"{safe}.json"
                            if bf.exists():
                                continue
                            bp = fetch_blob_page(client, burl)
                            if bp:
                                save_blob_page(burl, bp)
                                blob_count += 1
                            time.sleep(DELAY)
                    log.info("  fetched %d blob pages", blob_count)
            # Update the model in models.json with fresh data
            if (DATA / "models.json").exists():
                try:
                    models_data = json.loads((DATA / "models.json").read_text())
                    for i, mm in enumerate(models_data.get("models", [])):
                        if mm["path"] == m.path:
                            mm["tag_count"] = len(m.tags)
                            mm["tags"] = [asdict(t) for t in m.tags]
                            if m.cloud and m.tags:
                                local_tags = [
                                    t
                                    for t in m.tags
                                    if t.name != "cloud"
                                    and not t.name.endswith("-cloud")
                                    and t.size_bytes
                                ]
                                mm["cloud_only"] = len(local_tags) == 0
                            break
                    json.dump(models_data, open(DATA / "models.json", "w"), indent=2)
                    log.info("  updated models.json for %s", m.path)
                except Exception as e:
                    log.warning("  failed to update models.json: %s", e)
            return 0

        # ---- full crawl ----
        log.info("=== crawling official catalog ===")
        models, sort_orders = crawl_official(client)
        log.info("official models: %d", len(models))

        # Load previous model data for smart comparison
        prev_models = {}
        if args.smart and (DATA / "models.json").exists():
            try:
                prev_data = json.loads((DATA / "models.json").read_text())
                for pm in prev_data.get("models", []):
                    prev_models[pm["path"]] = pm
            except Exception:
                pass

        if not args.skip_search:
            log.info("=== crawling /search for user models ===")
            before = len(models)
            crawl_search(client, models)
            log.info(
                "user-model sweep added %d models (total %d)",
                len(models) - before,
                len(models),
            )

        # Save sort orderings + derived rank data EARLY so they're available
        # even if the run is cancelled before reaching the end.
        if sort_orders:
            save_sort_data(sort_orders, models)
            git_checkpoint("sort data + models")

        if not args.skip_tags:
            if args.smart and prev_models:
                # Determine which models changed
                to_fetch = []
                cached = 0
                for m in models.values():
                    slug = slugify(m.path)
                    tf = TAGS_DIR / f"{slug}.json"
                    pm = prev_models.get(m.path)
                    if (
                        pm
                        and tf.exists()
                        and pm.get("pulls") == m.pulls
                        and pm.get("tag_count") == m.tag_count
                        and pm.get("updated_title") == m.updated_title
                    ):
                        # Unchanged — load from cache
                        try:
                            existing = json.loads(tf.read_text())
                            m.tags = [Tag(**t) for t in existing.get("tags", [])]
                            cached += 1
                            continue
                        except Exception:
                            pass
                    to_fetch.append(m)
                log.info(
                    "=== fetching per-model tags (%d cached, %d to fetch) ===",
                    cached,
                    len(to_fetch),
                )
                total = len(to_fetch)
                for i, m in enumerate(to_fetch, 1):
                    slug = slugify(m.path)
                    log.info("  [%d/%d] %s", i, total, m.path)
                    m.tags = fetch_tags(client, m)
                    save_tags(m, m.tags)
                    if i % 10 == 0:
                        save_models(models.values())
                        log.info(
                            "  checkpoint: saved models.json (%d models)",
                            len(models),
                        )
                    time.sleep(DELAY)
            else:
                log.info("=== fetching per-model tags ===")
                total = len(models)
                models_done = 0
                for i, m in enumerate(models.values(), 1):
                    if client.bail_out or _time_up():
                        log.warning(
                            "STOPPING at model %d/%d (bail=%s, time_up=%s)",
                            i,
                            total,
                            client.bail_out,
                            _time_up(),
                        )
                        break
                    slug = slugify(m.path)
                    tf = TAGS_DIR / f"{slug}.json"
                    if tf.exists():
                        try:
                            existing = json.loads(tf.read_text())
                            cached_count = len(existing.get("tags", []))
                            if cached_count == m.tag_count:
                                m.tags = [Tag(**t) for t in existing.get("tags", [])]
                                log.info(
                                    "  [%d/%d] %s (cached %d tags)",
                                    i,
                                    total,
                                    m.path,
                                    len(m.tags),
                                )
                                continue
                            else:
                                log.info(
                                    "  [%d/%d] %s (tag count changed: %d -> %d)",
                                    i,
                                    total,
                                    m.path,
                                    cached_count,
                                    m.tag_count,
                                )
                        except Exception:
                            pass
                    log.info("  [%d/%d] %s", i, total, m.path)
                    m.tags = fetch_tags(client, m)
                    save_tags(m, m.tags)
                    models_done += 1
                    if models_done % CHECKPOINT_EVERY == 0:
                        save_models(models.values())
                        git_checkpoint(f"tags {i}/{total}")
                    time.sleep(DELAY)

        if not args.skip_pages:
            log.info("=== fetching model pages (official only) ===")
            PAGES_DIR.mkdir(parents=True, exist_ok=True)
            official_models = [m for m in models.values() if m.official]
            total = len(official_models)
            pages_done = 0
            for i, m in enumerate(official_models, 1):
                if client.bail_out or _time_up():
                    log.warning("STOPPING at page %d/%d", i, total)
                    break
                slug = slugify(m.path)
                pf = PAGES_DIR / f"{slug}.json"
                if args.smart and prev_models:
                    pm = prev_models.get(m.path)
                    if (
                        pm
                        and pf.exists()
                        and pm.get("updated_title") == m.updated_title
                    ):
                        try:
                            json.loads(pf.read_text())
                            continue
                        except Exception:
                            pass
                elif pf.exists():
                    continue
                log.info("  [%d/%d] %s", i, total, m.path)
                page = fetch_model_page(client, m)
                if page:
                    save_model_page(m, page)
                    pages_done += 1
                    if pages_done % CHECKPOINT_EVERY == 0:
                        git_checkpoint(f"pages {i}/{total}")
                else:
                    log.warning("  no page data for %s", m.path)
                time.sleep(DELAY)

        if not args.skip_tag_pages:
            log.info("=== fetching tag pages (official only: all tags) ===")
            TAG_PAGES_DIR.mkdir(parents=True, exist_ok=True)
            official_models = [m for m in models.values() if m.official]
            total_tag_pages = 0
            tag_pages_done = 0
            for i, m in enumerate(official_models, 1):
                if client.bail_out:
                    log.warning(
                        "BAILING OUT at tag pages %d/%d", i, len(official_models)
                    )
                    break
                if not m.tags:
                    continue
                # Fetch ALL tags for this model
                tags_to_fetch = list(m.tags)
                if not tags_to_fetch:
                    continue
                for t in tags_to_fetch:
                    slug = slugify(m.path)
                    tf = TAG_PAGES_DIR / f"{slug}__{t.name}.json"
                    # Smart mode: skip if model unchanged and file exists
                    if args.smart and prev_models:
                        pm = prev_models.get(m.path)
                        if (
                            pm
                            and tf.exists()
                            and pm.get("updated_title") == m.updated_title
                        ):
                            continue
                    if tf.exists() and not args.smart:
                        continue  # already cached
                    log.info("  [%d/%d] %s:%s", i, len(official_models), m.path, t.name)
                    tp = fetch_tag_page(client, m, t.name)
                    if tp:
                        save_tag_page(m, t.name, tp)
                        total_tag_pages += 1
                        tag_pages_done += 1
                        if tag_pages_done % CHECKPOINT_EVERY == 0:
                            git_checkpoint(f"tag pages {i}/{len(official_models)}")
                    time.sleep(DELAY)
            log.info("fetched %d tag pages", total_tag_pages)
            save_models(models.values())
            git_checkpoint("tag pages done")

        if not args.skip_blobs:
            log.info("=== fetching blob pages (official only) ===")
            BLOBS_DIR.mkdir(parents=True, exist_ok=True)
            official_models = [m for m in models.values() if m.official]
            total_blobs = 0
            blobs_done = 0
            for i, m in enumerate(official_models, 1):
                if client.bail_out or _time_up():
                    log.warning("STOPPING at blobs %d/%d", i, len(official_models))
                    break
                if not m.tags:
                    continue
                for t in m.tags:
                    slug = slugify(m.path)
                    tp_file = TAG_PAGES_DIR / f"{slug}__{t.name}.json"
                    if not tp_file.exists():
                        continue
                    try:
                        tp_data = json.loads(tp_file.read_text())
                    except Exception:
                        continue
                    for f in tp_data.get("files", []):
                        blob_url = f.get("blob_url", "")
                        if not blob_url:
                            continue
                        # Sanitize blob_url for filename
                        safe = blob_url.strip("/").replace("/", "__").replace(":", "_")
                        bf = BLOBS_DIR / f"{safe}.json"
                        # Smart mode: skip if model unchanged and blob file exists
                        if args.smart and prev_models:
                            pm = prev_models.get(m.path)
                            if (
                                pm
                                and bf.exists()
                                and pm.get("updated_title") == m.updated_title
                            ):
                                continue
                        if bf.exists() and not args.smart:
                            continue  # already cached
                        log.info(
                            "  [%d/%d] blob: %s:%s",
                            i,
                            len(official_models),
                            m.path,
                            t.name,
                        )
                        bp = fetch_blob_page(client, blob_url)
                        if bp:
                            save_blob_page(blob_url, bp)
                            total_blobs += 1
                            blobs_done += 1
                            log.info(
                                "    blob done: %d tensors, %d chars content",
                                len(bp.tensors),
                                len(bp.content),
                            )
                            if blobs_done % CHECKPOINT_EVERY == 0:
                                git_checkpoint(
                                    f"blobs {i}/{len(official_models)} ({total_blobs} total)"
                                )
                        time.sleep(DELAY)
                if i % 10 == 0:
                    log.info(
                        "  processed %d/%d models (%d blobs)",
                        i,
                        len(official_models),
                        total_blobs,
                    )
                    save_models(models.values())
            log.info("fetched %d blob pages", total_blobs)
            save_models(models.values())
            git_checkpoint("blobs done")

        # Compute cloud_only flag: True if model has cloud badge but no
        # downloadable local tags (all tags are named "cloud" or have no size).
        for m in models.values():
            if m.cloud and m.tags:
                local_tags = [
                    t
                    for t in m.tags
                    if t.name != "cloud"
                    and not t.name.endswith("-cloud")
                    and t.size_bytes
                ]
                m.cloud_only = len(local_tags) == 0

        save_models(models.values())

        # Update sort orderings + derived rank data (in case models changed)
        if sort_orders:
            save_sort_data(sort_orders, models)

        git_checkpoint("final")
        if client.bail_out:
            log.warning("DONE (bailed out, %d HTTP requests)", client.requests)
        elif _time_up():
            log.warning("DONE (time limit reached, %d HTTP requests)", client.requests)
        else:
            log.info("done (%d HTTP requests)", client.requests)
        return 0
    finally:
        client.close()


def self_check() -> int:
    """Validate models.json + tags/*.json for internal consistency."""
    ok = True
    models_file = DATA / "models.json"
    if not models_file.exists():
        print("models.json missing", file=sys.stderr)
        return 1
    data = json.loads(models_file.read_text())
    print(f"models.json: {data['count']} models")
    missing_tags = 0
    bad_format = 0
    bad_owner = 0
    for m in data["models"]:
        slug = slugify(m["path"])
        tf = TAGS_DIR / f"{slug}.json"
        if not tf.exists():
            missing_tags += 1
            ok = False
            print(f"  MISSING tags: {m['path']}")
            continue
        td = json.loads(tf.read_text())
        for t in td.get("tags", []):
            if t.get("format") not in ("gguf", "mlx"):
                bad_format += 1
        if m["official"] and not m["path"].startswith("/library/"):
            bad_owner += 1
            print(f"  BAD official flag: {m['path']}")
        if not m["official"]:
            parts = m["path"].strip("/").split("/")
            if len(parts) < 2 or parts[0] == "library":
                bad_owner += 1
                print(f"  BAD user path: {m['path']}")
    print(f"missing tag files: {missing_tags}")
    print(f"bad format classifications: {bad_format}")
    print(f"bad owner/path: {bad_owner}")
    return 0 if ok and missing_tags == 0 and bad_format == 0 and bad_owner == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

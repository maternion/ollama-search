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
import re
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

# Polite crawling: small delay between requests, generous timeout.
DELAY = 1.0
TIMEOUT = 30.0
UA = "ollama-search-scraper/0.1 (+https://github.com/anomalyco/opencode)"

log = logging.getLogger("scraper")

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

    def get(self, url: str) -> str | None:
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=TIMEOUT)
                self.requests += 1
                if r.status_code == 200:
                    return r.text
                log.warning("GET %s -> %s", url, r.status_code)
                if r.status_code in (404, 410):
                    return None
                if r.status_code in (429, 502, 503, 504):
                    time.sleep(5 * (attempt + 1))
                    continue
                return None
            except requests.RequestException as e:
                log.warning("error %s: %s", url, e)
                time.sleep(2 * (attempt + 1))
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
    tags: list[Tag] = field(default_factory=list)


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


def detect_format(tag_name: str) -> str:
    """MLX tags use the -mlx suffix; everything else is GGUF."""
    return (
        "mlx"
        if re.search(r"(?:^|[-_])mlx(?:$|[-_])", tag_name, re.IGNORECASE)
        else "gguf"
    )


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
        tags.append(
            Tag(
                name=tag_name,
                size_bytes=parse_size_bytes(size_text),
                size_text=size_text,
                context=context,
                input_type=input_type,
                digest=digest,
                updated=updated,
                format=detect_format(tag_name),
            )
        )

    return tags


# --------------------------------------------------------------------------- #
# Crawl orchestration
# --------------------------------------------------------------------------- #


def crawl_official(client: Client) -> tuple[dict[str, Model], dict[str, list[str]]]:
    """Crawl /library?sort=popular and newest; union the results.

    Returns (models_by_path, sort_orders) where sort_orders maps
    sort name → list of model paths in document order.
    """
    found: dict[str, Model] = {}
    orders: dict[str, list[str]] = {}
    for sort in ("popular", "newest"):
        url = f"{BASE}/library?sort={sort}"
        log.info("crawling %s", url)
        html = client.get(url)
        if html is None:
            log.error("failed to fetch %s", url)
            continue
        cards = parse_cards(html, url)
        log.info("  %s: %d cards", sort, len(cards))
        orders[sort] = [m.path for m in cards]
        for m in cards:
            if m.path not in found:
                found[m.path] = m
        time.sleep(DELAY)
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
# Persistence
# --------------------------------------------------------------------------- #


def save_models(models: Iterable[Model]) -> None:
    data: dict = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "count": 0,
        "models": [],
    }
    ms = sorted(models, key=lambda m: (not m.official, m.name.lower()))
    data["models"] = [asdict(m) for m in ms]
    data["count"] = len(data["models"])
    (DATA / "models.json").write_text(json.dumps(data, indent=2))
    log.info("wrote models.json (%d models)", data["count"])


def save_tags(model: Model, tags: list[Tag]) -> None:
    TAGS_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(model.path)
    out = {
        "path": model.path,
        "name": model.name,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "tags": [asdict(t) for t in tags],
    }
    (TAGS_DIR / f"{slug}.json").write_text(json.dumps(out, indent=2))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def check_only() -> int:
    """Fetch /library?sort=newest (1 request), hash card data, compare to cache.

    Exit 0 = no change, 1 = changed (or first run).
    Writes the hash to scraper/.catalog-hash.
    """
    import hashlib

    client = Client()
    try:
        html = client.get(f"{BASE}/library?sort=newest")
        if html is None:
            log.error("failed to fetch /library?sort=newest")
            return 1
        cards = parse_cards(html, f"{BASE}/library?sort=newest")
        log.info("fetched %d cards from /library?sort=newest", len(cards))
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
        cache_file.write_text(current)
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
        "--skip-search",
        action="store_true",
        help="skip /search user-model sweep (official only)",
    )
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="validate existing data and exit",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="fetch only /library?sort=newest (1 request), hash card data, "
        "compare to scraper/.catalog-hash, exit 0=unchanged 1=changed",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

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
            return 0

        # ---- full crawl ----
        log.info("=== crawling official catalog ===")
        models, sort_orders = crawl_official(client)
        log.info("official models: %d", len(models))

        if not args.skip_search:
            log.info("=== crawling /search for user models ===")
            before = len(models)
            crawl_search(client, models)
            log.info(
                "user-model sweep added %d models (total %d)",
                len(models) - before,
                len(models),
            )

        if not args.skip_tags:
            log.info("=== fetching per-model tags ===")
            total = len(models)
            for i, m in enumerate(models.values(), 1):
                slug = slugify(m.path)
                tf = TAGS_DIR / f"{slug}.json"
                if tf.exists():
                    try:
                        existing = json.loads(tf.read_text())
                        m.tags = [Tag(**t) for t in existing.get("tags", [])]
                        log.info(
                            "  [%d/%d] %s (cached %d tags)",
                            i,
                            total,
                            m.path,
                            len(m.tags),
                        )
                        continue
                    except Exception:
                        pass
                log.info("  [%d/%d] %s", i, total, m.path)
                m.tags = fetch_tags(client, m)
                save_tags(m, m.tags)
                time.sleep(DELAY)

        save_models(models.values())

        # Save sort orderings + derived rank data for build.py
        if sort_orders:
            (DATA / "sort_orders.json").write_text(json.dumps(sort_orders, indent=2))
            log.info("wrote sort_orders.json")
            # Build per-model rank dict from the orderings
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
            # Fill missing ranks with 9999
            for name in all_names:
                if name not in ranks:
                    ranks[name] = {}
                for sort_name in sort_orders:
                    key = f"{sort_name}_rank"
                    ranks[name].setdefault(key, 9999)
            (DATA / "sort_ranks.json").write_text(json.dumps(ranks, indent=2))
            log.info("wrote sort_ranks.json")

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
            if t.get("format") == "mlx" and not re.search(
                r"(?:^|[-_])mlx(?:$|[-_])", t["name"], re.IGNORECASE
            ):
                bad_format += 1
                print(f"  BAD MLX classification: {m['path']} tag={t['name']}")
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

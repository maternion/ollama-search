from __future__ import annotations

from .parse import extract_hparams, map_arch, get_kv_dtype_bytes, parse_head_count_kv
from . import standard, swa, mla, hybrid

NO_KV_ARCHS = {
    "bert",
    "nomic-bert",
    "nomic-bert-moe",
    "neo-bert",
    "jina-bert-v2",
    "jina-bert-v3",
    "eurobert",
    "modern-bert",
    "clip",
    "gemma-embedding",
    "wavtokenizer-dec",
    "dream",
    "llada",
    "llada-moe",
    "rnd1",
    "canary",
    "t5encoder",
    "pangu-embed",
    "paddleocr",
}

MLA_ARCHS = {"deepseek4", "deepseek32", "glm-dsa", "glm4moelite"}

HYBRID_ARCHS = {
    "qwen35",
    "qwen35moe",
    "qwen3next",
    "granitehybrid",
    "nemotron_h",
    "nemotron_h_moe",
    "lfm2",
    "lfm2moe",
    "jamba",
    "falcon-h1",
    "kimi-linear",
    "plamo2",
}

SWA_ARCHS = {"gemma2", "gemma3", "gemma3n", "gemma4", "llama4", "mellum", "olmo3"}


def detect_family(hparams: dict) -> str:
    """Return the memory family for a blob's hyperparameters.

    Returns one of ``"standard"``, ``"swa"``, ``"mla"``, ``"hybrid"`` or
    ``"none"``. The ``"none"`` family has no KV cache.
    """
    block_count = hparams.get("block_count", 0)
    if not block_count:
        return "none"

    arch = map_arch(hparams.get("general.architecture", ""))

    if arch in NO_KV_ARCHS:
        return "none"
    if arch in MLA_ARCHS:
        return "mla"
    if arch == "deepseek2":
        if "attention.kv_lora_rank" in hparams:
            return "mla"
        return "standard"
    if arch in HYBRID_ARCHS:
        return "hybrid"
    if any(k.startswith("ssm.") or k.startswith("shortconv.") for k in hparams):
        return "hybrid"
    has_swa = hparams.get("attention.sliding_window", 0) or hparams.get(
        "attention.chunk_size", 0
    )
    if has_swa and int(has_swa) > 0:
        return "swa"
    return "standard"


def compute_memory_at_context(hparams: dict, context: int, kv_bpe: float = 2.0) -> dict:
    """Dispatch to the formula module matching the blob's memory family.

    Returns a dict describing KV cache memory at the given context length.
    Default kv_bpe=2.0 (F16) is the llama.cpp default KV cache dtype.
    """
    family = detect_family(hparams)

    if family == "none":
        result: dict = {"kv_bytes": 0, "family": "none"}
        return result
    if family == "standard":
        result = standard.compute_standard_kv(hparams, context, kv_bpe)
    elif family == "swa":
        result = swa.compute_swa_kv(hparams, context, kv_bpe)
    elif family == "mla":
        result = mla.compute_mla_kv(hparams, context, kv_bpe)
    elif family == "hybrid":
        result = hybrid.compute_hybrid_kv(hparams, context, kv_bpe)
    else:
        result = {"kv_bytes": 0, "family": family}

    if "family" not in result:
        result["family"] = family
    return result


def compute_memory_curve(
    hparams: dict,
    max_context: int | None = None,
    n_points: int = 50,
    kv_bpe: float = 2.0,
) -> list[dict]:
    """Compute KV cache memory at multiple context points for graphing.

    Generates ``n_points`` linearly spaced context values from 0 to
    ``max_context`` (inclusive) and returns a list of per-point memory dicts.
    """
    if max_context is None:
        max_context = hparams.get("context_length", 8192)

    if n_points < 2:
        n_points = 2

    step = max_context / (n_points - 1) if max_context > 0 else 0
    contexts = [int(round(i * step)) for i in range(n_points)]
    contexts[-1] = max_context

    curve: list[dict] = []
    for ctx in contexts:
        point = compute_memory_at_context(hparams, ctx, kv_bpe)
        kv_bytes = point.get("kv_bytes", 0)
        entry = {
            "context": ctx,
            "kv_bytes": kv_bytes,
            "kv_gib": kv_bytes / 1073741824,
            "family": point.get("family", detect_family(hparams)),
        }
        for key, value in point.items():
            if key not in entry:
                entry[key] = value
        curve.append(entry)
    return curve

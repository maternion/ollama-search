from __future__ import annotations

import json
import re

__all__ = [
    "extract_hparams",
    "map_arch",
    "get_kv_dtype_bytes",
    "weight_bytes_per_param",
    "parse_head_count_kv",
    "get_hparam",
]

_ARCH_MAP = {
    "mllama": "llama4",
    "gptoss": "gpt-oss",
    "qwen25vl": "qwen2vl",
    "deepseekocr": "deepseek2-ocr",
    "nemotron_h_omni": "nemotron_h_moe",
}

_DTYPE_BYTES: dict[str, float] = {
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "Q8_0": 1.0625,
    "Q4_0": 0.5625,
    "Q4_1": 0.625,
    "Q5_0": 0.6875,
    "Q5_1": 0.75,
    "IQ4_NL": 0.5625,
    "Q2_K": 0.328125,
    "Q4_K": 0.5625,
    "Q4_K_S": 0.5625,
    "Q4_K_M": 0.5625,
    "Q5_K": 0.6875,
    "Q5_K_S": 0.6875,
    "Q5_K_M": 0.6875,
    "Q6_K": 0.8203125,
    "Q3_K_S": 0.430,
    "Q3_K_M": 0.430,
    "Q3_K_L": 0.430,
    "IQ2_XXS": 0.328125,
    "IQ2_XS": 0.328125,
    "IQ3_XXS": 0.430,
    "IQ3_S": 0.430,
    "IQ4_XS": 0.5625,
    "MXFP4": 0.53125,
    "TQ1_0": 0.15625,
    "TQ2_0": 0.1875,
    "IQ1_S": 0.15625,
    "IQ1_M": 0.203125,
}

_DEFAULT_BPE = 2.0


def _coerce_value(value):
    """Coerce a scraped GGUF metadata value to its native Python type.

    GGUF metadata arrives as strings from the scraper. Numeric strings become
    ``int`` or ``float``; JSON-array strings become ``list``; anything else is
    returned as-is. Empty strings become ``None``.

    The scraper truncates long arrays with a trailing ``...`` (e.g.
    ``"[16, 16, 16, 16, 16, ...]"``). Such truncated arrays are parsed by
    extracting the visible elements; the caller can replicate the last value
    to fill the expected length if needed.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return None
    if stripped.startswith("["):
        if stripped.endswith("...]"):
            inner = stripped.rstrip("...]").strip().rstrip(",").strip()
            if inner.startswith("["):
                inner = inner[1:].strip()
            try:
                parsed = json.loads("[" + inner + "]")
            except (json.JSONDecodeError, ValueError):
                return value
            if isinstance(parsed, list):
                return [_coerce_scalar(v) for v in parsed]
            return value
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return value
        if isinstance(parsed, list):
            return [_coerce_scalar(v) for v in parsed]
        return value
    return _coerce_scalar(stripped)


def _coerce_scalar(s):
    """Coerce a single scalar string to int or float, else return as-is."""
    if isinstance(s, bool):
        return s
    if isinstance(s, (int, float)):
        return s
    if not isinstance(s, str):
        return s
    stripped = s.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    try:
        return float(stripped)
    except ValueError:
        return s


def extract_hparams(blob_dict: dict) -> dict:
    """Extract and normalize GGUF hparams from a scraped blob metadata dict.

    The blob dict carries a ``metadata`` list of ``{"key": ..., "value": ...}``
    pairs. Keys are prefixed with the architecture name (e.g.
    ``"llama.block_count"``); this prefix is stripped for keys belonging to the
    declared architecture while ``general.*`` keys are kept verbatim.

    Returns a flat dict of normalized hparams. Values are type-coerced:
    numeric strings become int/float, JSON-array strings become lists.
    Returns an empty dict when metadata is missing or empty.
    """
    metadata = blob_dict.get("metadata") or []
    if not isinstance(metadata, list) or not metadata:
        return {}

    arch: str | None = None
    normalized: dict = {}

    for entry in metadata:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        value = entry.get("value")
        if not isinstance(key, str):
            continue

        if key == "general.architecture" and isinstance(value, str):
            arch = value
            normalized[key] = value
            continue

        if key.startswith("general."):
            normalized[key] = _coerce_value(value)
            continue

        if arch and key.startswith(arch + "."):
            stripped = key[len(arch) + 1 :]
            normalized[stripped] = _coerce_value(value)
        else:
            normalized[key] = _coerce_value(value)

    tensors = blob_dict.get("tensors") or []
    if isinstance(tensors, list):
        normalized["_tensor_names"] = [
            t.get("name", "") for t in tensors if isinstance(t, dict)
        ]

    return normalized


def map_arch(arch_string: str) -> str:
    """Map a blob architecture name to its llama.cpp canonical family name.

    Unknown architectures are returned unchanged.
    """
    if not isinstance(arch_string, str):
        return arch_string
    return _ARCH_MAP.get(arch_string, arch_string)


def get_kv_dtype_bytes(file_type: str) -> float:
    """Return bytes-per-element for a GGUF quantization type as used by the KV cache.

    The default KV cache type in llama.cpp is F16 (2.0 bytes) regardless of the
    weight quantization, but this function supports arbitrary KV types for
    "what if I quantize the KV cache too" scenarios. Unknown types default to
    2.0 (F16).
    """
    if not isinstance(file_type, str):
        return _DEFAULT_BPE
    return _DTYPE_BYTES.get(file_type, _DEFAULT_BPE)


def weight_bytes_per_param(file_type: str) -> float:
    """Return bytes-per-weight-param for a GGUF quantization type.

    This is the same ggml type bytes/element table used for the KV cache; it is
    provided separately for weight-memory computations (e.g. estimating weight
    footprint from a param count rather than the blob ``size`` field).
    """
    return get_kv_dtype_bytes(file_type)


def parse_head_count_kv(raw_value):
    """Normalize a raw ``attention.head_count_kv`` value.

    Returns a list of ints when ``raw_value`` is a JSON array string or an
    existing list, an int when it is a scalar int or numeric string, and
    ``None`` on failure.
    """
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        if raw_value.is_integer():
            return int(raw_value)
        return None
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return None
        if stripped.startswith("["):
            # Scraper truncates long arrays with a trailing "..." element;
            # parse the visible portion into a list of ints.
            if stripped.endswith("...]"):
                inner = stripped.rstrip("...]").strip().rstrip(",").strip()
                if inner.startswith("["):
                    inner = inner[1:].strip()
                candidate = "[" + inner + "]"
            else:
                candidate = stripped
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(parsed, list):
                out = []
                for v in parsed:
                    if isinstance(v, bool):
                        out.append(int(v))
                    elif isinstance(v, int):
                        out.append(v)
                    elif isinstance(v, float) and v.is_integer():
                        out.append(int(v))
                    else:
                        try:
                            out.append(int(v))
                        except (TypeError, ValueError):
                            out.append(v)
                return out
            return None
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        try:
            f = float(stripped)
        except ValueError:
            return None
        if f.is_integer():
            return int(f)
        return None
    return None


def get_hparam(hparams: dict, key: str, arch: str, default=None):
    """Fetch a hparam, trying the raw ``arch + "." + key`` form first.

    This is a safety net for cases where :func:`extract_hparams` did not strip
    the architecture prefix as expected. Falls back to the stripped ``key``
    form, then to ``default``.
    """
    if not isinstance(hparams, dict):
        return default
    if arch:
        raw_key = f"{arch}.{key}"
        if raw_key in hparams:
            return hparams[raw_key]
    if key in hparams:
        return hparams[key]
    return default

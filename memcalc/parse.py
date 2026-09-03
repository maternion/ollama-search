from __future__ import annotations

import json
import re

__all__ = [
    "extract_hparams",
    "extract_hparams_from_config",
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
    "kimi_linear": "kimi-linear",
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
        normalized["_tensor_shapes"] = {
            t.get("name", ""): t.get("shape")
            for t in tensors
            if isinstance(t, dict) and t.get("shape")
        }

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


def extract_hparams_from_config(config: dict) -> dict:
    """Convert a HuggingFace-style config.json dict to memcalc hparams.

    This is used for MLX models (whose blob pages on ollama.com expose a
    config.json rather than GGUF metadata) and as a HuggingFace fallback
    for frob models. The mapping mirrors the ``key_map`` in the scraper's
    ``_hf_fetch_base_model_config`` but operates directly on the parsed
    JSON dict rather than constructing GGUF MetadataEntry objects.

    Returns a hparams dict with the same structure as
    :func:`extract_hparams` — arch prefix already stripped, keys in the
    ``attention.*``, ``block_count``, etc. format that memcalc dispatch
    expects.
    """
    tc = config.get("text_config", config)

    arch = tc.get("model_type", config.get("model_type", ""))
    if not arch:
        return {}

    # Core keys common to most architectures.
    key_map = {
        "num_hidden_layers": "block_count",
        "num_attention_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv",
        "hidden_size": "embedding_length",
        "intermediate_size": "feed_forward_length",
        "max_position_embeddings": "context_length",
        "head_dim": "attention.key_length",
        "sliding_window": "attention.sliding_window",
        "kv_lora_rank": "attention.kv_lora_rank",
        "q_lora_rank": "attention.q_lora_rank",
        "first_k_dense_replace": "first_k_dense_replace",
        "nextn_predict_layers": "nextn_predict_layers",
        "mtp_num_hidden_layers": "nextn_predict_layers",
        "full_attention_interval": "full_attention_interval",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
        "moe_intermediate_size": "expert_feed_forward_length",
    }

    hparams: dict = {
        "general.architecture": arch,
        "general.file_type": "0",
    }

    for hf_key, gguf_key in key_map.items():
        val = tc.get(hf_key, config.get(hf_key))
        if val is not None and not isinstance(val, (dict, list)):
            hparams[gguf_key] = val

    # attention.value_length — some configs provide it separately.
    v_head_dim = tc.get("v_head_dim")
    if v_head_dim is not None:
        hparams["attention.value_length"] = v_head_dim

    # Hybrid: Gated DeltaNet / linear attention recurrent state.
    # Map the linear_* config keys to ssm.* / linear.* hparams that
    # hybrid.py's _compute_recurrent_state can consume.
    linear_num_key_heads = tc.get("linear_num_key_heads")
    linear_num_value_heads = tc.get("linear_num_value_heads")
    linear_key_head_dim = tc.get("linear_key_head_dim")
    linear_value_head_dim = tc.get("linear_value_head_dim")
    linear_conv_kernel_dim = tc.get("linear_conv_kernel_dim")
    if linear_num_key_heads and linear_key_head_dim:
        hparams["linear.num_key_heads"] = linear_num_key_heads
        hparams["linear.num_value_heads"] = (
            linear_num_value_heads or linear_num_key_heads
        )
        hparams["linear.key_head_dim"] = linear_key_head_dim
        hparams["linear.value_head_dim"] = linear_value_head_dim or linear_key_head_dim
        if linear_conv_kernel_dim:
            hparams["linear.conv_kernel"] = linear_conv_kernel_dim

    # Indexer attention (Qwen4-Exp n-gram embedding attention).
    indexer_kv_heads = tc.get("indexer_kv_heads")
    indexer_head_dim = tc.get("indexer_head_dim")
    indexer_budget = tc.get("indexer_budget")
    if indexer_kv_heads and indexer_head_dim:
        hparams["attention.indexer.head_count_kv"] = indexer_kv_heads
        hparams["attention.indexer.key_length"] = indexer_head_dim
        if indexer_budget:
            hparams["attention.indexer.budget"] = indexer_budget

    # linear_attn_config (Kimi-K3 and similar hybrid models).
    lac = tc.get("linear_attn_config")
    if isinstance(lac, dict):
        n_layers = int(tc.get("num_hidden_layers", 0))
        full_attn = lac.get("full_attn_layers", [])
        if full_attn and n_layers > 0:
            n_kv = int(
                tc.get("num_key_value_heads", config.get("num_key_value_heads", 0))
            )
            kv_arr = [0] * n_layers
            for idx in full_attn:
                li = int(idx) - 1
                if 0 <= li < n_layers:
                    kv_arr[li] = n_kv
            hparams["attention.head_count_kv"] = kv_arr

        head_dim_lac = lac.get("head_dim")
        if head_dim_lac:
            hparams["linear.head_dim"] = head_dim_lac

        n_heads_lin = int(lac.get("num_heads", 0))
        conv_k = int(lac.get("short_conv_kernel_size", 0))
        if n_heads_lin > 0 and head_dim_lac:
            inner = n_heads_lin * int(head_dim_lac)
            hparams["ssm.inner_size"] = inner
            hparams["ssm.state_size"] = head_dim_lac
            hparams["ssm.group_count"] = n_heads_lin
            if conv_k > 0:
                hparams["ssm.conv_kernel"] = conv_k

    # Ensure we have block_count — essential for dispatch.
    if "block_count" not in hparams:
        return {}

    return hparams

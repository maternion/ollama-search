from __future__ import annotations

__all__ = ["compute_mla_kv"]

_DSA_ARCHS = {"deepseek32", "deepseek4", "glm-dsa"}
_DEFAULT_N_ROT = 64


def _as_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except (ValueError, TypeError):
            return default
    return default


def _resolve_stored_k_width(hparams: dict) -> int:
    key_length = hparams.get("attention.key_length")
    if key_length is not None:
        w = _as_int(key_length)
        if w > 0:
            return w
    kv_lora_rank = hparams.get("attention.kv_lora_rank")
    if kv_lora_rank is not None:
        return _as_int(kv_lora_rank) + _DEFAULT_N_ROT
    embedding = _as_int(hparams.get("embedding_length"), 0)
    head_count = _as_int(hparams.get("attention.head_count"), 0)
    if head_count > 0:
        return embedding // head_count
    return 0


def _dense_kv_heads(hparams: dict, leading_dense: int) -> int:
    raw = hparams.get("attention.head_count_kv")
    if isinstance(raw, list):
        if leading_dense <= 0 or not raw:
            return 1
        return _as_int(raw[0], 1)
    val = _as_int(raw, 0)
    if val > 0:
        return val
    return 1


def compute_mla_kv(hparams: dict, context: int, kv_bpe: float = 2.0) -> dict:
    """Compute KV cache memory for MLA-family architectures.

    Covers DeepSeek-V2/V3 (``deepseek2`` with ``kv_lora_rank``), DeepSeek-V3.2
    (``deepseek32``/``deepseek4``), GLM-DSA and ``glm4moelite``.

    MLA stores a compressed K latent only (no V cache). The main MLA layers are
    converted to MQA (``n_kv_heads = 1``). Hybrid models with a few leading
    DENSE layers (``leading_dense_block_count``) use the standard K+V formula
    for those layers. DSA architectures additionally carry an indexer cache
    that is also K-only MQA.
    """
    n_layers = _as_int(hparams.get("block_count"), 0)
    arch = hparams.get("general.architecture", "")

    has_indexer = "attention.indexer.key_length" in hparams or arch in _DSA_ARCHS

    leading_dense = _as_int(hparams.get("leading_dense_block_count"), 0)
    if leading_dense < 0:
        leading_dense = 0
    if leading_dense > n_layers:
        leading_dense = n_layers

    n_mla_layers = n_layers - leading_dense
    n_dense_layers = leading_dense

    stored_k_width = _resolve_stored_k_width(hparams)

    mla_bytes = 0
    if n_mla_layers > 0:
        mla_bytes = n_mla_layers * context * 1 * stored_k_width * kv_bpe

    dense_bytes = 0
    if n_dense_layers > 0:
        head_dim_dense = _as_int(hparams.get("attention.key_length"), stored_k_width)
        if head_dim_dense <= 0:
            head_dim_dense = stored_k_width
        head_dim_value = _as_int(hparams.get("attention.value_length"), head_dim_dense)
        if head_dim_value <= 0:
            head_dim_value = head_dim_dense
        n_kv_dense = _dense_kv_heads(hparams, n_dense_layers)
        dense_bytes = (
            n_dense_layers
            * context
            * n_kv_dense
            * (head_dim_dense + head_dim_value)
            * kv_bpe
        )

    indexer_bytes = 0
    indexer_head_size = 0
    if has_indexer:
        indexer_head_size = _as_int(hparams.get("attention.indexer.key_length"), 0)
        if indexer_head_size > 0:
            indexer_bytes = n_layers * context * 1 * indexer_head_size * kv_bpe

    kv_bytes = mla_bytes + dense_bytes + indexer_bytes

    parts: list[str] = []
    if n_mla_layers > 0:
        parts.append(f"mla: {n_mla_layers}L * ctx * 1 * {stored_k_width} * {kv_bpe}")
    if n_dense_layers > 0:
        head_dim_dense = _as_int(hparams.get("attention.key_length"), stored_k_width)
        if head_dim_dense <= 0:
            head_dim_dense = stored_k_width
        head_dim_value = _as_int(hparams.get("attention.value_length"), head_dim_dense)
        if head_dim_value <= 0:
            head_dim_value = head_dim_dense
        n_kv_dense = _dense_kv_heads(hparams, n_dense_layers)
        parts.append(
            f"dense: {n_dense_layers}L * ctx * {n_kv_dense} * ({head_dim_dense}+{head_dim_value}) * {kv_bpe}"
        )
    if indexer_bytes > 0:
        parts.append(f"indexer: {n_layers}L * ctx * 1 * {indexer_head_size} * {kv_bpe}")
    formula = " + ".join(parts) if parts else "mla: 0"

    result: dict = {
        "kv_bytes": kv_bytes,
        "mla_bytes": mla_bytes,
        "dense_bytes": dense_bytes,
        "indexer_bytes": indexer_bytes,
        "n_layers": n_layers,
        "n_mla_layers": n_mla_layers,
        "n_dense_layers": n_dense_layers,
        "stored_k_width": stored_k_width,
        "has_indexer": has_indexer,
        "formula": formula,
        "family": "mla",
    }
    if has_indexer:
        result["indexer_head_size"] = indexer_head_size
    return result

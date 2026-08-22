from __future__ import annotations


def compute_standard_kv(hparams: dict, context: int, kv_bpe: float = 2.0) -> dict:
    """Compute KV cache memory for the standard family.

    Implements the llama.cpp formula (llama-kv-cache.cpp:245-246):

        KV_cache_bytes = sum_over_layers( kv_size * n_stream * (n_embd_k_gqa(il) * bpe_k + n_embd_v_gqa(il) * bpe_v) )

    For uniform layers (n_kv_heads constant, head_dim_k == head_dim_v) this
    simplifies to ``2 * n_layers * n_kv_heads * head_dim * context * bpe`` where
    the leading ``2`` accounts for K and V.

    ``hparams`` is the dict produced by ``parse.extract_hparams`` with the arch
    prefix already stripped. ``context`` is the context length to size the
    cache for (the caller supplies this; ``context_length`` in hparams is not
    used). ``kv_bpe`` is the bytes per element of the KV cache dtype.

    Returns a dict with ``kv_bytes`` and the metadata used to derive it
    (``n_layers``, ``n_kv_heads``, ``head_dim``, ``formula``, ``family``).
    """
    block_count = hparams.get("block_count")
    if not block_count:
        return {
            "kv_bytes": 0,
            "n_layers": 0,
            "n_kv_heads": 0,
            "head_dim": 0,
            "formula": "0 (block_count missing)",
            "family": "standard",
            "error": "block_count missing or zero",
        }

    n_layers = block_count

    head_count = hparams.get("attention.head_count", 0) or 0
    if head_count == 0:
        return {
            "kv_bytes": 0,
            "n_layers": n_layers,
            "n_kv_heads": 0,
            "head_dim": 0,
            "formula": "0 (attention.head_count is 0)",
            "family": "standard",
            "error": "attention.head_count is 0",
        }

    # Per-layer head_count (e.g. laguna) is not supported by the standard
    # formula; those arches route to SWA, but guard here so a misrouted blob
    # returns a clean error rather than a TypeError on embedding // head_count.
    if isinstance(head_count, list):
        return {
            "kv_bytes": 0,
            "n_layers": n_layers,
            "n_kv_heads": 0,
            "head_dim": 0,
            "formula": "0 (per-layer head_count unsupported in standard family)",
            "family": "standard",
            "error": "attention.head_count is a per-layer array; use SWA family",
        }

    embedding_length = hparams.get("embedding_length", 0) or 0

    if "attention.key_length" in hparams and hparams["attention.key_length"]:
        head_dim_k = hparams["attention.key_length"]
    else:
        head_dim_k = embedding_length // head_count

    if "attention.value_length" in hparams and hparams["attention.value_length"]:
        head_dim_v = hparams["attention.value_length"]
    else:
        head_dim_v = head_dim_k

    head_count_kv_raw = hparams.get("attention.head_count_kv")

    per_layer = kv_bytes_total = 0
    if isinstance(head_count_kv_raw, list):
        per_layer = 1
        # The scraper truncates long per-layer arrays to a handful of visible
        # elements with a trailing "..." (see parse._coerce_value). When the
        # array is shorter than n_layers, replicate the last visible value to
        # fill the expected length. This is correct for the uniform per-layer
        # arrays emitted by llama.cpp for arches like granite; a genuinely
        # non-uniform array would require the un-truncated source metadata.
        kv_list = list(head_count_kv_raw)
        if len(kv_list) < n_layers and kv_list:
            last = kv_list[-1]
            kv_list = kv_list + [last] * (n_layers - len(kv_list))
        head_count_kv_summary = sum(kv_list[:n_layers])
        kv_bytes_total = 0
        for il in range(n_layers):
            hkv = kv_list[il] if il < len(kv_list) else 0
            if not hkv:
                continue
            kv_bytes_total += (
                context * hkv * (head_dim_k * kv_bpe + head_dim_v * kv_bpe)
            )
    else:
        n_kv_heads = head_count_kv_raw if head_count_kv_raw else head_count
        head_count_kv_summary = n_kv_heads
        kv_bytes_total = (
            n_layers
            * context
            * n_kv_heads
            * (head_dim_k * kv_bpe + head_dim_v * kv_bpe)
        )

    head_dim = head_dim_k if head_dim_k == head_dim_v else (head_dim_k, head_dim_v)

    if per_layer:
        formula = (
            f"sum_il(context * head_count_kv[il] * "
            f"({head_dim_k} * {kv_bpe} + {head_dim_v} * {kv_bpe})) "
            f"over {n_layers} layers"
        )
    else:
        if head_dim_k == head_dim_v:
            formula = (
                f"2 * {n_layers} * {n_kv_heads} * {head_dim_k} * {context} * {kv_bpe}"
            )
        else:
            formula = (
                f"{n_layers} * {context} * {n_kv_heads} * "
                f"({head_dim_k} * {kv_bpe} + {head_dim_v} * {kv_bpe})"
            )

    return {
        "kv_bytes": kv_bytes_total,
        "n_layers": n_layers,
        "n_kv_heads": head_count_kv_summary,
        "head_dim": head_dim,
        "formula": formula,
        "family": "standard",
    }

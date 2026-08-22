from __future__ import annotations

# Per-architecture default SWA interleave patterns (local:global ratio) used
# when the GGUF metadata lacks an explicit ``attention.sliding_window_pattern``
# key. ``n`` = group size; the LAST layer of each group is dense (global), the
# rest are SWA — matching ``_is_swa_for_layer(il, n) => (il % n) < n-1``.
# Values confirmed from llama.cpp's hparams/arch code:
#   llama4  -> 4  (3 chunked SWA : 1 global iRoPE, chunk 8192)
#   gemma3  -> 6  (5 local : 1 global, window 1024)
#   gemma2  -> 2  (1:1, window 4096)
#   cohere2 -> 4  (3 SWA : 1 global, window 4096)
SWA_DEFAULT_PATTERNS = {
    "llama4": 4,
    "gemma3": 6,
    "gemma-embedding": 6,
    "gemma3n": 6,
    "gemma2": 2,
    "cohere2": 4,
    "gpt-oss": 2,
    "gptoss": 2,
}


def _as_int_list(value) -> list[int] | None:
    """Coerce ``value`` to a list of ints, or return ``None`` if absent."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) if not isinstance(v, bool) else (1 if v else 0) for v in value]
    if isinstance(value, bool):
        return [1 if value else 0]
    return [int(value)]


def _is_swa_for_layer(il: int, pattern, n_layers: int = 0) -> bool:
    """Return True if layer ``il`` is a SWA layer under ``pattern``.

    ``pattern`` may be an int (periodicity) or a per-layer array of 0/1/bool flags.
    For an int ``p``: ``is_swa = (il % p) < (p - 1)`` — the last layer of each
    group of ``p`` is dense, the rest are SWA.

    If the array is shorter than ``n_layers`` (scraper truncation), it is
    tiled cyclically to cover all layers. If all visible elements are True
    (SWA) and the array is truncated, a False (dense) is appended at the end
    of each cycle, since SWA patterns typically end with a dense layer.
    """
    if isinstance(pattern, (list, tuple)):
        plen = len(pattern)
        if plen == 0:
            return False
        all_true = all(
            (
                v
                if isinstance(v, bool)
                else (v.strip().lower() == "true")
                if isinstance(v, str)
                else bool(int(v))
            )
            for v in pattern
        )
        truncated = n_layers > 0 and plen < n_layers
        if all_true and truncated:
            idx = il % (plen + 1)
            return idx < plen
        idx = il if il < plen else il % plen
        v = pattern[idx]
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(int(v))
    p = int(pattern)
    if p <= 0:
        return False
    return (il % p) < (p - 1)


def compute_swa_kv(hparams: dict, context: int, kv_bpe: float = 2.0) -> dict:
    """Compute KV cache memory for sliding-window-attention architectures.

    SWA archs (gemma2/3/3N/4, llama4, mellum, olmo3, ...) maintain two KV
    caches: a dense one sized to the full context and a SWA one capped at the
    sliding window size ``n_swa``. This returns the combined byte count at the
    given ``context`` along with a breakdown.
    """
    n_layers_total = int(hparams.get("block_count", 0))
    n_swa = int(hparams.get("attention.sliding_window", 0) or 0)
    if not n_swa:
        n_swa = int(hparams.get("attention.chunk_size", 0) or 0)
    if not n_layers_total or not n_swa:
        return {"kv_bytes": 0, "family": "swa", "error": "no sliding_window"}

    head_count_raw = hparams.get("attention.head_count", 0)
    if isinstance(head_count_raw, (list, tuple)):
        head_count = int(head_count_raw[0]) if head_count_raw else 0
    else:
        head_count = int(head_count_raw)
    embedding_length = int(hparams.get("embedding_length", 0))

    key_length = hparams.get("attention.key_length")
    if key_length is not None:
        head_dim_dense = int(key_length)
    elif head_count > 0:
        head_dim_dense = embedding_length // head_count
    else:
        head_dim_dense = 0

    value_length = hparams.get("attention.value_length")
    head_dim_v_dense = int(value_length) if value_length is not None else head_dim_dense

    key_length_swa = hparams.get("attention.key_length_swa")
    head_dim_swa = int(key_length_swa) if key_length_swa is not None else head_dim_dense

    value_length_swa = hparams.get("attention.value_length_swa")
    if value_length_swa is not None:
        head_dim_v_swa = int(value_length_swa)
    elif key_length_swa is not None:
        head_dim_v_swa = head_dim_swa
    else:
        head_dim_v_swa = head_dim_v_dense

    pattern = hparams.get("attention.sliding_window_pattern")
    if pattern is None:
        # Blobs converted before the pattern key existed omit it. The true
        # pattern is arch-specific — llama4 interleaves 3 chunked SWA layers
        # with 1 global iRoPE layer (=4), gemma3 runs 5:1 local:global (=6),
        # gemma2 alternates 1:1 (=2, the old default). gpt-oss alternates
        # every other layer (=2). Anything unknown keeps the conservative 2.
        arch = hparams.get("general.architecture", "")
        pattern = SWA_DEFAULT_PATTERNS.get(arch, 2)

    shared_kv_layers = hparams.get("attention.shared_kv_layers")
    if shared_kv_layers is not None and int(shared_kv_layers) > 0:
        n_alloc = int(shared_kv_layers)
    else:
        n_alloc = n_layers_total
    if n_alloc > n_layers_total:
        n_alloc = n_layers_total

    n_kv_heads_raw = hparams.get("attention.head_count_kv")
    n_kv_heads_list = _as_int_list(n_kv_heads_raw)
    n_kv_heads_scalar = n_kv_heads_list[0] if n_kv_heads_list else head_count

    swa_full = False
    if swa_full:
        swa_size = context
    else:
        swa_size = min(context, n_swa)

    dense_bytes = 0
    swa_bytes = 0
    n_dense_layers = 0
    n_swa_layers = 0

    for il in range(n_alloc):
        is_swa = _is_swa_for_layer(il, pattern, n_alloc)
        if isinstance(n_kv_heads_raw, (list, tuple)):
            kvlen = len(n_kv_heads_raw)
            if kvlen > 0:
                idx = il if il < kvlen else il % kvlen
                n_kv = int(n_kv_heads_raw[idx])
            else:
                n_kv = 0
        else:
            n_kv = n_kv_heads_scalar
        if n_kv <= 0:
            continue

        if is_swa:
            n_swa_layers += 1
            swa_bytes += (
                swa_size * n_kv * (head_dim_swa * kv_bpe + head_dim_v_swa * kv_bpe)
            )
        else:
            n_dense_layers += 1
            dense_bytes += (
                context * n_kv * (head_dim_dense * kv_bpe + head_dim_v_dense * kv_bpe)
            )

    kv_bytes = dense_bytes + swa_bytes

    formula = (
        f"swa: dense={n_dense_layers}L*ctx + swa={n_swa_layers}L*min(ctx,{n_swa}), "
        f"kv_heads={n_kv_heads_scalar}, kdim={head_dim_dense}, kdim_swa={head_dim_swa}"
    )

    return {
        "kv_bytes": kv_bytes,
        "dense_bytes": dense_bytes,
        "swa_bytes": swa_bytes,
        "n_layers": n_layers_total,
        "n_dense_layers": n_dense_layers,
        "n_swa_layers": n_swa_layers,
        "n_swa": n_swa,
        "head_dim_dense": head_dim_dense,
        "head_dim_swa": head_dim_swa,
        "swa_effective_size": swa_size,
        "n_kv_heads": n_kv_heads_scalar,
        "formula": formula,
        "family": "swa",
    }

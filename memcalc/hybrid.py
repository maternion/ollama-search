from __future__ import annotations


def _as_int_list(value) -> list[int] | None:
    """Coerce ``value`` to a list of ints, or return ``None`` if absent."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _to_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _infer_attn_layers_from_tensors(
    tensor_names: list[str], n_layers: int
) -> set[int] | None:
    """Infer which layers have attention from tensor names.

    If any ``blk.N.attn_k.weight`` (or ``attn_q``) tensors exist, return the set
    of layer indices that have attention. Returns ``None`` if no attention
    tensors found (caller should fall back to other methods).
    """
    import re

    attn_layers: set[int] = set()
    pat = re.compile(r"^blk\.(\d+)\.attn_[kvq](_norm)?\.weight$")
    for name in tensor_names:
        m = pat.match(name)
        if m:
            attn_layers.add(int(m.group(1)))
    return attn_layers if attn_layers else None


def _split_layers(
    n_layers: int,
    head_count_kv_raw,
    full_attention_interval,
    feed_forward_length_raw,
    tensor_names: list[str] | None = None,
) -> tuple[list[bool], list[bool], list[bool]]:
    """Determine per-layer recurrent / attention / FFN-only flags.

    Returns three parallel lists of length ``n_layers``:
    ``(is_recr, is_attn, is_ffn_only)``.
    """
    is_recr = [False] * n_layers
    is_attn = [False] * n_layers
    is_ffn_only = [False] * n_layers

    has_kv_array = isinstance(head_count_kv_raw, (list, tuple))
    has_ffn_array = isinstance(feed_forward_length_raw, (list, tuple))

    interval = _to_int(full_attention_interval, 0) or 0

    kv_array_full = has_kv_array and len(head_count_kv_raw) >= n_layers

    if kv_array_full:
        kv_list = [int(v) for v in head_count_kv_raw]
        for il in range(n_layers):
            n_kv = kv_list[il]
            n_ff = 0
            if has_ffn_array:
                n_ff = (
                    int(feed_forward_length_raw[il])
                    if il < len(feed_forward_length_raw)
                    else 0
                )
            if n_ff > 0:
                is_ffn_only[il] = True
            elif n_kv == 0:
                is_recr[il] = True
            else:
                is_attn[il] = True
        return is_recr, is_attn, is_ffn_only

    # Truncated array. Try tensor-name inference first (most accurate).
    kv_array_truncated = has_kv_array and len(head_count_kv_raw) < n_layers
    if (kv_array_truncated or not has_kv_array) and tensor_names:
        attn_set = _infer_attn_layers_from_tensors(tensor_names, n_layers)
        if attn_set is not None:
            for il in range(n_layers):
                if il in attn_set:
                    is_attn[il] = True
                else:
                    is_recr[il] = True
            if has_ffn_array:
                for il in range(n_layers):
                    n_ff = (
                        int(feed_forward_length_raw[il])
                        if il < len(feed_forward_length_raw)
                        else 0
                    )
                    if n_ff > 0:
                        is_ffn_only[il] = True
                        is_attn[il] = False
                        is_recr[il] = False
            return is_recr, is_attn, is_ffn_only

    if has_kv_array and not interval:
        kv_list = [int(v) for v in head_count_kv_raw]
        has_nonzero = any(v > 0 for v in kv_list)
        if has_nonzero:
            period = len(kv_list)
            ffn_list = (
                [int(v) for v in feed_forward_length_raw] if has_ffn_array else []
            )
            for il in range(n_layers):
                n_kv = kv_list[il % period]
                n_ff = ffn_list[il % len(ffn_list)] if ffn_list else 0
                if n_ff > 0:
                    is_ffn_only[il] = True
                elif n_kv == 0:
                    is_recr[il] = True
                else:
                    is_attn[il] = True
            return is_recr, is_attn, is_ffn_only

    if interval <= 0:
        interval = 4

    for il in range(n_layers):
        is_attn[il] = (il + 1) % interval == 0
        is_recr[il] = not is_attn[il]

    if has_ffn_array:
        for il in range(n_layers):
            n_ff = (
                int(feed_forward_length_raw[il])
                if il < len(feed_forward_length_raw)
                else 0
            )
            if n_ff > 0:
                is_ffn_only[il] = True
                is_attn[il] = False
                is_recr[il] = False

    return is_recr, is_attn, is_ffn_only


def _compute_recurrent_state(
    n_recr_layers: int,
    embedding_length: int,
    ssm: dict,
    shortconv: dict,
) -> tuple[int, str]:
    """Return ``(recr_state_per_layer, recr_state_type)`` for the recurrent state.

    The state is stored in F32 (4 bytes). ``n_seq_max`` is assumed to be 1, the
    single-user inference default.
    """
    if ssm:
        d_conv = _to_int(ssm.get("conv_kernel"), 0)
        d_inner = _to_int(ssm.get("inner_size"), 0)
        d_state = _to_int(ssm.get("state_size"), 0)
        n_group = _to_int(ssm.get("group_count"), 0)
        if d_conv > 0 and d_inner > 0 and d_state > 0:
            n_embd_r = (d_conv - 1) * (d_inner + 2 * n_group * d_state)
            n_embd_s = d_state * d_inner
            return (n_embd_r + n_embd_s) * 4, "ssm"

    if shortconv:
        l_cache = _to_int(shortconv.get("l_cache"), 0)
        if l_cache > 0 and embedding_length > 0:
            n_embd_r = embedding_length * (l_cache - 1)
            return n_embd_r * 4, "shortconv"

    return 0, "none"


def compute_hybrid_kv(hparams: dict, context: int, kv_bpe: float = 2.0) -> dict:
    """Compute KV cache memory for HYBRID architectures.

    Hybrid models interleave attention layers (per-token KV cache, linear in
    context) with recurrent layers (fixed-size state, constant in context).
    The total memory graph is a straight line with a positive y-intercept equal
    to the recurrent state size.
    """
    n_layers = int(hparams.get("block_count", 0))
    embedding_length = int(hparams.get("embedding_length", 0))
    head_count = int(hparams.get("attention.head_count", 0))

    key_length = hparams.get("attention.key_length")
    if key_length is not None:
        head_dim = int(key_length)
    elif head_count > 0:
        head_dim = embedding_length // head_count
    else:
        head_dim = 0

    value_length = hparams.get("attention.value_length")
    head_dim_v = int(value_length) if value_length is not None else head_dim

    head_count_kv_raw = hparams.get("attention.head_count_kv")
    feed_forward_length_raw = hparams.get("feed_forward_length")
    full_attention_interval = hparams.get("full_attention_interval")

    is_recr, is_attn, is_ffn_only = _split_layers(
        n_layers,
        head_count_kv_raw,
        full_attention_interval,
        feed_forward_length_raw,
        tensor_names=hparams.get("_tensor_names"),
    )

    n_attn_layers = sum(1 for v in is_attn if v)
    n_recr_layers = sum(1 for v in is_recr if v)
    n_ffn_only_layers = sum(1 for v in is_ffn_only if v)

    has_kv_array = isinstance(head_count_kv_raw, (list, tuple))
    kv_scalar = 0
    if not has_kv_array and head_count_kv_raw is not None:
        kv_scalar = _to_int(head_count_kv_raw, 0)
    elif not has_kv_array and head_count_kv_raw is None:
        kv_scalar = head_count
    kv_list = [int(v) for v in head_count_kv_raw] if has_kv_array else []

    # Scraper truncates long head_count_kv arrays to a handful of visible
    # elements (e.g. "[0, 0, 0, 4, 0, ...]"). When the array is shorter than
    # block_count, _split_layers fell back to the full_attention_interval to
    # decide which layers are attention. The visible non-zero value is the KV
    # head count that applies to *every* attention layer, so tile it across
    # all attention layers. When the truncated array is entirely zeros (which
    # happens for hybrid arches whose recurrent layers come first and whose
    # attention layers are truncated away), there is no usable KV info, so
    # fall back to head_count (MHA) — consistent with the absent-key fallback
    # above and the documented "head_count_kv defaults to head_count" rule.
    kv_array_truncated = has_kv_array and len(kv_list) < n_layers
    if kv_array_truncated:
        kv_nonzero = next((v for v in kv_list if v > 0), 0) or head_count
        kv_list_tiled = [kv_nonzero] * n_layers if kv_nonzero else kv_list
    else:
        kv_list_tiled = kv_list

    # Check if this hybrid arch uses MLA-style compressed attention.
    # When kv_lora_rank is present, MLA layers store only a compressed K
    # latent (kv_lora_rank + decoupled rope dim), not full K+V.
    kv_lora_rank = hparams.get("attention.kv_lora_rank")
    has_mla = kv_lora_rank is not None
    _DEFAULT_N_ROT = 64
    mla_stored_k_width = 0
    if has_mla:
        mla_stored_k_width = _to_int(kv_lora_rank, 0) + _DEFAULT_N_ROT

    # Leading dense layers (before MLA layers) use standard K+V.
    leading_dense = _to_int(hparams.get("leading_dense_block_count"), 0)
    if leading_dense < 0:
        leading_dense = 0

    attn_bytes = 0
    per_token_bytes = head_dim * kv_bpe + head_dim_v * kv_bpe  # standard K+V
    for il in range(n_layers):
        if not is_attn[il]:
            continue
        if has_mla and il >= leading_dense:
            # MLA layer: only compressed K, 1 kv head (MQA)
            attn_bytes += context * 1 * mla_stored_k_width * kv_bpe
            continue
        if has_kv_array:
            n_kv = (
                kv_list_tiled[il]
                if il < len(kv_list_tiled)
                else (kv_list_tiled[-1] if kv_list_tiled else 0)
            )
        else:
            n_kv = kv_scalar
        if n_kv <= 0:
            continue
        attn_bytes += context * n_kv * per_token_bytes

    ssm = {
        k.split(".", 1)[1]: v
        for k, v in hparams.items()
        if isinstance(k, str) and k.startswith("ssm.")
    }
    shortconv = {
        k.split(".", 1)[1]: v
        for k, v in hparams.items()
        if isinstance(k, str) and k.startswith("shortconv.")
    }

    recr_state_per_layer, recr_state_type = _compute_recurrent_state(
        n_recr_layers, embedding_length, ssm, shortconv
    )
    recr_bytes = n_recr_layers * recr_state_per_layer

    kv_bytes = attn_bytes + recr_bytes

    if has_kv_array:
        if kv_array_truncated:
            n_kv_heads_report = next((v for v in kv_list if v > 0), 0) or head_count
        else:
            n_kv_heads_report = next(
                (v for v in kv_list if v > 0), kv_list[0] if kv_list else 0
            )
    else:
        n_kv_heads_report = kv_scalar

    formula = (
        f"hybrid: attn={n_attn_layers}L*ctx*{n_kv_heads_report}kv*"
        f"{head_dim}dim + recr={n_recr_layers}L*{recr_state_per_layer}B "
        f"({recr_state_type})"
    )

    return {
        "kv_bytes": kv_bytes,
        "attn_bytes": attn_bytes,
        "recr_bytes": recr_bytes,
        "n_attn_layers": n_attn_layers,
        "n_recr_layers": n_recr_layers,
        "n_ffn_only_layers": n_ffn_only_layers,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads_report,
        "head_dim": head_dim,
        "head_dim_v": head_dim_v,
        "recr_state_per_layer": recr_state_per_layer,
        "recr_state_type": recr_state_type,
        "formula": formula,
        "family": "hybrid",
    }

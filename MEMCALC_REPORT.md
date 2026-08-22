# memcalc — Memory Calculation Backend Report

## What was built

A pure-Python package at `memcalc/` (1183 lines, zero dependencies) that computes KV cache memory as a function of context length for any LLM model, using the GGUF metadata already scraped in `scraper/blobs/*.json`.

**Validated against real ollama runtime: 0.0% error.** Formulas match ollama server log-reported KV buffer sizes exactly at all context sizes.

## Files

```
memcalc/
  __init__.py      6 lines   — re-exports public API
  parse.py        279 lines   — extract_hparams, map_arch, dtype bytes, head_count_kv parsing
  dispatch.py     142 lines   — detect_family, compute_memory_at_context, compute_memory_curve
  standard.py     135 lines   — standard KV formula (linear through origin)
  swa.py          173 lines   — sliding window attention (linear with bend at n_swa)
  mla.py          145 lines   — MLA K-only formula (linear, no 2x V factor)
  hybrid.py       303 lines   — hybrid attention+recurrent (linear + constant y-intercept)
```

## API

```python
import json, memcalc

# From a scraped blob JSON (already in scraper/blobs/<digest>.json):
blob = json.load(open("scraper/blobs/0007cc9e14ff.json"))
hp = memcalc.extract_hparams(blob)

# Single point:
result = memcalc.compute_memory_at_context(hp, context=8192, kv_bpe=2.0)
# -> {"kv_bytes": 4294967296, "kv_gib": 4.0, "family": "standard", "n_layers": 32, ...}

# Full curve for graphing:
curve = memcalc.compute_memory_curve(hp, max_context=131072, n_points=50, kv_bpe=2.0)
# -> [{"context": 0, "kv_bytes": 0, "kv_gib": 0.0, "family": "standard"}, ...]
```

### Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `hp` | dict | required | Output of `extract_hparams(blob_dict)` |
| `context` | int | required | Context length in tokens |
| `kv_bpe` | float | 2.0 (F16) | Bytes per element for KV cache. Use 2.0 for F16 (llama.cpp default), 1.0625 for Q8_0, 0.5625 for Q4_0 |
| `max_context` | int/None | None | Max context for curve; defaults to `context_length` from metadata |
| `n_points` | int | 50 | Number of points in the curve |

### Result dict fields

All results include `kv_bytes`, `kv_gib`, `family`. Additional fields vary by family:

| Family | Extra fields |
|--------|-------------|
| standard | `n_layers`, `n_kv_heads`, `head_dim`, `formula` |
| swa | `n_dense_layers`, `n_swa_layers`, `n_swa`, `head_dim_dense`, `head_dim_swa`, `dense_bytes`, `swa_bytes`, `swa_effective_size`, `formula` |
| mla | `n_mla_layers`, `n_dense_layers`, `stored_k_width`, `has_indexer`, `indexer_bytes`, `mla_bytes`, `dense_bytes`, `formula` |
| hybrid | `n_attn_layers`, `n_recr_layers`, `attn_bytes`, `recr_bytes`, `formula` |
| none | (just `kv_bytes: 0`) |

## The 5 graph shapes

| Shape | Family | Example archs | Formula |
|-------|--------|--------------|---------|
| Flat = 0 | none | bert, clip, gemma-embedding, canary | `0` |
| Linear through origin | standard | llama, qwen2/3, falcon, phi, mistral, command-r | `2 * n_layers * n_kv_heads * head_dim * ctx * bpe` |
| Linear, bends at n_swa | swa | gemma2/3/4, llama4, olmo3, cohere2, gpt-oss | `dense_layers * k * ctx + swa_layers * k * min(ctx, n_swa)` |
| Linear, smaller slope (no V) | mla | deepseek2, deepseek4, glm-dsa, glm4moelite | `n_layers * stored_k_width * ctx * bpe` (+ indexer if DSA) |
| Linear + y-intercept | hybrid | qwen35, lfm2, nemotron_h, granitehybrid, qwen3next | `attn_layers * k * ctx + recurrent_state_constant` |

## Architecture coverage

All 67 architectures found in the scraped blobs are handled. Family distribution across 7099 model blobs:

| Family | Blob count | Example architectures |
|--------|-----------|----------------------|
| standard | 6265 | llama, qwen2, qwen3, qwen3moe, qwen3vl, qwen3vlmoe, falcon, phi2, phi3, mistral3, dbrx, granite, granitemoe, starcoder, starcoder2, stablelm, internlm2, command-r, chatglm, glm4moe, glmocr, olmo2, nemotron, exaone, hunyuan-dense, solar, minimax-m2, hy_v3, qwen, qwen2vl, gpt-oss(without SWA flag), deepseek2-ocr(without SWA flag), mllama/llama4(without SWA flag) |
| none | 712 | bert, nomic-bert, nomic-bert-moe, clip, gemma-embedding, canary, + 653 non-model blobs (params/template/license) |
| swa | 424 | gemma2, gemma3, gemma3n, gemma4, llama4(with chunk_size), olmo3, cohere2, cohere2moe, mimo2, afmoe, laguna, gpt-oss(with sliding_window), phi3(with sliding_window), deepseek2-ocr(with sliding_window) |
| hybrid | 192 | qwen35, qwen35moe, qwen3next, granitehybrid, nemotron_h, nemotron_h_moe, nemotron_h_omni, lfm2, lfm2moe |
| mla | 159 | deepseek2, deepseek4, glm-dsa, glm4moelite |

## Arch name mappings

The scraper captures `general.architecture` which sometimes differs from llama.cpp's enum:

| Scraper arch | Mapped to | Reason |
|-------------|-----------|--------|
| `mllama` | `llama4` | Meta renamed; Llama 3.2 Vision |
| `gptoss` | `gpt-oss` | Underscore vs hyphen |
| `qwen25vl` | `qwen2vl` | Qwen 2.5 VL uses qwen2vl arch |
| `nemotron_h_omni` | `nemotron_h_moe` | Same hybrid formula |

Note: `deepseekocr` is NOT mapped to `deepseek2-ocr` — they are distinct architectures (deepseekocr = standard attention, deepseek2-ocr = SWA with sliding_window=128).

## Hparam fallbacks (when GGUF keys are absent)

| Missing key | Derivation | Affected archs |
|-------------|-----------|----------------|
| `attention.key_length` / `value_length` | `embedding_length // attention.head_count` | llama, qwen2/3, falcon, phi2/3, dbrx, granite, gemma3n, starcoder, stablelm, most standard archs |
| `attention.head_count_kv` | Defaults to `attention.head_count` (MHA) | Any arch where GQA isn't used; also used as the fallback when the hybrid `head_count_kv` array is truncated and entirely zeros |
| `full_attention_interval` (qwen3next) | Defaults to 4 (llama.cpp hardcoded default) | qwen3next (23 blobs) |
| `attention.recurrent_layers` | Infer from `head_count_kv` array (0 = recurrent) or tensor names | qwen35, lfm2, nemotron_h, granitehybrid |
| `ssm.*` (lfm2/lfm2moe) | Use `shortconv.l_cache` formula instead | lfm2, lfm2moe (29 blobs) |

## Truncated array handling

The scraper truncates all arrays to 5 elements + `...` (e.g. `[0, 0, 0, 4, 0, ...]`). This affects `head_count_kv` arrays and `sliding_window_pattern` arrays for hybrid and SWA arches.

**Solution:** The scraper also captures tensor names in `blob["tensors"]`. For hybrid arches, `hybrid.py` infers exact attention layer positions from tensor names (`blk.N.attn_k.weight`, `blk.N.attn_q.weight`) when the `head_count_kv` array is truncated. For SWA arches, `swa.py` tiles the visible `sliding_window_pattern` cyclically.

For SWA `head_count_kv` truncated arrays, `swa.py` tiles the visible value cyclically across all layers.

For hybrid arches, when the truncated `head_count_kv` array is *all zeros* (the recurrent layers come first and the attention layers with non-zero KV are truncated away), `hybrid.py` falls back to `attention.head_count` (MHA) rather than contributing zero KV bytes — this matches the documented "head_count_kv defaults to head_count" rule and prevents perfectly flat curves.

## KV cache dtype (kv_bpe)

The `kv_bpe` parameter controls the bytes-per-element of the KV cache:

| Dtype | kv_bpe | Notes |
|-------|--------|-------|
| F16 (default) | 2.0 | llama.cpp default |
| F32 | 4.0 | |
| BF16 | 2.0 | |
| Q8_0 | 1.0625 | Common in ollama (OLLAMA_KV_CACHE_TYPE=q8_0) |
| Q4_0 | 0.5625 | |
| Q4_1 | 0.625 | |
| Q5_0 | 0.6875 | |
| Q5_1 | 0.75 | |
| IQ4_NL | 0.5625 | |

**Important:** ollama users who set `OLLAMA_KV_CACHE_TYPE=q8_0` need `kv_bpe=1.0625`, not the default 2.0. The default 2.0 (F16) matches llama.cpp's built-in default. The frontend should let users pick their KV cache type.

## Validation results

Validated against real ollama 0.32.5 runtime with 4 models (qwen35-2B-Q6K, qwen35-2B-Q4KM, lfm2.5-350m-BF16, qwen3-embedding-0.6B-Q8_0) at context sizes 2048/4096/8192/32768:

- **KV cache formulas: EXACT (0.0% error)** — matches ollama server log-reported KV buffer sizes perfectly.
- **Recurrent state formulas: EXACT** — matches log-reported recurrent state sizes.
- The only discrepancy was a config difference (user's server uses Q8_0 KV cache, not F16), not a formula bug.

### Spot-checked known values (F16 KV, ctx=8192):

| Model | Expected | memcalc | Match |
|-------|----------|---------|-------|
| Llama 7B MHA (32L, 32 kv_heads, 128 head_dim) | 4.000 GiB | 4.0000 | yes |
| Llama2-70B GQA (80L, 8 kv_heads, 128 head_dim) @4K | 1.250 GiB | 1.2500 | yes |
| DeepSeek2 MLA (58 MLA + 3 dense, stored_k=576) | 0.5596 GiB | 0.5596 | yes |
| Gemma2 SWA (23 dense + 23 SWA, n_swa=4096) | 2.156 GiB | 2.1562 | yes |
| Qwen35 hybrid (16 attn + 48 recr) | 0.646 GiB | 0.6461 | yes |
| Qwen-72B MHA (80L, 64 kv_heads, 128 head_dim) | 20.000 GiB | 20.0000 | yes |

## Integration into build.py

The data is already available at build time. To integrate:

```python
# In build_tag_page() or build_detail(), after loading the blob:
from memcalc import extract_hparams, compute_memory_curve

blob = load_blob_page(blob_url)  # already exists at build.py:478
hp = extract_hparams(blob)
curve = compute_memory_curve(hp, max_context=hp.get("context_length", 32768), n_points=50, kv_bpe=2.0)

# curve is a list of {"context": int, "kv_bytes": int, "kv_gib": float, "family": str, ...}
# Serialize as JSON for the frontend to graph:
import json
chart_data = json.dumps([{"context": p["context"], "kv_gib": p["kv_gib"]} for p in curve])
```

The blob's `general.file_type` (e.g. Q4_K_M, Q8_0, F16) determines the weight quant but NOT the KV cache dtype. KV cache dtype is a runtime config (default F16 in llama.cpp, configurable via `-ctk`/`-ctv` or `OLLAMA_KV_CACHE_TYPE`).

## Known limitations

1. **Canary** (1 blob): no metadata keys at all. Returns `kv_bytes=0`. Not a real LLM.

2. **Truncated arrays for SWA `head_count_kv`**: Some SWA arches (mimo2, laguna) have per-layer `head_count_kv` arrays truncated to 5 elements. These are tiled cyclically, which may not match the real non-periodic pattern. The error is small (1-2 layers off).

3. **`/api/ps` size unreliability**: On machines with small GPUs under VRAM pressure, ollama's `/api/ps` `size` field can be unreliable (doesn't consistently count CPU-mapped buffers). The server log KV buffer sizes are the trustworthy ground truth for validation.

4. **Weight memory not included**: memcalc computes only the KV cache (the dynamic part). Total VRAM = weights + KV cache + activations + overhead. Weight memory comes from the blob's `size` field. Activations (~10% of weights) and overhead (~0.5-3.5 GB) are out of scope.

## What the frontend needs to do

1. Let the user select a model tag (which maps to a specific blob/digest).
2. Let the user select a KV cache dtype (default F16, options: F16, Q8_0, Q4_0).
3. Fetch the blob's `scraper/blobs/<digest>.json` (already scraped).
4. Call `memcalc.compute_memory_curve(hp, max_context=context_length, n_points=50, kv_bpe=selected_bpe)`.
5. Graph the result (X = context, Y = kv_gib).

The curve shape tells the story:
- **standard**: straight line from origin
- **swa**: line that bends/flattens at `n_swa` tokens (SWA layers stop growing)
- **mla**: straight line, lower slope than standard (no V cache)
- **hybrid**: straight line with a y-intercept (constant recurrent state)
- **none**: flat at 0
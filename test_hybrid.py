#!/usr/bin/env python3
"""Test harness: load all HYBRID-family blobs, compute KV at 8192 and 128."""

from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from memcalc.parse import extract_hparams, map_arch
from memcalc.dispatch import detect_family, compute_memory_at_context
from memcalc import hybrid

BLOBS = os.path.join(os.path.dirname(__file__), "scraper", "blobs")
HYBRID = {
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


def load():
    by_arch = {}
    for f in sorted(os.listdir(BLOBS)):
        if not f.endswith(".json"):
            continue
        try:
            b = json.load(open(os.path.join(BLOBS, f)))
        except Exception:
            continue
        md = b.get("metadata") or []
        arch = None
        for e in md:
            if isinstance(e, dict) and e.get("key") == "general.architecture":
                arch = e.get("value")
                break
        if arch in HYBRID:
            hp = extract_hparams(b)
            fam = detect_family(hp)
            by_arch.setdefault(arch, []).append((f, hp, fam))
    return by_arch


def main():
    by_arch = load()
    print("=" * 100)
    print(
        f"{'arch':<16} {'n':>3} {'nL':>4} {'attn':>5} {'recr':>5} {'ffn':>4} {'intv':>5} {'kv@8K(GiB)':>11} {'kv@128(MiB)':>11} {'recr_const(MiB)':>15} type"
    )
    print("-" * 100)
    for arch in sorted(by_arch):
        for f, hp, fam in by_arch[arch]:
            if fam != "hybrid":
                print(f"{arch:<16} WARN family={fam} (not hybrid) file={f}")
                continue
            r8 = compute_memory_at_context(hp, 8192)
            r128 = compute_memory_at_context(hp, 128)
            print(
                f"{arch:<16} {1:>3} {r8['n_layers']:>4} {r8['n_attn_layers']:>5} "
                f"{r8['n_recr_layers']:>5} {r8['n_ffn_only_layers']:>4} "
                f"{str(hp.get('full_attention_interval', '')):>5} "
                f"{r8['kv_bytes'] / 1073741824:>11.4f} "
                f"{r128['kv_bytes'] / 1048576:>11.4f} "
                f"{r8['recr_bytes'] / 1048576:>15.4f} {r8['recr_state_type']}"
            )
    print("=" * 100)
    # Detailed per-arch summary: pick the first blob of each arch.
    print("\nDETAIL (first blob per arch):")
    for arch in sorted(by_arch):
        f, hp, fam = by_arch[arch][0]
        r8 = compute_memory_at_context(hp, 8192)
        r128 = compute_memory_at_context(hp, 128)
        print(f"\n--- {arch}  ({f}) family={fam} ---")
        print(
            f"  block_count={hp.get('block_count')} embedding_length={hp.get('embedding_length')}"
        )
        print(
            f"  head_count={hp.get('attention.head_count')} head_count_kv={hp.get('attention.head_count_kv')!r}"
        )
        print(
            f"  key_length={hp.get('attention.key_length')} value_length={hp.get('attention.value_length')}"
        )
        print(f"  full_attention_interval={hp.get('full_attention_interval')!r}")
        print(f"  feed_forward_length={hp.get('feed_forward_length')!r}")
        ssm = {
            k: v for k, v in hp.items() if isinstance(k, str) and k.startswith("ssm.")
        }
        shortconv = {
            k: v
            for k, v in hp.items()
            if isinstance(k, str) and k.startswith("shortconv.")
        }
        if ssm:
            print(f"  ssm={ssm}")
        if shortconv:
            print(f"  shortconv={shortconv}")
        print(
            f"  n_attn={r8['n_attn_layers']} n_recr={r8['n_recr_layers']} n_ffn_only={r8['n_ffn_only_layers']}"
        )
        print(
            f"  recr_state_per_layer={r8['recr_state_per_layer']}B recr_state_type={r8['recr_state_type']} recr_total={r8['recr_bytes']}B"
        )
        print(
            f"  kv@8192 = {r8['kv_bytes']} B = {r8['kv_bytes'] / 1073741824:.6f} GiB  (attn={r8['attn_bytes']}B recr={r8['recr_bytes']}B)"
        )
        print(
            f"  kv@128  = {r128['kv_bytes']} B = {r128['kv_bytes'] / 1048576:.6f} MiB (attn={r128['attn_bytes']}B recr={r128['recr_bytes']}B)"
        )
        print(
            f"  delta(attn 8K-128) = {(r8['attn_bytes'] - r128['attn_bytes']) / 1048576:.4f} MiB (should scale ~64x)"
        )
        print(f"  formula: {r8['formula']}")


if __name__ == "__main__":
    main()

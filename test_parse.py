from __future__ import annotations

import glob
import json
import os
import unittest

from memcalc.parse import (
    _coerce_value,
    extract_hparams,
    get_hparam,
    get_kv_dtype_bytes,
    map_arch,
    parse_head_count_kv,
    weight_bytes_per_param,
)

BLOB_DIR = os.path.join(os.path.dirname(__file__), "scraper", "blobs")

ALL_ARCHES = [
    "llama",
    "qwen2",
    "qwen3",
    "qwen3moe",
    "qwen3vl",
    "qwen3vlmoe",
    "qwen35",
    "qwen35moe",
    "qwen3next",
    "falcon",
    "phi2",
    "phi3",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3n",
    "gemma4",
    "starcoder",
    "starcoder2",
    "stablelm",
    "internlm2",
    "command-r",
    "chatglm",
    "glm4moe",
    "glm-dsa",
    "glm4moelite",
    "glmocr",
    "deepseek2",
    "deepseek2-ocr",
    "deepseek4",
    "dbrx",
    "olmo2",
    "olmo3",
    "nemotron",
    "nemotron_h",
    "nemotron_h_moe",
    "exaone",
    "granite",
    "granitemoe",
    "granitehybrid",
    "lfm2",
    "lfm2moe",
    "hunyuan-dense",
    "solar",
    "gpt-oss",
    "minimax-m2",
    "mistral3",
    "afmoe",
    "hy_v3",
    "mimo2",
    "cohere2",
    "cohere2moe",
    "laguna",
    "mllama",
    "gptoss",
    "qwen25vl",
    "deepseekocr",
    "bert",
    "nomic-bert",
    "nomic-bert-moe",
    "clip",
    "gemma-embedding",
    "canary",
    "nemotron_h_omni",
]


class TestCoerceValue(unittest.TestCase):
    def test_int(self):
        self.assertEqual(_coerce_value("32"), 32)
        self.assertIsInstance(_coerce_value("32"), int)

    def test_negative_int(self):
        self.assertEqual(_coerce_value("-5"), -5)

    def test_float(self):
        self.assertEqual(_coerce_value("1.25"), 1.25)
        self.assertIsInstance(_coerce_value("1.25"), float)

    def test_scientific_notation(self):
        self.assertAlmostEqual(_coerce_value("1e-06"), 0.000001)
        self.assertEqual(_coerce_value("1e+06"), 1000000.0)
        self.assertEqual(_coerce_value("5e+06"), 5000000.0)

    def test_bool(self):
        self.assertIs(_coerce_value("true"), True)
        self.assertIs(_coerce_value("false"), False)
        self.assertIs(_coerce_value("True"), True)
        self.assertIs(_coerce_value("FALSE"), False)

    def test_list(self):
        self.assertEqual(_coerce_value("[1, 2, 3]"), [1, 2, 3])

    def test_mixed_list(self):
        self.assertEqual(_coerce_value("[1, 2.5, 3]"), [1, 2.5, 3])

    def test_truncated_list(self):
        self.assertEqual(_coerce_value("[1, 2, 3, ...]"), [1, 2, 3])

    def test_bool_list(self):
        self.assertEqual(_coerce_value("[true, false, true]"), [True, False, True])

    def test_truncated_bool_list(self):
        self.assertEqual(
            _coerce_value("[true, true, true, true, false, ...]"),
            [True, True, True, True, False],
        )

    def test_empty_string(self):
        self.assertIsNone(_coerce_value(""))

    def test_empty_string_whitespace(self):
        self.assertIsNone(_coerce_value("   "))

    def test_non_string_passthrough(self):
        self.assertEqual(_coerce_value(5), 5)
        self.assertEqual(_coerce_value(None), None)

    def test_non_numeric_string(self):
        self.assertEqual(_coerce_value("hello"), "hello")

    def test_malformed_array(self):
        # Not a valid JSON array -> returned as-is.
        self.assertEqual(_coerce_value("[1, 2,"), "[1, 2,")


class TestExtractHparams(unittest.TestCase):
    def _blob(self, metadata):
        return {"metadata": metadata, "blob_type": "model"}

    def test_strips_arch_prefix(self):
        blob = self._blob(
            [
                {"key": "general.architecture", "value": "llama"},
                {"key": "llama.block_count", "value": "32"},
                {"key": "llama.attention.head_count", "value": "32"},
                {"key": "llama.attention.head_count_kv", "value": "8"},
            ]
        )
        hp = extract_hparams(blob)
        self.assertEqual(hp.get("block_count"), 32)
        self.assertEqual(hp.get("attention.head_count"), 32)
        self.assertEqual(hp.get("attention.head_count_kv"), 8)
        self.assertNotIn("llama.block_count", hp)

    def test_keeps_general_keys_verbatim(self):
        blob = self._blob(
            [
                {"key": "general.architecture", "value": "qwen2"},
                {"key": "general.name", "value": "Qwen2"},
                {"key": "general.file_type", "value": "F16"},
            ]
        )
        hp = extract_hparams(blob)
        self.assertEqual(hp.get("general.architecture"), "qwen2")
        self.assertEqual(hp.get("general.name"), "Qwen2")
        self.assertEqual(hp.get("general.file_type"), "F16")

    def test_empty_metadata(self):
        self.assertEqual(extract_hparams({}), {})
        self.assertEqual(extract_hparams({"metadata": []}), {})
        self.assertEqual(extract_hparams({"metadata": None}), {})
        self.assertEqual(extract_hparams({"metadata": "x"}), {})

    def test_flat_dict(self):
        blob = self._blob(
            [
                {"key": "general.architecture", "value": "llama"},
                {"key": "llama.block_count", "value": "4"},
                {"key": "llama.attention.head_count", "value": "4"},
            ]
        )
        hp = extract_hparams(blob)
        self.assertIsInstance(hp, dict)
        for k in hp:
            self.assertIsInstance(k, str)

    def test_substring_arch_prefix(self):
        # "qwen2" must not be stripped off "qwen25vl.block_count".
        blob = self._blob(
            [
                {"key": "general.architecture", "value": "qwen25vl"},
                {"key": "qwen25vl.block_count", "value": "28"},
                {"key": "qwen25vl.attention.head_count", "value": "16"},
            ]
        )
        hp = extract_hparams(blob)
        self.assertEqual(hp.get("block_count"), 28)
        self.assertEqual(hp.get("attention.head_count"), 16)
        self.assertNotIn("qwen25vl.block_count", hp)
        # No leftover "5vl." prefix from a wrong strip.
        self.assertNotIn("5vl.block_count", hp)

    def test_dashed_arch_prefix(self):
        blob = self._blob(
            [
                {"key": "general.architecture", "value": "deepseek2-ocr"},
                {"key": "deepseek2-ocr.attention.head_count", "value": "16"},
            ]
        )
        hp = extract_hparams(blob)
        self.assertEqual(hp.get("attention.head_count"), 16)
        self.assertNotIn("deepseek2-ocr.attention.head_count", hp)

    def test_arch_not_stripped_before_declaration(self):
        # Keys appearing before general.architecture keep their full key.
        blob = self._blob(
            [
                {"key": "llama.block_count", "value": "8"},
                {"key": "general.architecture", "value": "llama"},
                {"key": "llama.attention.head_count", "value": "32"},
            ]
        )
        hp = extract_hparams(blob)
        self.assertEqual(hp.get("llama.block_count"), 8)
        self.assertEqual(hp.get("attention.head_count"), 32)

    def test_other_arch_keys_kept(self):
        blob = self._blob(
            [
                {"key": "general.architecture", "value": "llama"},
                {"key": "clip.vision.head_count", "value": "16"},
            ]
        )
        hp = extract_hparams(blob)
        self.assertEqual(hp.get("clip.vision.head_count"), 16)


class TestMapArch(unittest.TestCase):
    def test_known_aliases(self):
        self.assertEqual(map_arch("mllama"), "llama4")
        self.assertEqual(map_arch("gptoss"), "gpt-oss")
        self.assertEqual(map_arch("qwen25vl"), "qwen2vl")
        self.assertEqual(map_arch("deepseekocr"), "deepseek2-ocr")

    def test_all_blob_arches(self):
        # Every arch observed in real blobs should map (identity if no alias).
        for arch in ALL_ARCHES:
            mapped = map_arch(arch)
            self.assertIsInstance(mapped, str, arch)
            self.assertTrue(mapped, arch)

    def test_empty_string(self):
        self.assertEqual(map_arch(""), "")

    def test_non_string(self):
        self.assertIsNone(map_arch(None))

    def test_unknown_passthrough(self):
        self.assertEqual(map_arch("totally-fake-arch"), "totally-fake-arch")


class TestParseHeadCountKV(unittest.TestCase):
    def test_scalar_int(self):
        self.assertEqual(parse_head_count_kv("32"), 32)

    def test_scalar_int_str(self):
        self.assertEqual(parse_head_count_kv(32), 32)

    def test_scalar_float(self):
        self.assertEqual(parse_head_count_kv("32.0"), 32)
        self.assertEqual(parse_head_count_kv(32.0), 32)

    def test_array(self):
        self.assertEqual(
            parse_head_count_kv("[16, 16, 16, 16, 16]"),
            [16, 16, 16, 16, 16],
        )

    def test_truncated_array(self):
        self.assertEqual(
            parse_head_count_kv("[0, 0, 0, 4, 0, ...]"),
            [0, 0, 0, 4, 0],
        )

    def test_list_passthrough(self):
        self.assertEqual(parse_head_count_kv([1, 2, 3]), [1, 2, 3])

    def test_empty_string(self):
        self.assertIsNone(parse_head_count_kv(""))

    def test_none(self):
        self.assertIsNone(parse_head_count_kv(None))

    def test_garbage(self):
        self.assertIsNone(parse_head_count_kv("abc"))

    def test_malformed_array(self):
        self.assertIsNone(parse_head_count_kv("[1, 2,"))

    def test_non_integer_scalar(self):
        self.assertIsNone(parse_head_count_kv("3.5"))


class TestGetKVDtypeBytes(unittest.TestCase):
    def test_floats(self):
        self.assertEqual(get_kv_dtype_bytes("F16"), 2.0)
        self.assertEqual(get_kv_dtype_bytes("F32"), 4.0)
        self.assertEqual(get_kv_dtype_bytes("BF16"), 2.0)

    def test_quants(self):
        self.assertEqual(get_kv_dtype_bytes("Q8_0"), 1.0625)
        self.assertEqual(get_kv_dtype_bytes("Q4_0"), 0.5625)
        self.assertEqual(get_kv_dtype_bytes("Q4_1"), 0.625)

    def test_q6_k_present(self):
        # Q6_K is in the table.
        self.assertEqual(get_kv_dtype_bytes("Q6_K"), 0.8203125)

    def test_unknown_default(self):
        self.assertEqual(get_kv_dtype_bytes("NOT_A_TYPE"), 2.0)

    def test_non_string(self):
        self.assertEqual(get_kv_dtype_bytes(None), 2.0)

    def test_weight_alias(self):
        self.assertEqual(weight_bytes_per_param("F16"), 2.0)


class TestGetHparam(unittest.TestCase):
    def test_stripped_key(self):
        hp = {"attention.head_count": 32}
        self.assertEqual(get_hparam(hp, "attention.head_count", "llama"), 32)

    def test_raw_key_fallback(self):
        hp = {"llama.attention.head_count": 32}
        self.assertEqual(get_hparam(hp, "attention.head_count", "llama"), 32)

    def test_default(self):
        self.assertIsNone(get_hparam({}, "attention.head_count", "llama"))
        self.assertEqual(get_hparam({}, "attention.head_count", "llama", 8), 8)


class TestAllBlobs(unittest.TestCase):
    """Parse every blob file to ensure no crashes and stable arch mapping."""

    @classmethod
    def setUpClass(cls):
        cls.blobs = []
        for path in glob.glob(os.path.join(BLOB_DIR, "*.json")):
            with open(path) as fh:
                cls.blobs.append((os.path.basename(path), json.load(fh)))
        if not cls.blobs:
            raise unittest.SkipTest("no blobs found at " + BLOB_DIR)

    def test_all_blobs_parse(self):
        crashes = []
        for name, blob in self.blobs:
            try:
                hp = extract_hparams(blob)
                arch = hp.get("general.architecture", "")
                map_arch(arch)
            except Exception as exc:  # noqa: BLE001 - we want all failures
                crashes.append((name, repr(exc)))
                continue
        self.assertFalse(crashes, "crashes: " + json.dumps(crashes[:5]))

    def test_all_arches_mapped(self):
        mapped = {}
        for _, blob in self.blobs:
            hp = extract_hparams(blob)
            arch = hp.get("general.architecture")
            if isinstance(arch, str) and arch:
                mapped[map_arch(arch)] = True
        # All real arch names should produce a non-empty string.
        for arch in mapped:
            self.assertIsInstance(arch, str)
            self.assertTrue(arch)

    def test_head_count_kv_all(self):
        bad = []
        for name, blob in self.blobs:
            hp = extract_hparams(blob)
            raw = hp.get("attention.head_count_kv")
            if raw is None:
                continue
            parsed = parse_head_count_kv(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list):
                for v in parsed:
                    if not isinstance(v, int):
                        bad.append((name, raw, parsed))
            elif parsed is not None and not isinstance(parsed, int):
                bad.append((name, raw, parsed))
        self.assertFalse(bad, "bad head_count_kv: " + json.dumps(bad[:5]))


if __name__ == "__main__":
    unittest.main()

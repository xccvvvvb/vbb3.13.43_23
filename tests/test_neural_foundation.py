"""Dependency-free tests for the neural model configuration and tokenizer."""

import unittest

from neural.config import ModelConfig, PRESETS
from neural.tokenizer import ByteTokenizer


class NeuralFoundationTests(unittest.TestCase):
    def test_unicode_tokenizer_roundtrip(self):
        tokenizer = ByteTokenizer()
        text = "IA local: razón, código y privacidad 🧠"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text, bos=True, eos=True)), text)

    def test_all_large_presets_are_structurally_valid(self):
        for config in PRESETS.values():
            ModelConfig(**config.to_dict())
            self.assertEqual(config.dim % config.n_heads, 0)
            self.assertEqual(config.n_heads % config.n_kv_heads, 0)

    def test_large_parameter_scales_are_real(self):
        self.assertTrue(1_200_000_000 < PRESETS["1.3b"].parameter_estimate() < 1_400_000_000)
        self.assertTrue(6_500_000_000 < PRESETS["7b"].parameter_estimate() < 7_200_000_000)
        self.assertTrue(12_500_000_000 < PRESETS["13b"].parameter_estimate() < 13_500_000_000)

    def test_invalid_attention_shape_fails(self):
        with self.assertRaises(ValueError):
            ModelConfig(dim=1000, n_heads=12)


if __name__ == "__main__":
    unittest.main()

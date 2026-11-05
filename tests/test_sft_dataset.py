"""Tests for the Soup dataset and the SFT formatting (no torch required)."""

import json
import unittest
from pathlib import Path

from neural.sft_format import IGNORE_INDEX, SYSTEM, encode_dialogue, load_dataset
from neural.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "soup" / "data" / "train.jsonl"


class SoupDatasetTests(unittest.TestCase):
    def test_dataset_exists_and_is_valid_jsonl(self):
        self.assertTrue(DATASET.is_file(), "Ejecuta python soup/build_dataset.py")
        rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 30)
        for row in rows:
            self.assertTrue(row["instruction"].strip())
            self.assertTrue(row["output"].strip())
            self.assertIn("input", row)

    def test_dataset_is_unique_and_spanish(self):
        rows = load_dataset(DATASET)
        self.assertEqual(len({row["instruction"] for row in rows}), len(rows))
        markers = {"¿", "á", "é", "í", "ó", "ú", "ñ"}
        for row in rows:
            text = row["instruction"] + row["output"]
            self.assertTrue(
                any(marker in text.lower() for marker in markers),
                f"Texto sin marcas del español: {row['instruction']}",
            )

    def test_prompt_tokens_are_masked_and_answer_is_supervised(self):
        tokens, labels = encode_dialogue(
            {"instruction": "¿Quién eres?", "input": "", "output": "Soy Hyperion."}, max_length=512
        )
        self.assertEqual(len(tokens), len(labels))
        supervised = [tag for tag in labels if tag != IGNORE_INDEX]
        self.assertTrue(supervised)
        decoded = ByteTokenizer().decode(supervised)
        self.assertIn("Soy Hyperion.", decoded)
        self.assertIn("<|fin|>", decoded)
        self.assertNotIn(SYSTEM, decoded)

    def test_max_length_truncates_both_sequences(self):
        tokens, labels = encode_dialogue(
            {"instruction": "x" * 5000, "input": "", "output": "y" * 5000}, max_length=128
        )
        self.assertEqual(len(tokens), 128)
        self.assertEqual(len(labels), 128)


if __name__ == "__main__":
    unittest.main()

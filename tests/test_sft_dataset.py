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

    def test_labels_are_shifted_targets(self):
        """El modelo no desplaza: labels[i] debe ser el token siguiente."""
        tokens, labels = encode_dialogue(
            {"instruction": "¿Qué es Soup?", "input": "", "output": "Una CLI."}, max_length=512
        )
        for index, label in enumerate(labels[:-1]):
            if label != IGNORE_INDEX:
                self.assertEqual(label, tokens[index + 1])
        self.assertEqual(labels[0], IGNORE_INDEX)  # el prompt nunca se supervisa
        self.assertNotEqual(labels[-1], IGNORE_INDEX)  # el último byte también se aprende

    def test_first_answer_token_is_reachable(self):
        """La posición previa a la respuesta debe estar supervisada."""
        tokens, labels = encode_dialogue(
            {"instruction": "Hola", "input": "", "output": "Hola."}, max_length=512
        )
        start = len(tokens) - len(ByteTokenizer().encode("Hola.<|fin|>")) + 1
        self.assertNotEqual(labels[start - 1], IGNORE_INDEX)

    def test_max_length_truncates_both_sequences(self):
        tokens, labels = encode_dialogue(
            {"instruction": "x" * 5000, "input": "", "output": "y" * 5000}, max_length=128
        )
        self.assertEqual(len(tokens), 127)  # el input son tokens[i] -> token[i+1]
        self.assertEqual(len(labels), 127)


if __name__ == "__main__":
    unittest.main()

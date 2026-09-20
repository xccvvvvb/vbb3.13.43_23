"""Tests for the dependency-free Hyperion engine. Run: python -m unittest -v"""

import unittest
from core.engine import HyperionEngine, SafeMath


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = HyperionEngine()

    def test_safe_math(self):
        self.assertEqual(SafeMath.evaluate("2 + 3 * 4"), 14)
        self.assertEqual(SafeMath.evaluate("sqrt(81)"), 9)

    def test_math_rejects_code(self):
        with self.assertRaises(ValueError):
            SafeMath.evaluate("__import__('os').system('id')")

    def test_privacy_answer(self):
        result = self.engine.respond("¿Mis datos salen a Internet?")
        self.assertIn("local", result["answer"].lower())
        self.assertEqual(result["mode"], "private-local")

    def test_realtime_is_honest(self):
        result = self.engine.respond("Busca el clima de hoy en internet")
        self.assertIn("ERROR DE RED", result["answer"])

    def test_retrieval(self):
        result = self.engine.respond("¿Qué es la atención de un transformer?")
        self.assertIn("Transformer", result["answer"])
        self.assertGreater(result["confidence"], 0)

    def test_volatile_learning(self):
        chunks = self.engine.learn("Proyecto Boreal", "El código secreto del Proyecto Boreal es AZUL-739. " * 30)
        self.assertGreaterEqual(chunks, 1)
        result = self.engine.respond("¿Cuál es el código secreto del Proyecto Boreal?")
        self.assertIn("AZUL-739", result["answer"])
        self.engine.reset_learned()
        self.assertEqual(self.engine.learned_count, 0)


if __name__ == "__main__":
    unittest.main()

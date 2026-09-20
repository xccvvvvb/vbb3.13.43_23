"""Dependency-free, auditable local language engine for Hyperion."""

from __future__ import annotations

import ast
import hashlib
import math
import operator
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .knowledge import KNOWLEDGE
from .knowledge_extended import EXTENDED_KNOWLEDGE


TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9+#.-]+", re.IGNORECASE)
STOPWORDS = {
    "a", "al", "algo", "como", "con", "cual", "de", "del", "el", "en", "es",
    "esta", "esto", "la", "las", "lo", "los", "me", "mi", "para", "por", "que",
    "se", "si", "su", "un", "una", "y", "ya",
}


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().strip()


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(normalize(text)) if t not in STOPWORDS and len(t) > 1]


def sparse_embedding(text: str, dimensions: int = 384) -> dict[int, float]:
    """Feature-hashed word/character n-gram embedding with L2 normalization."""
    tokens = tokenize(text)
    features = tokens + [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    compact = " ".join(tokens)
    features += [f"#{compact[i:i + 3]}" for i in range(max(0, len(compact) - 2))]
    vector: dict[int, float] = {}
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        index = raw % dimensions
        sign = 1.0 if raw & (1 << 63) else -1.0
        vector[index] = vector.get(index, 0.0) + sign
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    return {index: value / norm for index, value in vector.items()}


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


class SafeMath:
    """Small AST interpreter: arithmetic only, never eval()."""

    BINARY = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
    }
    UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    FUNCTIONS = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "abs": abs, "round": round,
    }
    CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

    @classmethod
    def evaluate(cls, expression: str) -> int | float:
        if len(expression) > 160:
            raise ValueError("La expresión supera el límite de 160 caracteres.")
        try:
            tree = ast.parse(expression.replace("^", "**"), mode="eval")
        except SyntaxError as exc:
            raise ValueError("Sintaxis matemática no válida.") from exc
        value = cls._visit(tree.body, depth=0)
        if not math.isfinite(float(value)):
            raise ValueError("El resultado no es finito.")
        return value

    @classmethod
    def _visit(cls, node: ast.AST, depth: int) -> int | float:
        if depth > 20:
            raise ValueError("Expresión demasiado profunda.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 1e100:
                raise ValueError("Número fuera del límite permitido.")
            return node.value
        if isinstance(node, ast.Name) and node.id in cls.CONSTANTS:
            return cls.CONSTANTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in cls.BINARY:
            left = cls._visit(node.left, depth + 1)
            right = cls._visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponente fuera del límite permitido.")
            return cls.BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls.UNARY:
            return cls.UNARY[type(node.op)](cls._visit(node.operand, depth + 1))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in cls.FUNCTIONS:
            if node.keywords or len(node.args) > 2:
                raise ValueError("Argumentos no permitidos.")
            return cls.FUNCTIONS[node.func.id](*(cls._visit(arg, depth + 1) for arg in node.args))
        raise ValueError("Solo se permiten números, operadores y funciones matemáticas seguras.")


@dataclass(frozen=True)
class IndexedDocument:
    title: str
    text: str
    tags: tuple[str, ...]
    tokens: tuple[str, ...]
    frequencies: Counter[str]
    embedding: dict[int, float]


class HyperionEngine:
    """Hybrid symbolic/retrieval engine with no third-party runtime dependency."""

    VERSION = "2.0.0"

    def __init__(self) -> None:
        self.base_count = len(KNOWLEDGE) + len(EXTENDED_KNOWLEDGE)
        self.documents = [self._index(item) for item in KNOWLEDGE + EXTENDED_KNOWLEDGE]
        self._rebuild_statistics()

    def _rebuild_statistics(self) -> None:
        self.document_frequency = Counter(
            token for document in self.documents for token in set(document.tokens)
        )
        self.average_length = sum(len(d.tokens) for d in self.documents) / max(1, len(self.documents))

    def learn(self, title: str, content: str) -> int:
        """Index user-provided text in volatile memory and return the chunk count."""
        title, content = title.strip(), content.strip()
        if not title or not content:
            raise ValueError("Título y contenido son obligatorios.")
        if len(title) > 120 or len(content) > 500_000:
            raise ValueError("El documento supera el límite local permitido.")
        words = content.split()
        chunks = [" ".join(words[index:index + 220]) for index in range(0, len(words), 180)]
        if not chunks:
            raise ValueError("El documento no contiene texto indexable.")
        for index, chunk in enumerate(chunks):
            item = {
                "title": title if len(chunks) == 1 else f"{title} · {index + 1}/{len(chunks)}",
                "tags": tokenize(title)[:8] + ["memoria", "usuario"],
                "text": chunk,
            }
            self.documents.append(self._index(item))
        self._rebuild_statistics()
        return len(chunks)

    def reset_learned(self) -> None:
        self.documents = self.documents[:self.base_count]
        self._rebuild_statistics()

    @property
    def learned_count(self) -> int:
        return max(0, len(self.documents) - self.base_count)

    @staticmethod
    def _index(item: dict[str, Any]) -> IndexedDocument:
        searchable = f"{item['title']} {' '.join(item['tags'])} {item['text']}"
        tokens = tuple(tokenize(searchable))
        return IndexedDocument(
            title=item["title"], text=item["text"], tags=tuple(item["tags"]),
            tokens=tokens, frequencies=Counter(tokens), embedding=sparse_embedding(searchable),
        )

    def _bm25(self, query: list[str], document: IndexedDocument) -> float:
        score = 0.0
        count = len(self.documents)
        for token in query:
            frequency = document.frequencies.get(token, 0)
            if not frequency:
                continue
            doc_frequency = self.document_frequency[token]
            idf = math.log(1 + (count - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(document.tokens) / self.average_length)
            score += idf * frequency * 2.5 / denominator
        return score

    def search(self, query: str, limit: int = 3) -> list[tuple[IndexedDocument, float]]:
        query_tokens = tokenize(query)
        query_embedding = sparse_embedding(query)
        ranked = []
        for document in self.documents:
            lexical = self._bm25(query_tokens, document)
            semantic = max(0.0, cosine(query_embedding, document.embedding))
            tag_bonus = sum(0.35 for token in query_tokens if token in document.tags)
            ranked.append((document, lexical + semantic * 2.2 + tag_bonus))
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]

    @staticmethod
    def _extract_expression(message: str) -> str | None:
        lowered = normalize(message)
        prefixes = ("calcula ", "calcular ", "cuánto es ", "cuanto es ", "resuelve ")
        for prefix in prefixes:
            if lowered.startswith(prefix):
                candidate = lowered[len(prefix):].strip(" ?=.")
                if re.fullmatch(r"[0-9a-z_+*/%()., ^-]+", candidate):
                    return candidate.replace(",", ".")
        if re.fullmatch(r"[0-9+*/%(). ^-]+", lowered):
            return lowered
        return None

    def respond(self, message: str, style: str = "balanced") -> dict[str, Any]:
        started = time.perf_counter()
        clean = normalize(message)
        if not clean or len(clean) > 4000:
            raise ValueError("El mensaje debe contener entre 1 y 4000 caracteres.")

        realtime = ("internet", "web", "tiempo real", "noticias de hoy", "clima de hoy")
        if any(term in clean for term in realtime):
            return self._result(
                "[ERROR DE RED: Sistema operando en modo local aislado. No existe conexión a servicios web o APIs externas.]",
                1.0, "network-guard", started, [],
            )

        expression = self._extract_expression(message)
        if expression:
            try:
                value = SafeMath.evaluate(expression)
                formatted = f"{value:.12g}" if isinstance(value, float) else str(value)
                return self._result(
                    f"**Resultado:** `{expression} = {formatted}`\n\nEvaluado localmente con un intérprete AST seguro; no se ejecutó código arbitrario.",
                    1.0, "symbolic-math", started, ["Motor matemático"],
                )
            except (ValueError, ZeroDivisionError, OverflowError) as exc:
                return self._result(f"**Error matemático:** {exc}", 0.95, "symbolic-math", started, [])

        if any(term in clean for term in ("hola", "buenas", "hey", "quién eres", "quien eres")):
            answer = (
                "Soy **Hyperion**, una IA neuro-simbólica ejecutada íntegramente en este servidor local. "
                "No utilizo OpenAI, Gemini, Claude ni ninguna API de inferencia externa.\n\n"
                "Puedo razonar con mi base local, explicar conceptos técnicos, evaluar matemáticas seguras y analizar texto."
            )
            return self._result(answer, 0.99, "conversation", started, ["Arquitectura local"])

        ranked = self.search(message)
        best, score = ranked[0]
        confidence = min(0.96, max(0.18, score / 5.5))
        if score < 0.45:
            answer = (
                "No encuentro evidencia suficiente en mi conocimiento local para responder con precisión. "
                "Puedo analizarlo si aportas más contexto o datos dentro del mensaje.\n\n"
                "**Límite actual:** no inventaré información ni consultaré servicios externos."
            )
            return self._result(answer, confidence, "local-fallback", started, [])

        sources = [document.title for document, item_score in ranked if item_score > 0.35]
        if style == "concise":
            answer = best.text
        else:
            support = ranked[1][0] if ranked[1][1] > 0.8 else None
            answer = f"**{best.title}**\n\n{best.text}"
            if support and support.title != best.title:
                answer += f"\n\n**Contexto relacionado — {support.title}:** {support.text}"
            if style == "technical":
                answer += "\n\n`Inferencia: ranking híbrido BM25 + similitud coseno sobre embeddings locales.`"
        return self._result(answer, confidence, "hybrid-retrieval", started, sources)

    @staticmethod
    def _result(answer: str, confidence: float, intent: str, started: float, sources: list[str]) -> dict[str, Any]:
        return {
            "answer": answer,
            "confidence": round(confidence, 3),
            "intent": intent,
            "sources": sources,
            "latency_ms": max(1, round((time.perf_counter() - started) * 1000)),
            "mode": "private-local",
        }

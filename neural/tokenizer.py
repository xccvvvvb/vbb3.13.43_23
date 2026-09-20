"""Reversible UTF-8 byte tokenizer with explicit control tokens; no external tokenizer."""

from __future__ import annotations


class ByteTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    OFFSET = 3
    vocab_size = 259

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        tokens = [byte + self.OFFSET for byte in text.encode("utf-8", errors="strict")]
        if bos:
            tokens.insert(0, self.BOS)
        if eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        data = bytes(token - self.OFFSET for token in tokens if self.OFFSET <= token < self.vocab_size)
        return data.decode("utf-8", errors="replace")

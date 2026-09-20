"""Hyperion decoder-only Transformer: RMSNorm, RoPE, GQA, SwiGLU, tied embeddings."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.type_as(x) * self.weight


def precompute_rope(head_dim: int, length: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    return angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., ::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        query = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim)
        key = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
        value = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        repeats = self.n_heads // self.n_kv_heads
        if repeats > 1:
            key = key.repeat_interleave(repeats, dim=2)
            value = value.repeat_interleave(repeats, dim=2)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.out_proj(attended.transpose(1, 2).contiguous().view(batch, length, -1))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = Attention(config)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), cos, sin)
        return x + self.feed_forward(self.ffn_norm(x))


class HyperionTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self._initialize)
        for name, parameter in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.size(1) > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured context length")
        x = self.dropout(self.embedding(tokens))
        cos, sin = precompute_rope(
            self.config.dim // self.config.n_heads, tokens.size(1), self.config.rope_theta, tokens.device
        )
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_p: float = 0.9) -> torch.Tensor:
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature and top_p are outside valid ranges")
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.max_seq_len:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            next_logits[:, :3] = -torch.inf
            probabilities = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs >= top_p
            sorted_probs.masked_fill_(remove, 0.0)
            sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
            sampled = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices.gather(-1, sampled)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict:
        return {"architecture": "HyperionTransformer", "config": asdict(self.config), "parameters": self.parameter_count()}

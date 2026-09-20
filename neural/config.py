"""Model configurations for Hyperion Transformer variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 259
    max_seq_len: int = 2048
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    hidden_dim: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.vocab_size < 259 or self.max_seq_len < 8:
            raise ValueError("invalid vocabulary or context length")

    def to_dict(self) -> dict:
        return asdict(self)

    def parameter_estimate(self) -> int:
        head_dim = self.dim // self.n_heads
        kv_dim = self.n_kv_heads * head_dim
        attention = 2 * self.dim * self.dim + 2 * self.dim * kv_dim
        feed_forward = 3 * self.dim * self.hidden_dim
        norms = 2 * self.dim
        return self.vocab_size * self.dim + self.n_layers * (attention + feed_forward + norms) + self.dim


PRESETS = {
    "tiny": ModelConfig(dim=384, n_layers=6, n_heads=6, n_kv_heads=2, hidden_dim=1024, max_seq_len=512),
    "1.3b": ModelConfig(dim=2048, n_layers=24, n_heads=16, n_kv_heads=4, hidden_dim=7168, max_seq_len=4096),
    "7b": ModelConfig(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, hidden_dim=13824, max_seq_len=8192, rope_theta=500_000.0),
    "13b": ModelConfig(dim=5120, n_layers=40, n_heads=40, n_kv_heads=8, hidden_dim=16896, max_seq_len=8192, rope_theta=500_000.0),
}

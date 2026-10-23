"""Decoder-only transformer shapes used by the roofline model.

Shapes come from the public Hugging Face configs for each model. Only the
quantities that matter for FLOP and byte counting are kept.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied_embeddings: bool = False

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def attn_params_per_layer(self) -> int:
        q = self.hidden * self.heads * self.head_dim
        o = self.heads * self.head_dim * self.hidden
        kv = 2 * self.hidden * self.kv_dim
        return q + o + kv

    @property
    def mlp_params_per_layer(self) -> int:
        # SwiGLU: gate, up, down projections
        return 3 * self.hidden * self.intermediate

    @property
    def layer_params(self) -> int:
        return self.layers * (self.attn_params_per_layer + self.mlp_params_per_layer)

    @property
    def lm_head_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def matmul_params(self) -> int:
        """Weights read and multiplied on every decode step (layers + LM head)."""
        return self.layer_params + self.lm_head_params

    @property
    def total_params(self) -> int:
        embedding = self.vocab * self.hidden
        return self.layer_params + embedding + (0 if self.tied_embeddings else self.lm_head_params)


MODELS: dict[str, ModelConfig] = {
    m.name: m
    for m in [
        ModelConfig("tinyllama-1.1b", hidden=2048, layers=22, heads=32, kv_heads=4,
                    head_dim=64, intermediate=5632, vocab=32000),
        ModelConfig("llama-3.2-1b", hidden=2048, layers=16, heads=32, kv_heads=8,
                    head_dim=64, intermediate=8192, vocab=128256, tied_embeddings=True),
        ModelConfig("llama-3.2-3b", hidden=3072, layers=28, heads=24, kv_heads=8,
                    head_dim=128, intermediate=8192, vocab=128256, tied_embeddings=True),
        ModelConfig("llama-3.1-8b", hidden=4096, layers=32, heads=32, kv_heads=8,
                    head_dim=128, intermediate=14336, vocab=128256),
    ]
}

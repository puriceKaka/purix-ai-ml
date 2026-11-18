"""Small decoder-only Transformer language model implemented with PyTorch."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        if self.vocab_size < 4 or self.block_size < 2 or self.d_model < 1:
            raise ValueError("vocab_size >= 4, block_size >= 2 and d_model >= 1 are required")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by positive n_heads")
        if self.n_layers < 1 or not 0 <= self.dropout < 1:
            raise ValueError("n_layers must be positive and dropout in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(config.d_model, config.n_heads,
                                                dropout=config.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model), nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model), nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     attn_mask=causal_mask, need_weights=False)
        x = x + attended
        return x + self.feed_forward(self.norm2(x))


class LanguageModel(nn.Module):
    """GPT-style causal LM; logits at each position predict its next token."""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        _, length = input_ids.shape
        if length > self.config.block_size:
            raise ValueError(f"Sequence length {length} exceeds block_size {self.config.block_size}")
        positions = torch.arange(length, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.final_norm(x))

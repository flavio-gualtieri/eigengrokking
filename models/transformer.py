# models/transformer.py

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.registry import ACTIVATIONS


class SelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, device: torch.device):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False, device=device)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class MLPBlock(nn.Module):

    def __init__(self, d_model: int, d_mlp: int, activation_fn: str, device: torch.device):
        super().__init__()
        activation_cls = ACTIVATIONS[activation_fn]
        self.fc1 = nn.Linear(d_model, d_mlp, device=device)
        self.act = activation_cls()
        self.fc2 = nn.Linear(d_mlp, d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_mlp: int, activation_fn: str, device: torch.device):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, device=device)
        self.attn = SelfAttention(d_model, num_heads, device)
        self.ln2 = nn.LayerNorm(d_model, device=device)
        self.mlp = MLPBlock(d_model, d_mlp, activation_fn, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """
    Small decoder-only transformer for token-sequence classification tasks
    (e.g. modular arithmetic). Reads off logits at the final sequence position.
    """

    def __init__(
            self,
            vocab_size: int,
            output_dim: int,
            n_ctx: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            device: torch.device,
            d_mlp: int | None = None,
            activation_fn: str = "ReLU",
            init_scale: float = 1.0,
    ):
        super().__init__()
        d_mlp = d_mlp or 4 * d_model

        self.token_embed = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_embed = nn.Embedding(n_ctx, d_model, device=device)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_mlp, activation_fn, device)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, device=device)
        self.unembed = nn.Linear(d_model, output_dim, bias=False, device=device)

        with torch.no_grad():
            for p in self.parameters():
                p.data = init_scale * p.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared training/eval code paths cast inputs to float; token ids need to be long.
        x = x.long()
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.token_embed(x) + self.pos_embed(positions)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)

        logits = self.unembed(h)
        return logits[:, -1, :]

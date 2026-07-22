"""
Plain decoder-only transformer, used as the baseline comparison for
HierarchicalTransformer.

This module has no file I/O; it is safe to import
on its own.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mask(seq):
    """Additive causal mask: (B, T, T) with -1e9 above the diagonal.

    Not currently called anywhere in this script.
    """
    B, T = seq.shape
    device = seq.device
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    additive = causal.float().masked_fill(causal, float("-1e9"))
    return additive.unsqueeze(0).expand(B, T, T)


class BatchedAttentionLayer(nn.Module):
    """Pre-norm transformer block: causal self-attention + FFN.
    """

    def __init__(self, d_model: int, d_ffn: int, nhead: int, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

        self.W_a = nn.Linear(d_model, d_ffn, bias=bias)
        self.W_b = nn.Linear(d_ffn, d_model, bias=bias)

        self.dropout = dropout
        self.relu = nn.ReLU()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H = self.nhead
        dH = self.d_head

        residual_input_attn = x
        norm_x_attn = self.ln1(x)

        q = self.W_q(norm_x_attn)
        k = self.W_k(norm_x_attn)
        v = self.W_v(norm_x_attn)

        q = q.view(B, T, H, dH).transpose(1, 2)
        k = k.view(B, T, H, dH).transpose(1, 2)
        v = v.view(B, T, H, dH).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        x = residual_input_attn + attn_output

        residual_input_ffn = x
        norm_x_ffn = self.ln2(x)

        ffn_output = self.W_a(norm_x_ffn)
        ffn_output = self.relu(ffn_output)
        ffn_output = F.dropout(ffn_output, p=self.dropout, training=self.training)
        ffn_output = self.W_b(ffn_output)
        ffn_output = F.dropout(ffn_output, p=self.dropout, training=self.training)

        x = residual_input_ffn + ffn_output

        return x


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        return self.dropout(x)


class BatchedAttention(nn.Module):
    """The baseline (no dynamic chunking) comparison model
    for HierarchicalTransformer."""

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dropout,
        d_ffn=768,
        vocab_size=8018,
        bias=True,
        max_len=256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                BatchedAttentionLayer(d_model, d_ffn, nhead, dropout, bias)
            )
        self.unembedding = nn.Linear(d_model, vocab_size, bias=False)
        self.unembedding.weight = self.embedding.weight
        self.max_len = max_len
        self.vocab_size = vocab_size

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.unembedding(x)
        return x

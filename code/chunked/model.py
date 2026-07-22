"""
Hierarchical Transformer

Notation used:
    B - batch size
    T - sequence length (tokens)
    K - number of chunks, K = min(max_chunks, T)
    D - embedding dimension
    L - padded chunk length
    V - vocabulary size
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mask(seq):
    """Additive causal mask: (B, T, T) with -1e9 above the diagonal."""
    B, T = seq.shape
    device = seq.device
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    additive = causal.float() * float("-1e9")
    return additive.unsqueeze(0).expand(B, T, T)


class BoundaryPredictor(nn.Module):
    """Predicts boundary probability over the token embeddings."""

    def __init__(self, emb_dim, hidden_dim, kernel_size=5, dilations=(1, 2, 4)):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations = dilations
        convs = []
        in_ch = emb_dim
        for d in dilations:
            convs.append(nn.Conv1d(in_ch, hidden_dim, kernel_size, dilation=d))
            in_ch = hidden_dim
        self.convs = nn.ModuleList(convs)
        self.act = nn.GELU()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T) for Conv1d
        h = x.transpose(1, 2)
        for conv, d in zip(self.convs, self.dilations):
            pad = (self.kernel_size - 1) * d
            h = F.pad(h, (pad, 0))  # causal: pad on the left only
            h = self.act(conv(h))
        h = h.transpose(1, 2)  # (B, T, hidden_dim)
        out = self.linear(h)
        out = torch.sigmoid(out)
        return out.squeeze(-1)  # (B, T)


def chunk_weights(C, K, beta):
    """Per-token distribution over K chunks, 
    given C: (B, T)"""
    device = C.device
    k = torch.arange(K, device=device).view(1, 1, K)
    C = C.unsqueeze(-1)
    logits = -beta * (C - k) ** 2
    return F.softmax(logits, dim=-1)  # (B, T, K)


def hard_chunk_bounds(chunk_id, K):
    """Converts hard chunk assignment (B, T) into per-chunk
    (start, end) token offsets (B, K)."""
    B, T = chunk_id.shape
    device = chunk_id.device

    one_hot = F.one_hot(chunk_id, K).bool()
    has = one_hot.any(dim=1)

    start = one_hot.float().argmax(dim=1)

    expected = (
        (torch.arange(K, device=device).view(1, K) * (T // K))
        .expand(B, K)
        .clamp(0, T - 1)
    )
    start = torch.where(has, start, expected)

    # end of chunk k = start of chunk k+1, with the last chunk ending at T
    end = torch.cat([start[:, 1:], torch.full((B, 1), T, device=device)], dim=1)

    return start, end


def gumbel_hard(weights):
    """Gumbel-softmax: returns the hard chunk index (B, T)
    and a one-hot tensor whose backward pass uses the soft weights."""
    logp = torch.log(weights.clamp_min(1e-9))

    gumbel = -torch.log(-torch.log(torch.rand_like(logp)))
    y_soft = F.softmax(logp + gumbel, dim=-1)

    index = y_soft.argmax(dim=-1)
    y_hard = F.one_hot(index, y_soft.size(-1)).type_as(y_soft)

    y = y_hard.detach() - y_soft.detach() + y_soft  # straight-through estimator

    return index, y


def extract_chunks(x, start, end, max_len=None):
    """Gathers token embeddings starting
    at `start`, zero-padded past `end`. Returns (chunks, mask) of shape
    (B, K, L, D) and (B, K, L)."""
    B, T, D = x.shape
    device = x.device

    lengths = (end - start).clamp_min(0)

    if max_len is None:
        L_max = lengths.max()
    else:
        L_max = max_len
        lengths = lengths.clamp_max(max_len)

    pos = torch.arange(L_max, device=device)
    idx = start.unsqueeze(-1) + pos
    mask = pos < lengths.unsqueeze(-1)
    idx = idx.clamp(0, T - 1)

    x_exp = x.unsqueeze(1).expand(B, start.size(1), T, D)
    chunks = torch.gather(x_exp, dim=2, index=idx.unsqueeze(-1).expand(-1, -1, -1, D))
    chunks = chunks * mask.unsqueeze(-1)

    return chunks, mask


class BatchedAttentionLayer(nn.Module):
    """Standard pre-norm transformer block, used
    for the chunk-level transformer."""

    def __init__(self, d_model, d_ffn, nhead, dropout=0.0, bias=True):
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

        if mask is not None:
            attn_mask = mask.unsqueeze(1)
            is_causal = False
        else:
            attn_mask = None
            is_causal = True

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
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
    """Standard sinusoidal positional encoding, added in-place by default."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x, inplace=True):
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        if not inplace:
            return self.dropout(x)
        return F.dropout(x, p=self.dropout.p, training=self.training, inplace=True)


class HierarchicalTransformer(nn.Module):
    """

    Args:
        d_model: embedding dimension.
        d_ffn: feed-forward hidden dimension.
        nhead: number of attention heads.
        chunk_layers: number of chunk-level transformer layers.
        vocab_size: tokenizer vocabulary size.
        max_chunks: upper bound on K, the number of chunks per sequence.
        beta: temperature for the soft chunk-assignment kernel (higher =
            sharper).
        max_len: maximum sequence length T (for positional
            encodings).
        max_chunk_len: cap L on tokens per chunk; defaults to
            min(max_len, 1.5 * max_len / max_chunks).
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        chunk_layers,
        vocab_size=8018,
        max_chunks=16,
        beta=5.0,
        max_len=256,
        dropout=0.0,
        max_chunk_len=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_chunks = max_chunks
        self.beta = beta
        self.max_chunk_len = max_chunk_len or min(
            max_len, 1.5 * max(1, max_len // max_chunks)
        )

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=max_len)
        self.max_len = max_len

        self.boundary = BoundaryPredictor(d_model, 64)

        self.chunk_layers = nn.ModuleList(
            [
                BatchedAttentionLayer(d_model, d_ffn, nhead, dropout)
                for i in range(chunk_layers)
            ]
        )

        self.unembedding = nn.Linear(d_model, vocab_size, bias=False)
        self.unembedding.weight = self.embedding.weight

        # chunk representation prepended in place of the "previous chunk"
        # summary for the first chunk, which has no predecessor
        self.start = nn.Parameter(torch.randn(self.d_model))

        # 1.0 = uniform boundaries (warm-up)
        # 0.0 = fully use the learned boundary predictor. Set by
        # the training loop (see train.boundary_force_ratio_at).
        self.boundary_force_ratio = 1.0

    def forward(self, tokens):
        device = tokens.device
        B, T = tokens.shape
        K = min(self.max_chunks, T)

        x = self.embedding(tokens)
        x = self.positional(x)

        # boundary prediction
        pred_B_prob = self.boundary(x)  # (B, T)
        ones_B_prob = torch.ones_like(pred_B_prob, device=device)

        ratio = self.boundary_force_ratio
        B_prob = ratio * ones_B_prob + (1 - ratio) * pred_B_prob

        target_rate = K / T
        rate_loss = (B_prob.mean() - target_rate).pow(2)

        C = torch.cumsum(B_prob, dim=1)

        # normalize C into [0, K)
        expected_final = B_prob.mean(dim=-1, keepdim=True) * T
        C = C / (expected_final + 1e-6) * (K - 1)

        # soft weights to hard chunk assignment
        weights = chunk_weights(C, K, self.beta)  # (B, T, K)
        chunk_id, hard_assign = gumbel_hard(weights)

        length_loss = (torch.square(T / K - torch.sum(hard_assign, dim=1))).mean()

        # hard chunk boundaries
        start, end = hard_chunk_bounds(chunk_id, K)  # (B, K)

        # extract reduced-length chunks
        cap = min(self.max_chunk_len, T)
        chunks, chunk_mask = extract_chunks(x, start, end, max_len=cap)
        # (B, K, cap, D), (B, K, cap)

        # chunk summaries, used as a causal "previous chunks" prefix
        chunk_repr = chunks.sum(dim=2) / chunk_mask.sum(dim=2).clamp(min=1).unsqueeze(
            -1
        )
        chunk_repr = chunk_repr.unsqueeze(2).expand(B, K, K, -1).transpose(1, 2)
        # (B, K, K, D)

        chunk_repr_mask = torch.tril(torch.ones(K, K, device=device), diagonal=-1)
        # (K, K)

        chunk_repr = chunk_repr * chunk_repr_mask.view(1, K, K, 1).clone()

        chunk_repr[:, :, -1, :] = self.start

        chunks = torch.cat((chunk_repr, chunks), 2)
        # (B, K, K+L, D): each chunk sees a causal prefix of prior-chunk
        # summaries, then its own tokens

        B, K, L, D = chunks.shape
        chunks = chunks.view(B * K, L, D)

        # chunk-level transformer
        for layer in self.chunk_layers:
            chunks = layer(chunks)

        chunks = chunks.view(B, K, L, D)
        chunks = chunks[:, :, K:, :]  # drop the prefix, keep only token outputs

        # map chunk outputs back to per-token vocab logits
        B, K, L, D = chunks.shape
        V = self.vocab_size

        logits = self.unembedding(chunks)  # (B, K, L, V)

        # original-sequence position of each (chunk, chunk-local-offset) pair
        positions = start.unsqueeze(-1) + torch.arange(L, device=device).view(1, 1, L)
        valid = (positions < end.unsqueeze(-1)) & (positions < T) & chunk_mask

        positions = positions.view(B, K, L).clamp(0, T - 1)

        idx = positions.unsqueeze(-1).expand(B, K, L, V).reshape(B, K * L, V)
        logits = (
            logits.view(B, K, L, V)
            .masked_fill(~valid.unsqueeze(-1), 0)
            .reshape(B, K * L, V)
        )
        counts = valid.float().reshape(B, K * L)

        out = logits.new_zeros(B, T, V, device=device)
        coverage = counts.new_zeros(B, T, device=device)

        out.scatter_add_(1, idx, logits)
        coverage.scatter_add_(1, positions.reshape(B, K * L), counts)

        out = out / coverage.clamp_min(1).unsqueeze(-1)

        return out, length_loss, B_prob, chunk_id, rate_loss

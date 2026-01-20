import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import time
import math

def read_vocab(fname):
    file = open(fname, encoding = "utf-8").read().splitlines()
    vocab = []
    for word in file:
        if any((ord(ch) > 128) for ch in word): continue
        if word[:2] == "##":
            vocab.append("§" + word[2:])
        else:
            vocab.append(word)
    vocab = list(filter(lambda x: len(x) <= 4 or ('§' in x and len(x) == 5), vocab))
    vocab += ["[SEP]", "[UNK]", "[CLS]"]
    return vocab

def get_vocab_dct_encoder(vocab):
    vocab_dct = {vocab[idx] : idx for idx in range(len(vocab))}
    return vocab_dct

def build_mask(seq):
    B, T = seq.shape
    device = seq.device
    # prevent seeing future tokens
    causal = torch.triu(torch.ones(T, T, dtype = torch.bool, device = device), diagonal = 1)
    # map 1 -> -1e9
    additive = causal.float().masked_fill(causal, float("-1e9"))
    return additive.unsqueeze(0).expand(B, T, T)

class BoundaryPredictor(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers = num_layers, bidirectional = True, batch_first = True)
        # hidden_dim * 2 because of bidirectional
        self.linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.linear(out)).squeeze(-1)

class ChunkedDataset(Dataset):
    def __init__(self, tokens, vocab_dct, block_size = 256):
        self.vocab_dct = vocab_dct
        ids = [vocab_dct[t] for t in tokens]
        self.ids = torch.tensor(ids, dtype = torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.ids) - self.block_size

    def __getitem__(self, idx):
        x = self.ids[idx: idx + self.block_size]
        y = self.ids[idx + 1: idx + 1 + self.block_size]
        return x, y

def collate(batch):
    x, y = zip(*batch)
    return torch.stack(x, dim = 0), torch.stack(y, dim = 0)

def chunk_weights(C, K, beta):
    device = C.device
    # for each chunk, cumulative sum should be approximately k-1
    k = torch.arange(K, device = device).view(1, 1, K)
    C = C.unsqueeze(-1)
    logits = -beta * (C - k) ** 2
    return F.softmax(logits, dim = -1)

def hard_chunk_bounds(chunk_id, K):
    B, T = chunk_id.shape
    device = chunk_id.device

    # one hot vector for each token representing chunk
    one_hot = torch.nn.functional.one_hot(chunk_id, K).bool()

    # for each sequence, does it have tokens in this chunk?
    has = one_hot.any(dim = 1)

    # find starting index (not differentiable)
    start = one_hot.float().argmax(dim = 1)

    # reverse the one hot to find the ends (not differentiable)
    rev = torch.flip(one_hot, dims=[1])
    last = rev.float().argmax(dim = 1)
    end = T - last

    start = torch.where(has, start, torch.full_like(start, T))
    end = torch.where(has, end, torch.zeros_like(end))

    return start, end

def gumbel_hard(weights, tau = 1.0):
    # convert probabilities to log for stability
    logp = torch.log(weights.clamp_min(1e-9))

    # sampling
    gumbel = -torch.log(-torch.log(torch.rand_like(logp)))

    # compute chunk assignments
    y_soft = F.softmax((logp + gumbel) / tau, dim = -1)

    index = y_soft.argmax(dim = -1)
    y_hard = F.one_hot(index, y_soft.size(-1)).type_as(y_soft)

    y = y_hard.detach() - y_soft.detach() + y_soft

    return index, y

def extract_chunks(x, start, end):
    B, T, D = x.shape
    K = start.size(1)
    device = x.device

    # find lengths and max length overall
    lengths = (end - start).clamp_min(0)
    L_max = lengths.max().item()

    chunks = torch.zeros(B, K, L_max, D, device = device)
    mask = torch.zeros(B, K, L_max, device = device)

    for b in range(B):
        for k in range(K):
            l = lengths[b, k]
            if l > 0:
                chunks[b, k, :l] = x[b, start[b, k]:end[b, k]]
                mask[b, k, :l] = 1

    return chunks, mask

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

class BatchedAttentionLayer(nn.Module):
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

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(B, T, H, dH).transpose(1, 2)
        k = k.view(B, T, H, dH).transpose(1, 2)
        v = v.view(B, T, H, dH).transpose(1, 2)

        if mask is not None:
            attn_mask = mask.unsqueeze(1)
        else:
            attn_mask = None

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        attn = attn.transpose(1, 2).contiguous().view(B, T, D)

        x = self.ln1(x + attn)

        out = self.W_a(x)
        out = self.relu(out)
        out = self.W_b(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        x = self.ln2(x + out)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        return self.dropout(x)

class HierarchicalTransformer(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, token_layers, chunk_layers, vocab_size = 8018, max_chunks = 8, beta = 5.0, max_len = 256, dropout = 0.0):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_chunks = max_chunks
        self.beta = beta

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len = max_len)

        self.token_layers = nn.ModuleList(
            [BatchedAttentionLayer(d_model, d_ffn, nhead, dropout) for _ in range(token_layers)]
        )

        self.boundary = BoundaryPredictor(d_model, 64, 2)

        self.chunk_layers = nn.ModuleList(
            [BatchedAttentionLayer(d_model, d_ffn, nhead, dropout) for _ in range(chunk_layers)]
        )

        self.unembedding = nn.Linear(d_model, vocab_size, bias = False)
        self.unembedding.weight = self.embedding.weight

    def forward(self, tokens, mask = None, tau = 0.7):
        device = tokens.device
        B, T = tokens.shape
        K = min(self.max_chunks, T)

        # token-level transformer
        x = self.embedding(tokens)
        x = self.positional(x)

        for layer in self.token_layers:
            x = layer(x, mask = mask) # (B, T, D)

        # boundary prediction
        B_prob = self.boundary(x).clamp(0.01, 0.99) # (B, T)
        C = torch.cumsum(B_prob, dim = 1)

        # normalize C into [0, K)
        C = C / (C[:, -1:].detach() + 1e-6) * (K - 1)

        # soft weights
        weights = chunk_weights(C, K, self.beta) # (B, T, K)

        # gumbel-softmax
        if self.training:
            chunk_id, hard_assign = gumbel_hard(weights, tau = tau)
        else:
            chunk_id = weights.argmax(dim = -1)
            hard_assign = torch.nn.functional.one_hot(chunk_id, K).float()

        # try to make chunks evenly sized
        length_loss = (torch.sum(torch.square(torch.sum(hard_assign, dim = 1))))
        length_loss /= (B * K * (T / K) ** 2)

        # hard chunk boundaries
        start, end = hard_chunk_bounds(chunk_id, K) # (B, K)

        # extract reduced-length chunks
        chunks, chunk_mask = extract_chunks(x, start, end) # (B, K, L_max, D), (B, K, L_max)

        B, K, L, D = chunks.shape
        chunks = chunks.view(B * K, L, D)

        # chunk-level transformer
        causal = torch.triu(torch.ones(L, L, device = device), diagonal = 1).bool()
        attn_mask = causal.float().masked_fill(causal, -1e9)
        attn_mask = attn_mask.unsqueeze(0)

        for layer in self.chunk_layers:
            chunks = layer(chunks, mask = attn_mask)

        chunks = chunks.view(B, K, L, D)

        # map chunk outputs to vocab

        B, K, L, D = chunks.shape
        V = self.vocab_size

        # unembed
        logits = self.unembedding(chunks) # (B, K, L, V)

        # absolute token positions
        positions = start.unsqueeze(-1) + torch.arange(L, device = device).view(1, 1, L)

        # valid mask
        valid = (positions < end.unsqueeze(-1)) & (positions < T)
        valid = valid & chunk_mask.bool()

        # flatten
        logits = logits.view(B * K * L, V)
        positions = positions.view(B * K * L)

        batch_idx = torch.arange(B, device = device).view(B, 1, 1).expand(B, K, L).reshape(-1)

        valid = valid.view(-1)

        # select valid entries
        logits = logits[valid]
        positions = positions[valid]
        batch_idx = batch_idx[valid]

        # allocate outputs
        out = logits.new_zeros(B, T, V, device = device)
        coverage = logits.new_zeros(B, T, device = device)

        out.index_put_((batch_idx, positions), logits, accumulate = True)

        coverage.index_put_((batch_idx, positions), torch.ones_like(positions, dtype = coverage.dtype), accumulate = True)

        out = out / coverage.clamp_min(1).unsqueeze(-1)

        return out, length_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

corpus = open("data.txt", encoding = "latin-1").read().split()

vocab = read_vocab("vocab.txt")
vocab_dct = get_vocab_dct_encoder(vocab)

checkpoint_path = "chunk192.pth"
d_model = 192
nhead = 8
token_layers = 1
chunk_layers = 8
dropout = 0.1
block_size = 256
batch_size = 16
epochs = 10
lr = 3e-4
warmup_steps = 2000
max_grad_norm = 1.0
save_every_steps = 1000
device = device
ds = ChunkedDataset(corpus, vocab_dct, block_size)
loader = DataLoader(ds, batch_size = batch_size, shuffle = True, collate_fn = collate, drop_last = True)

model = HierarchicalTransformer(d_model, d_model * 4, nhead, token_layers, chunk_layers, dropout = dropout, max_chunks = 32, max_len = block_size).to(device)

start_step = 0
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, weights_only = False, map_location = device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    start_step = ckpt.get("step", 0)
    print(f"Loaded checkpoint from {checkpoint_path}, start_step is {start_step}")

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, betas = (0.9, 0.98), weight_decay = 0.01)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    return (0.99) ** (step // 1000)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler()

for step in range(start_step):
    scheduler.step()

t = time.time()

iters = start_step
model.train()
for epoch in range(1, epochs + 1):
    t0 = time.time()
    running_loss = 0.0
    for batch_idx, (seq, target) in enumerate(loader, start = 1):
        seq = seq.to(device)
        target = target.to(device)
        mask = build_mask(seq).to(device)

        optimizer.zero_grad(set_to_none = True)

        with torch.amp.autocast(device_type = device.type, dtype = torch.float16 if device.type == "cuda" else torch.float32):
            logits, length_loss = model(seq, mask = mask)
            B, T, V = logits.shape
            loss_val = loss_f(logits.view(-1, V), target.view(-1))

        loss_val += length_loss

        scaler.scale(loss_val).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        running_loss += loss_val.item()
        iters += 1

        if iters % 100 == 0:
            avg = running_loss / 100
            print(f"epoch {epoch}, iter {iters}, loss {avg:.4f}, lr {optimizer.param_groups[0]['lr']:.2e}, t {time.time() - t:.4f}")
            running_loss = 0.0
            t = time.time()

        if iters % save_every_steps == 0:
            torch.save({"model_state": model.state_dict(), "step": iters}, checkpoint_path)
            print(f"saved checkpoint")

    t1 = time.time()
    print(f"epoch {epoch} completed in {t1 - t0:.1f}s, lr {optimizer.param_groups[0]['lr']:.2e}")
    torch.save({"model_state": model.state_dict(), "step": iters}, checkpoint_path)

print("training finished")

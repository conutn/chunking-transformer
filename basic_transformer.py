import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import time
import os

class BatchedAttentionLayer(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, dropout = 0.0, bias = True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model, bias = bias)
        self.W_k = nn.Linear(d_model, d_model, bias = bias)
        self.W_v = nn.Linear(d_model, d_model, bias = bias)
        self.W_a = nn.Linear(d_model, d_ffn, bias = bias)
        self.relu = nn.ReLU()
        self.W_b = nn.Linear(d_ffn, d_model, bias = bias)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.LayerNorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask = None):
        B, T, D = x.shape
        H = self.nhead
        dH = self.d_head

        q = self.W_q(x).view(B, T, H, dH).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dH).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dH).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(2, 3)) / (dH ** 0.5)

        if mask is not None:
            attn_logits = attn_logits + mask.unsqueeze(1)

        attn = F.softmax(attn_logits, dim = -1)
        attn = self.dropout(attn)

        attn = torch.matmul(attn, v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)

        x = x + attn
        x = self.LayerNorm1(x)

        out = self.W_a(x)
        out = self.relu(out)
        out = self.W_b(out)
        out = self.dropout(out)

        x = x + out
        x = self.LayerNorm2(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        return self.dropout(x)

class BatchedAttention(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout, d_ffn = 768, vocab_size = 27628, bias = True, max_len = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoding(d_model, max_len = max_len)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(BatchedAttentionLayer(d_model, d_ffn, nhead, dropout, bias))
        self.unembedding = nn.Linear(d_model, vocab_size, bias = False)
        self.unembedding.weight = self.embedding.weight

        self.embedding.no_grad = True
        self.unembedding.no_grad = True

    def forward(self, x, mask = None):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        for layer in self.layers:
            x = layer(x, mask = mask)
        x = self.unembedding(x)
        return x

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
    vocab += ["[SEP]", "[PAD]", "[UNK]"]
    return vocab

def get_vocab_dct_encoder(vocab):
    vocab_dct = {vocab[idx] : idx for idx in range(len(vocab))}
    return vocab_dct

def load_embedding_matrix(fname, device):
    loaded = torch.load(fname, weights_only = False, map_location = device)
    if isinstance(loaded, dict) and "weight" in loaded:
        return loaded["weight"].to(device)

    return loaded.weight.data.to(device)

class TransformerDataset(Dataset):
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

def build_mask(seq):
    B, T = seq.shape
    device = seq.device
    causal = torch.triu(torch.ones(T, T, dtype = torch.bool, device = device), diagonal=1)
    additive = causal.float().masked_fill(causal, float("-1e9"))
    return additive.unsqueeze(0).expand(B, T, T)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

corpus = open("data.txt", encoding="latin-1").read().split()

vocab = read_vocab("vocab.txt")
vocab_dct = get_vocab_dct_encoder(vocab)

embedding_path = "input192.pt"
checkpoint_path = "transformer.pth"
d_model = 192
nhead = 8
num_layers = 12
dropout = 0.1
freeze_embeddings = False
block_size = 128
batch_size = 32
epochs = 10
lr = 3e-4
warmup_steps = 2000
max_grad_norm = 1.0
save_every_steps = 1000

device = device
ds = TransformerDataset(corpus, vocab_dct, block_size)
loader = DataLoader(ds, batch_size = batch_size, shuffle = True, collate_fn = collate, drop_last = True, num_workers = 2)

model = BatchedAttention(d_model, nhead, num_layers, dropout, d_ffn = 768, vocab_size = len(vocab_dct), max_len = 256).to(device)

if os.path.exists(embedding_path):
    emb_w = load_embedding_matrix(embedding_path, device)
    if emb_w.shape != model.embedding.weight.data.shape:
        print(f"embedding shape mismatch: file {emb_w.shape}, model {model.embedding.weight.data.shape}")
    else:
        model.embedding.weight.data.copy_(emb_w)
        model.unembedding.weight.data.copy_(emb_w)

model.unembedding.weight = model.embedding.weight
if freeze_embeddings:
    model.embedding.weight.requires_grad = False
    model.unembedding.weight.requires_grad = False

start_step = 0
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, weights_only = False, map_location = device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    start_step = ckpt.get("step", 0)
    print(f"Loaded checkpoint from {checkpoint_path}, start_step is {start_step}")

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, betas = (0.9, 0.95), weight_decay = 0.01)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    return (0.99) ** (step // 1000)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler()

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

        with torch.amp.autocast(device_type = device.type, dtype=torch.float16 if device.type == "cuda" else torch.float32):
            logits = model(seq, mask = mask)
            B, T, V = logits.shape
            loss_val = loss_f(logits.view(-1, V), target.view(-1))

        scaler.scale(loss_val).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        running_loss += loss_val.item()
        iters += 1

        if iters % 10 == 0:
            avg = running_loss / 10
            print(f"epoch {epoch}, iter {iters}, loss {avg:.4f}, lr {optimizer.param_groups[0]['lr']:.2e}")
            running_loss = 0.0

        if iters % save_every_steps == 0:
            torch.save({"model_state": model.state_dict(), "step": iters}, checkpoint_path)
            print(f"saved checkpoint")

    t1 = time.time()
    print(f"epoch {epoch} completed in {t1 - t0:.1f}s, lr {optimizer.param_groups[0]['lr']:.2e}")
    torch.save({"model_state": model.state_dict(), "step": iters}, checkpoint_path)

print("training finished")
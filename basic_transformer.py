import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class BatchedAttentionLayer(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, dropout = 0.0, bias = True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
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
        # x:    (batch, seq_len, d_model)
        # mask: (batch, seq_len, seq_len) or None
        #       0 = keep, -inf = stop
        B, T, D = x.shape
        H = self.nhead
        dH = self.d_head
        
        q = self.W_q(x).view(B, T, H, dH).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dH).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dH).transpose(1, 2)
        
        attn_logits = torch.matmul(q, k.transpose(2, 3)) / (dH ** 0.5)
        
        if mask is not None:
            #if mask.dtype != torch.bool:
            #    mask = torch.log(mask + 1e-9)
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
        self.dropout = nn.Dropout(p = dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BatchedAttention(nn.Module): # use cross-entropy loss
    def __init__(self, d_model, nhead, num_layers, dropout, d_ffn = 768, vocab_size = 27628, bias = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(BatchedAttentionLayer(d_model, d_ffn, nhead, dropout, bias))
        self.unembedding = nn.Linear(d_model, vocab_size, bias = False)
        self.unembedding.weight = self.embedding.weight

    def forward(self, x, mask = None):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        for layer in self.layers:
            x = layer(x, mask = mask)
        x = self.unembedding(x)
        return x

class TransformerDataset(Dataset):
    def __init__(self, tokens, vocab_dct):
        self.vocab_dct = vocab_dct
        self.ids = torch.tensor([vocab_dct[t] for t in tokens], dtype = torch.long)
        self.length = len(self.ids) - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq = self.ids[:idx]
        target = self.ids[1:idx+1]
        return seq, target

def collate(batch):
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    max_len = max(seq.size(0) for seq in sequences)
    padded = torch.zeros(len(batch), max_len, dtype = torch.long)
    padded_targets = torch.zeros(len(batch), max_len, dtype = torch.long)

    for i, seq in enumerate(sequences):
        padded[i, :seq.size(0)] = seq
    for i, target in enumerate(targets):
        padded_targets[i, :target.size(0)] = target

    return padded, padded_targets

def build_attention_mask(seq):
    B, T = seq.shape
    device = seq.device

    pad_mask = (seq == 0).unsqueeze(1).expand(B, T, T)
    causal_mask = torch.triu(torch.ones(T, T, dtype = torch.bool, device = device), diagonal = 1)
    causal_mask = causal_mask.unsqueeze(0).expand(B, T, T)

    full_mask = pad_mask | causal_mask
    full_mask = full_mask.float().masked_fill(full_mask.bool(), -1e9)
    return full_mask.to(device)

def chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def read_vocab(fname):
    file = open(fname, encoding = "utf-8").read().splitlines()
    vocab = []
    for word in file:
        if any((ord(ch) > 128) for ch in word): continue
        if word[:2] == "##":
            vocab.append("§" + word[2:])
        else:
            vocab.append(word)
    return vocab

def get_vocab_dct_encoder(vocab):
    vocab_dct = {vocab[idx] : idx for idx in range(len(vocab))}
    return vocab_dct

vocab = read_vocab("vocab.txt")
vocab_set = set(vocab)
vocab_dct = get_vocab_dct_encoder(vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BatchedAttention(192, 8, 24, 0.1).to(device)
#model.load_state_dict(torch.load("transformer.pth"))
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

epochs = 100
batch_size = 64
chunk_size = 256
corpus = open("data.txt", encoding = "latin-1").read().split()

scaler = torch.amp.GradScaler("cuda")

for epoch in range(1, epochs + 1):
    total_loss = 0.0
    local_loss = 0.0
    iters = 0

    for chunk in chunks(corpus, chunk_size):
        ds = TransformerDataset(chunk, vocab_dct)

        loader = DataLoader(ds, batch_size = batch_size, shuffle = True, collate_fn = collate)

        for seq, next in loader:
            seq = seq.to(device)
            next = next.to(device)
            mask = build_attention_mask(seq)

            optimizer.zero_grad(set_to_none = True)

            with torch.amp.autocast("cuda", dtype = torch.float16):
              logits = model(seq, mask = mask)
              loss_val = loss(logits.transpose(1, 2), next)

            scaler.scale(loss_val).backward()
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(loss_val):
              print(iters)
              print(logits)
              idx = torch.isnan(logits)
              print(torch.nonzero(idx))
              print(logits[idx])
              print(next)
              print(loss_val)
              1/0

            total_loss += loss_val.item()
            local_loss += loss_val.item()

            iters += 1
            if iters % 10 == 0:
              avg_loss = total_loss / max(1, iters)
              local_loss /= 10
              print(f"Epoch {epoch}, processed {iters * batch_size} pairs, avg loss {avg_loss:.4f}, cur loss {local_loss:.4f}")
              local_loss = 0.0
            if iters % 100 == 0:
              torch.save(model.state_dict(), "transformer.pth")
              print("Saved model state")
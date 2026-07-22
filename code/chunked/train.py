"""
Training script for the HierarchicalTransformer defined in model.py.

Usage:
    python train.py

Expects in the working directory:
    vocab.txt - one token per line (continuation tokens prefixed with `##`)
    data_tokens.txt - whitespace-separated corpus (ints)

Writes, in the working directory:
    chunk192.pth  - model/optimizer/scheduler checkpoint
    loss.txt      - "<train_ce_loss> <iter_time_ms>"
    valid.txt     - validation CE loss

The GPU telemetry requires
pynvml and an NVIDIA GPU; delete if unavailable

IMPORTANT:
Run in a separate directory than the standard version, both use `loss.txt` and `valid.txt`
for writing.
"""

import math
import os
import time

import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    ChunkedDataset,
    Tokenizer,
    collate,
    get_split,
    get_vocab_dct_encoder,
    read_vocab,
)
from model import HierarchicalTransformer

# Config

VOCAB_PATH = "vocab.txt"
DATA_PATH = "data_tokens.txt"
CHECKPOINT_PATH = "chunk192.pth"

D_MODEL = 192
NHEAD = 12
CHUNK_LAYERS = 8
DROPOUT = 0.1

BLOCK_SIZE = 1024
BATCH_SIZE = 16
EPOCHS = 1

LR = 1e-4
WARMUP_STEPS = 2000
MAX_GRAD_NORM = 1.0

VALIDATE_STEPS = 10000
SAVE_STEPS = 1000
PRINT_STEPS = 100
BETA = 5.0

BOUNDARY_FORCE_STEPS = 10000
BOUNDARY_ANNEAL_STEPS = 10000


# GPU Telemetry
def gpu_temp(handle):
    return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)


def gpu_clock(handle):
    return pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)


def gpu_power(handle):
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


# REINFORCE Loss
class BoundaryReinforceLoss(nn.Module):
    """pushes the boundary predictor's
    probabilities towards the assignment that had lower
    per-sample cross-entropy, using a running mean/variance baseline."""

    def __init__(self, momentum=0.99, eps=1e-4):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.zeros(1))
        self.register_buffer("step", torch.zeros(1))

    def forward(self, B_prob, chunk_id, per_sample_ce):
        B_prob = B_prob.float().clamp(1e-3, 1 - 1e-3)

        actual_boundary = torch.zeros_like(B_prob)
        actual_boundary[:, 1:] = (chunk_id[:, 1:] != chunk_id[:, :-1]).float()

        pos = actual_boundary
        neg = 1 - actual_boundary
        pos_logp = (pos * torch.log(B_prob)).sum(dim=1) / pos.sum(dim=1).clamp(min=1)
        neg_logp = (neg * torch.log(1 - B_prob)).sum(dim=1) / neg.sum(dim=1).clamp(
            min=1
        )
        log_p = 0.5 * (pos_logp + neg_logp)  # (B,)

        reward = -per_sample_ce  # already detached at the call site

        if self.training:
            with torch.no_grad():
                batch_mean = reward.mean()
                batch_var = reward.var(unbiased=False)
                self.step += 1
                self.running_mean.mul_(self.momentum).add_(
                    batch_mean, alpha=1 - self.momentum
                )
                self.running_var.mul_(self.momentum).add_(
                    batch_var, alpha=1 - self.momentum
                )

        bias_correction = 1 - self.momentum ** self.step.clamp(min=1)
        mean = self.running_mean / bias_correction
        var = self.running_var / bias_correction

        std = var.clamp_min(self.eps).sqrt()
        advantage = (reward - mean) / std
        advantage = advantage.clamp(-3, 3)

        return -(advantage * log_p).mean()


def boundary_force_ratio_at(step):
    """schedule for HierarchicalTransformer.boundary_force_ratio."""
    if step <= BOUNDARY_FORCE_STEPS:
        return 1.0
    progress = (step - BOUNDARY_FORCE_STEPS) / BOUNDARY_ANNEAL_STEPS
    return max(0.0, 1.0 - progress)


def sample_from_model(model, tokenizer, vocab, vocab_dct, device, input_text, length):
    """autoregressive generation from a
    trained HierarchicalTransformer."""
    model.eval()

    tokens = list(map(vocab_dct.get, tokenizer.tokenize(input_text)))
    input_idxs = [tokens[-model.max_len :]]
    ret = input_idxs[0][:]
    with torch.no_grad():
        for i in range(length):
            input_tensor = torch.tensor(input_idxs).to(device)
            pred = model(input_tensor)[0]
            pred = pred.view(-1, model.vocab_size)[-1]

            pred = torch.softmax(pred, 0)

            sampled_idx = torch.multinomial(pred, num_samples=1).item()
            ret.append(sampled_idx)

            input_idxs[0].append(sampled_idx)
            input_idxs[0] = input_idxs[0][-model.max_len :]

    model.train()

    r = " ".join(list(map(vocab.__getitem__, ret)))
    r = r.replace(" §", "")
    return r


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    vocab = read_vocab(VOCAB_PATH)
    tokenizer = Tokenizer(vocab, "[UNK]")
    vocab_dct = get_vocab_dct_encoder(vocab)
    # tokenizer / vocab_dct are not used in the training loop itself; they're
    # built here so sample_from_model(model, tokenizer, vocab, vocab_dct,
    # device, prompt, length) can be called for qualitative checks.

    tokens = list(map(int, open(DATA_PATH).read().split()))
    train_data, valid_data, test_data = get_split(tokens)

    losses = []
    times = []
    loss_writer = open("loss.txt", "a", 1)
    valid_writer = open("valid.txt", "a", 1)

    ds = ChunkedDataset(train_data, BLOCK_SIZE)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True
    )

    valid_ds = ChunkedDataset(valid_data, BLOCK_SIZE)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )

    test_ds = ChunkedDataset(test_data, BLOCK_SIZE)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True
    )

    model = HierarchicalTransformer(
        D_MODEL,
        D_MODEL * 4,
        NHEAD,
        CHUNK_LAYERS,
        dropout=DROPOUT,
        max_chunks=int(BLOCK_SIZE**0.5),
        max_len=BLOCK_SIZE,
        vocab_size=len(vocab),
        beta=BETA,
        max_chunk_len=int(1.5 * BLOCK_SIZE**0.5),
    ).to(device)

    start_step = 0
    reinforce_loss_fn = BoundaryReinforceLoss(momentum=0.99).to(device)
    loss_f = nn.CrossEntropyLoss()

    boundary_params = list(model.boundary.parameters())
    boundary_param_ids = set(id(p) for p in boundary_params)
    main_params = [p for p in model.parameters() if id(p) not in boundary_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": LR},
            {"params": boundary_params, "lr": LR / 5},
        ],
        betas=(0.9, 0.98),
        weight_decay=0.01,
        fused=True,
    )

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS

        progress = (step - WARMUP_STEPS) / (len(loader) * EPOCHS - WARMUP_STEPS)
        progress = min(progress, 1.0)

        min_ratio = 0.03
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_ratio + (1 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for step in range(start_step):
        scheduler.step()

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "rf_loss_state" in ckpt:
            reinforce_loss_fn.load_state_dict(ckpt["rf_loss_state"])
        start_step = ckpt.get("step", 0)
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}, start_step is {start_step}")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()

    iters = start_step
    model.train()

    need_valid = start_step % VALIDATE_STEPS == 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        running_loss = 0.0
        if iters > len(loader) * EPOCHS:
            break
        for seq, target in loader:
            if iters > len(loader) * EPOCHS:
                break

            model.boundary_force_ratio = boundary_force_ratio_at(iters)

            seq = seq.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, length_loss, B_prob, chunk_id, rate_loss = model(seq)
                B, T, V = logits.shape

                per_sample_ce = (
                    F.cross_entropy(
                        logits.view(B * T, V), target.view(B * T), reduction="none"
                    )
                    .view(B, T)
                    .mean(dim=1)
                )

                ce_loss = per_sample_ce.mean()

            rf_loss = reinforce_loss_fn(
                B_prob, chunk_id.detach(), per_sample_ce.detach()
            )

            rf_weight = 1 - model.boundary_force_ratio
            total_loss = ce_loss + length_loss + rf_weight * rf_loss + rate_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += ce_loss.detach()
            iters += 1

            if iters % PRINT_STEPS == 0:
                ender.record()
                torch.cuda.synchronize()
                et = starter.elapsed_time(ender)
                starter.record()

                avg = running_loss.item() / PRINT_STEPS

                print(
                    f"iter {iters}, loss {avg:.4f}, t {et:.4f}, "
                    f"{gpu_temp(handle)}, {gpu_clock(handle)}, {gpu_power(handle)}, "
                    f"{et * gpu_power(handle)}"
                )

                running_loss = 0.0
                losses.append(str(avg))
                times.append(str(et))

            if iters % SAVE_STEPS == 0:
                for i in range(len(losses)):
                    loss_writer.write(losses[i] + " " + times[i] + "\n")
                losses = []
                times = []

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "rf_loss_state": reinforce_loss_fn.state_dict(),
                        "step": iters,
                    },
                    CHECKPOINT_PATH,
                )
                print("saved checkpoint")
                starter.record()

            if iters % VALIDATE_STEPS == 0 or need_valid:
                need_valid = False
                print("beginning validation for step", iters)
                model.eval()
                with torch.no_grad():
                    total_valid_loss = 0
                    valid_iters = 0
                    for seq, target in valid_loader:
                        seq = seq.to(device)
                        target = target.to(device)

                        with torch.amp.autocast(
                            device_type=device.type, dtype=torch.bfloat16
                        ):
                            logits, length_loss, B_prob, chunk_id, rate_loss = model(
                                seq
                            )
                            B, T, V = logits.shape
                            loss_val = loss_f(logits.view(-1, V), target.view(-1))

                        valid_iters += 1
                        total_valid_loss += loss_val.item()

                    print(f"validation loss {total_valid_loss / valid_iters:.4f}")
                    valid_writer.write(str(total_valid_loss / valid_iters) + " \n")
                print("validation complete")
                starter.record()
                model.train()

        t1 = time.time()
        print(f"epoch {epoch} completed in {t1 - t0:.1f}")
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "rf_loss_state": reinforce_loss_fn.state_dict(),
                "step": iters,
            },
            CHECKPOINT_PATH,
        )
        if iters > len(loader) * EPOCHS:
            break

    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        test_iters = 0
        for seq, target in test_loader:
            seq = seq.to(device)
            target = target.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, length_loss, B_prob, chunk_id, rate_loss = model(seq)
                B, T, V = logits.shape
                loss_val = loss_f(logits.view(-1, V), target.view(-1))

            test_iters += 1
            total_test_loss += loss_val.item()

        print(f"test loss {total_test_loss / test_iters:.4f}")


if __name__ == "__main__":
    main()

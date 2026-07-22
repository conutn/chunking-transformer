"""
Training script for the baseline BatchedAttention model

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
Run in a separate directory than the chunked version, both use `loss.txt` and `valid.txt`
for writing.
"""

import itertools
import math
import os
import time

import pynvml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    TransformerDataset,
    Tokenizer,
    collate,
    get_split,
    get_vocab_dct_encoder,
    read_vocab,
)
from model import BatchedAttention

# Config
VOCAB_PATH = "vocab.txt"
DATA_PATH = "data_tokens.txt"
CHECKPOINT_PATH = "transformer192.pth"

D_MODEL = 192
NHEAD = 12
NUM_LAYERS = 8
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


# Optional GPU telemetry
def gpu_temp(handle):
    return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)


def gpu_clock(handle):
    return pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)


def gpu_power(handle):
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    vocab = read_vocab(VOCAB_PATH)
    tokenizer = Tokenizer(vocab, "[UNK]")
    vocab_dct = get_vocab_dct_encoder(vocab)
    # tokenizer / vocab_dct are not used in the training loop itself; kept
    # available here for tokenizing prompts if you add generation later.

    tokens = list(map(int, open(DATA_PATH).read().split()))
    train_data, valid_data, test_data = get_split(tokens)

    losses = []
    times = []
    loss_writer = open("loss.txt", "a", 1)
    valid_writer = open("valid.txt", "a", 1)

    ds = TransformerDataset(train_data, BLOCK_SIZE)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True
    )

    valid_ds = TransformerDataset(valid_data, BLOCK_SIZE)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )

    test_ds = TransformerDataset(test_data, BLOCK_SIZE)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True
    )

    model = BatchedAttention(
        D_MODEL,
        NHEAD,
        NUM_LAYERS,
        DROPOUT,
        d_ffn=D_MODEL * 4,
        vocab_size=len(vocab),
        max_len=BLOCK_SIZE,
    ).to(device)

    start_step = 0

    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
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
        start_step = ckpt.get("step", 0)
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}, start_step is {start_step}")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    print(torch.__version__, torch.version.cuda)
    iters = start_step
    model.train()

    resume_offset = start_step % len(loader)
    start_epoch = start_step // len(loader)

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        epoch_loader = loader
        if epoch == 1 and resume_offset > 0:
            epoch_loader = itertools.islice(loader, resume_offset, None)

        starter.record()

        t0 = time.time()
        running_loss = 0.0
        for seq, target in epoch_loader:
            if iters > len(loader) * EPOCHS:
                break
            seq = seq.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(seq)
                B, T, V = logits.shape
                loss_val = loss_f(logits.view(-1, V), target.view(-1))

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss_val.detach()
            iters += 1

            if iters % PRINT_STEPS == 0:
                ender.record()
                torch.cuda.synchronize()
                et = starter.elapsed_time(ender)

                avg = (running_loss / PRINT_STEPS).item()
                print(
                    f"iter {iters}, loss {avg:.4f}, t {et}, "
                    f"{gpu_temp(handle)}, {gpu_clock(handle)}, {gpu_power(handle)}, "
                    f"{et * gpu_power(handle)}"
                )

                # if gpu_power(handle) < 75: time.sleep(10)

                starter.record()

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
                        "step": iters,
                    },
                    CHECKPOINT_PATH,
                )
                print("saved checkpoint")
                starter.record()

            if iters % VALIDATE_STEPS == 0:
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
                            logits = model(seq)
                            B, T, V = logits.shape
                            loss_val = loss_f(logits.view(-1, V), target.view(-1))

                        valid_iters += 1
                        total_valid_loss += loss_val.item()

                    print(f"validation loss {total_valid_loss / valid_iters:.4f}")
                    valid_writer.write(str(total_valid_loss / valid_iters) + " \n")
                print("validation complete")
                model.train()
                starter.record()

        t1 = time.time()
        print(f"epoch {epoch} completed in {t1 - t0:.1f}")
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
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
                logits = model(seq)
                B, T, V = logits.shape
                loss_val = loss_f(logits.view(-1, V), target.view(-1))

            test_iters += 1
            total_test_loss += loss_val.item()

        print(f"test loss {total_test_loss / test_iters:.4f}")


if __name__ == "__main__":
    main()

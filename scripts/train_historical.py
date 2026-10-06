#!/usr/bin/env python3
"""Reconstructed, path-configurable two-epoch historical Hannah trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "hannah_model"))
from transformer_lm_trust import build_model  # noqa: E402


class TheoremDataset(Dataset):
    """Character-level corpus built from a deterministic sorted file list."""
    def __init__(self, data_dir: Path, seq_len: int, limit_files: int,
                 max_samples: int | None):
        files = sorted(data_dir.glob("THRM-*.yaml"))[:limit_files]
        if not files:
            raise FileNotFoundError(f"No THRM-*.yaml files found in {data_dir}")
        chunks = []
        for path in files:
            chunks.append(path.read_text(encoding="utf-8") + "\n")
        self.files = files
        self.data = "".join(chunks)
        self.chars = sorted(set(self.data))
        if not self.chars:
            raise ValueError("The selected theorem files contain no text")
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        token_ids = [self.char_to_idx[ch] for ch in self.data]
        if max_samples is not None:
            # Preserve the original demo's cap semantics: max_samples * seq_len tokens.
            token_ids = token_ids[:max_samples * seq_len]
        if len(token_ids) <= seq_len:
            raise ValueError(f"Need more than {seq_len} characters; found {len(token_ids)}")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        return self.tokens[idx:idx+self.seq_len], self.tokens[idx+1:idx+self.seq_len+1]

    def decode(self, ids):
        return "".join(self.idx_to_char[int(i)] for i in ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "historical_run")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--limit-files", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=10000,
                   help="Original cap is max_samples*seq_len tokens; use 0 for no cap")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=None,
                   help="Historical seed was not recorded; omit to use nondeterministic initialization/order")
    p.add_argument("--checkpoint-every-batches", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from an epoch-boundary checkpoint (partial-epoch replay is not exact)")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.seq_len < 1:
        p.error("epochs, batch-size, and seq-len must be positive")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_samples = args.max_samples or None
    ds = TheoremDataset(args.data_dir, args.seq_len, args.limit_files, max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(vocab_size=len(ds.chars), use_trust=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch, step = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("batch", 0) != 0:
            raise ValueError("Resume only from an epoch-boundary checkpoint (batch=0)")
        start_epoch, step = int(state["epoch"]), int(state["global_step"])
    trainable = sum(t.numel() for t in model.parameters() if t.requires_grad)
    run = {"data_dir": str(args.data_dir.resolve()), "files": [str(x) for x in ds.files],
           "file_sha256": {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in ds.files},
           "file_count": len(ds.files), "characters": len(ds.data), "tokens": len(ds.tokens),
           "vocab_size": len(ds.chars), "seq_len": args.seq_len,
           "batch_size": args.batch_size, "batches_per_epoch": len(loader),
           "epochs": args.epochs, "learning_rate": args.learning_rate,
           "seed": args.seed, "device": str(device), "parameters": trainable,
           "python": platform.python_version(), "pytorch": torch.__version__,
           "started_unix": time.time()}
    (args.output_dir / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    metrics_path = args.output_dir / "metrics.jsonl"
    with metrics_path.open("a" if args.resume else "w", encoding="utf-8") as metrics:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            losses = []
            started = time.time()
            bar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch_idx, (x, y) in enumerate(bar, 1):
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                value = float(loss.detach().cpu())
                losses.append(value)
                step += 1
                rec = {"event": "batch", "epoch": epoch+1, "batch": batch_idx,
                       "global_step": step, "loss": value, "elapsed_s": time.time()-started}
                metrics.write(json.dumps(rec) + "\n")
                bar.set_postfix(loss=f"{value:.4f}")
                if args.checkpoint_every_batches and batch_idx % args.checkpoint_every_batches == 0:
                    save_checkpoint(args.output_dir, model, optimizer, ds, epoch, batch_idx, step)
            epoch_loss = sum(losses) / len(losses)
            rec = {"event": "epoch", "epoch": epoch+1, "batches": len(losses),
                   "mean_loss": epoch_loss, "first_batch_loss": losses[0],
                   "last_batch_loss": losses[-1], "elapsed_s": time.time()-started}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            save_checkpoint(args.output_dir, model, optimizer, ds, epoch+1, 0, step)
            print(json.dumps(rec))
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(3):
            start = torch.randint(0, len(ds.chars), (1, 1), device=device)
            generated = model.generate(start, max_new_tokens=100)
            samples.append(ds.decode(generated[0].detach().cpu().tolist()))
    (args.output_dir / "generation.txt").write_text(
        "\n\n".join(f"Sample {i+1}\n{text}" for i, text in enumerate(samples)) + "\n",
        encoding="utf-8")
    run["finished_unix"] = time.time()
    (args.output_dir / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")


def save_checkpoint(out, model, optimizer, ds, epoch, batch, step):
    path = out / f"checkpoint-e{epoch:02d}-b{batch:05d}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "batch": batch, "global_step": step,
                "char_to_idx": ds.char_to_idx, "idx_to_char": ds.idx_to_char,
                "corpus_files": [str(x) for x in ds.files]}, path)
    return path


if __name__ == "__main__":
    main()

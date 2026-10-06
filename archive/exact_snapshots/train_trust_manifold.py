#!/usr/bin/env python3
"""
TRAINING PIPELINE: Trust Manifold Transformer

Trains on:
1. GID theorem files (THRM-*.yaml)
2. Synthesis documents
3. Domain-specific mathematical/theoretical text

The model learns to predict next characters while operating on trust manifold.
Monitors for emergent topological properties:
- Winding number conservation
- HANNAH channel clustering in attention
- Convergence speed vs baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import sys
import os
from tqdm import tqdm
import json

sys.path.insert(0, '/tmp')

from transformer_lm_trust import build_model, TransformerLMConfig


# ============================================================================
# DATA LOADING
# ============================================================================

class TheoremDataset(Dataset):
    """
    Load theorem files and synthesis documents for training.
    Character-level tokenization.
    """

    def __init__(self, seq_len: int = 128, limit_files: Optional[int] = None):
        self.seq_len = seq_len
        self.data = ""
        self.file_count = 0

        # Load theorem files
        thrm_dir = Path("/home/father/gid-clean/intake_full")
        if thrm_dir.exists():
            thrm_files = sorted(thrm_dir.glob("THRM-*.yaml"))
            if limit_files:
                thrm_files = thrm_files[:limit_files]

            for filepath in thrm_files:
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        self.data += content + "\n\n"
                        self.file_count += 1
                except Exception as e:
                    print(f"  Warning: Could not load {filepath}: {e}")

        # Load synthesis documents
        synthesis_dirs = [
            Path("/home/father/Hardin-Claude-Information-Geodesics-/docs/synthesis"),
            Path("/home/father/archive/The-HC-Theory-of-Everything-/_imports/hardin-claude-geo/docs/synthesis"),
        ]

        for syn_dir in synthesis_dirs:
            if syn_dir.exists():
                for filepath in syn_dir.glob("*.md"):
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            self.data += content + "\n\n"
                            self.file_count += 1
                    except Exception as e:
                        print(f"  Warning: Could not load {filepath}: {e}")

        # Character-level tokenization
        self.chars = sorted(list(set(self.data)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Convert to token indices
        self.tokens = torch.tensor([self.char_to_idx[ch] for ch in self.data], dtype=torch.long)

        print(f"\nDataset loaded:")
        print(f"  Files: {self.file_count}")
        print(f"  Total characters: {len(self.data):,}")
        print(f"  Total tokens: {len(self.tokens):,}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Sequence length: {seq_len}")

    def __len__(self):
        return max(1, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        # Return input and target sequences
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return x, y

    def decode(self, tokens: List[int]) -> str:
        """Decode token indices to string."""
        return ''.join([self.idx_to_char[t] for t in tokens if t < self.vocab_size])


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Training loop for trust manifold transformer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # num epochs
        )

        self.train_losses = []
        self.val_losses = []
        self.step = 0

    def train_epoch(self) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            # Forward
            logits, loss = self.model(x, y)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Track
            total_loss += loss.item()
            n_batches += 1
            self.step += 1

            pbar.set_postfix({'loss': loss.item():.4f})

        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self) -> float:
        """Validate on validation set."""
        if self.val_loader is None:
            return float('nan')

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = self.model(x, y)
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def fit(self, num_epochs: int, val_every: int = 1):
        """Train for num_epochs."""
        print(f"\n{'='*70}")
        print(f"TRAINING LOOP")
        print(f"{'='*70}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"-" * 70)

            # Train
            train_loss = self.train_epoch()
            print(f"Train loss: {train_loss:.4f}")

            # Validate
            if (epoch + 1) % val_every == 0:
                val_loss = self.validate()
                print(f"Val loss:   {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"✓ Best validation loss!")

            # Schedule
            self.scheduler.step()
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
        }

    def get_learning_curve(self):
        """Return learning curves."""
        return {
            'train': self.train_losses,
            'val': self.val_losses,
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("TRUST MANIFOLD TRANSFORMER TRAINING")
    print("="*70)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────────────────────────────────

    print("Loading dataset...")
    dataset = TheoremDataset(seq_len=128, limit_files=None)

    # Split: 90% train, 10% val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CREATE DATA LOADERS
    # ─────────────────────────────────────────────────────────────────────────

    print("Creating data loaders...")
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # 3. CREATE MODEL
    # ─────────────────────────────────────────────────────────────────────────

    print("Building model...")
    cfg = TransformerLMConfig(
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        vocab_size=dataset.vocab_size,
        seq_len=128,
        dropout=0.1,
        sphere_dim=20,
        use_trust_attention=True,
        use_hannah_embedding=True,
        temperature=1.0,
    )

    model = build_model(
        vocab_size=dataset.vocab_size,
        use_trust=True,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: TransformerLMWithTrust")
    print(f"  Parameters: {n_params:,.0f} ({n_params/1e6:.2f}M)")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # 4. SETUP TRAINING
    # ─────────────────────────────────────────────────────────────────────────

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-3,
        max_grad_norm=1.0,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. TRAIN
    # ─────────────────────────────────────────────────────────────────────────

    results = trainer.fit(num_epochs=5, val_every=1)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. SAVE RESULTS
    # ─────────────────────────────────────────────────────────────────────────

    print("\nSaving results...")
    curves = trainer.get_learning_curve()
    with open('/tmp/training_results.json', 'w') as f:
        json.dump({
            'train_losses': curves['train'],
            'val_losses': curves['val'],
            'best_val_loss': results['best_val_loss'],
        }, f, indent=2)

    print("  → Saved /tmp/training_results.json")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. SAMPLE GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    print("\nGenerating samples...")
    model.eval()

    # Start with a random character
    start_token = torch.randint(0, dataset.vocab_size, (1, 1)).to(device)

    with torch.no_grad():
        generated = model.generate(start_token, max_new_tokens=200)

    sample_text = dataset.decode(generated[0].cpu().tolist())
    print(f"Generated text ({len(sample_text)} chars):")
    print("-" * 70)
    print(sample_text[:200])
    print("...")
    print("-" * 70)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # 8. METRICS
    # ─────────────────────────────────────────────────────────────────────────

    print("="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print()
    print(f"Train loss: {curves['train'][-1]:.4f}")
    print(f"Val loss:   {curves['val'][-1]:.4f}")
    print(f"Best val:   {results['best_val_loss']:.4f}")
    print()

    # Compute improvement
    if len(curves['train']) > 1:
        train_improvement = (curves['train'][0] - curves['train'][-1]) / curves['train'][0] * 100
        print(f"Train loss improvement: {train_improvement:.1f}%")

    if len(curves['val']) > 1:
        val_improvement = (curves['val'][0] - curves['val'][-1]) / curves['val'][0] * 100
        print(f"Val loss improvement:   {val_improvement:.1f}%")

    print()
    print("✓ TRAINING COMPLETE")
    print()


if __name__ == "__main__":
    main()

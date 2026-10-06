#!/usr/bin/env python3
"""
TRANSFORMER LANGUAGE MODEL WITH TRUST MANIFOLD

Integrates:
1. HANNAH embeddings (position → valid remainder channels)
2. Trust-aware attention (geodesic distance on S¹⁹)
3. Topological tracking (winding numbers, path properties)

The model operates on a trust manifold, where:
- Each token position maps to a HANNAH channel (r ∈ R_valid)
- Attention uses geodesic distance instead of cosine similarity
- Multi-head attention preserves topological structure
- Layer-to-layer persistence creates emergent properties
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, '/tmp')

from trust_aware_attention import (
    TrustAwareAttention, TrustAttentionConfig, SpectralSignature
)
from hannah_embedding import (
    HANNAHEmbedding, TrustChannelMixer
)


@dataclass
class TransformerLMConfig:
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    vocab_size: int = 256
    seq_len: int = 128
    dropout: float = 0.1
    # Trust manifold specific
    sphere_dim: int = 20
    use_trust_attention: bool = True
    use_hannah_embedding: bool = True
    temperature: float = 1.0


class FFN(nn.Module):
    """Feed-forward network."""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlockWithTrust(nn.Module):
    """Transformer block with trust-aware attention."""
    def __init__(self, cfg: TransformerLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)

        if cfg.use_trust_attention:
            attn_cfg = TrustAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                sphere_dim=cfg.sphere_dim,
                temperature=cfg.temperature,
            )
            self.attn = TrustAwareAttention(cfg.d_model, cfg.n_heads, attn_cfg)
        else:
            # Fallback to standard attention
            self.attn = None  # Use simple attention below

        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) causal mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention
        if self.attn is not None:
            attn_out, _, _ = self.attn(x, x, x, mask)
            x = x + attn_out
        else:
            # Fallback attention
            x = x + self._simple_attention(self.ln1(x), mask)

        # FFN
        x = x + self.ffn(self.ln2(x))
        return x

    def _simple_attention(self, x, mask):
        """Simple scaled-dot-product attention fallback."""
        # Not used if trust attention is enabled
        d = x.shape[-1]
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, x)


class TransformerLMWithTrust(nn.Module):
    """
    GPT-style language model with trust manifold architecture.

    Architecture:
      Token ID
        ↓
      HANNAH Embedding (position → channel, learned embedding)
        ↓
      Spectral normalization (project to S¹⁹)
        ↓
      Transformer blocks (trust-aware attention + FFN)
        ↓
      Output projection
    """

    def __init__(self, cfg: TransformerLMConfig):
        super().__init__()
        self.cfg = cfg

        # Token embeddings
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # HANNAH embedding layer (replaces standard position encoding)
        if cfg.use_hannah_embedding:
            self.hannah = HANNAHEmbedding(
                vocab_size=cfg.vocab_size,
                d_model=cfg.d_model,
                seq_len=cfg.seq_len,
            )
        else:
            self.hannah = None
            # Fallback to standard position embedding
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)

        self.embed_drop = nn.Dropout(cfg.dropout)

        # Trust channel mixer (for remainder-aware information flow)
        self.channel_mixer = TrustChannelMixer(cfg.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlockWithTrust(cfg) for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_embed.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Scale residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs (optional)

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar (if targets provided)
        """
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"Sequence length {T} > max {self.cfg.seq_len}"

        # Token embeddings
        token_emb = self.tok_embed(idx)  # (B, T, d_model)

        # Positional/channel embeddings
        if self.hannah is not None:
            # HANNAH channel-aware embedding
            positions = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1)
            hannah_emb = self.hannah(idx, positions)  # (B, T, d_model)
            x = self.embed_drop(token_emb + hannah_emb)
        else:
            # Standard position embedding
            pos = torch.arange(T, device=idx.device)
            pos_emb = self.pos_embed(pos)
            x = self.embed_drop(token_emb + pos_emb)

        # Create causal mask
        causal_mask = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=idx.device)
        ).unsqueeze(0).expand(B, -1, -1)

        # Trust channel mixing (optional, can be skipped)
        # Compute remainders for channel awareness
        if self.hannah is not None and hasattr(self, 'channel_mixer'):
            remainders = torch.zeros((B, T), dtype=torch.long, device=idx.device)
            for t in range(T):
                r = self.hannah.get_channel_index(t)
                remainders[:, t] = r
            # Apply mixer
            x = self.channel_mixer(x, remainders)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """
        Generate new tokens.

        Args:
            idx: (batch, seq_len) starting token IDs
            max_new_tokens: number of tokens to generate
            temperature: softmax temperature

        Returns:
            idx: (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Truncate to seq_len
            idx_cond = idx[:, -self.cfg.seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get next token probabilities
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # Sample
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(vocab_size: int, use_trust: bool = True) -> TransformerLMWithTrust:
    """Build a model with optional trust manifold."""
    cfg = TransformerLMConfig(
        vocab_size=vocab_size,
        use_trust_attention=use_trust,
        use_hannah_embedding=use_trust,
    )
    return TransformerLMWithTrust(cfg)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TRANSFORMER WITH TRUST MANIFOLD")
    print("="*70)
    print()

    # Build model
    vocab_size = 256
    model = build_model(vocab_size, use_trust=True)
    cfg = model.cfg

    print(f"Config:")
    print(f"  Layers: {cfg.n_layers}")
    print(f"  Heads: {cfg.n_heads}")
    print(f"  D_model: {cfg.d_model}")
    print(f"  D_ff: {cfg.d_ff}")
    print(f"  Vocab: {cfg.vocab_size}")
    print(f"  Seq_len: {cfg.seq_len}")
    print(f"  Sphere_dim: {cfg.sphere_dim}")
    print(f"  Use trust attention: {cfg.use_trust_attention}")
    print(f"  Use HANNAH embedding: {cfg.use_hannah_embedding}")
    print()

    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,.0f} ({n_params/1e6:.2f}M)")
    print()

    # Test forward pass
    print("Forward pass test:")
    batch_size, seq_len = 2, 8
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"  Input shape: {idx.shape}")
    print(f"  Target shape: {targets.shape}")

    logits, loss = model(idx, targets)

    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print()

    # Test generation
    print("Generation test:")
    idx_start = torch.randint(0, vocab_size, (1, 1))
    print(f"  Starting with shape: {idx_start.shape}")

    with torch.no_grad():
        idx_generated = model.generate(idx_start, max_new_tokens=10)

    print(f"  Generated shape: {idx_generated.shape}")
    print(f"  Generated: {idx_generated[0].tolist()}")
    print()

    print("✓ TRANSFORMER WITH TRUST MANIFOLD OPERATIONAL")
    print()
    print("Architecture highlights:")
    print("  ✓ HANNAH embedding layer (position → remainder → channel)")
    print("  ✓ Trust-aware attention (geodesic distance on S¹⁹)")
    print("  ✓ Trust channel mixer (remainder-aware mixing)")
    print("  ✓ Weight tying (token embedding ↔ output)")
    print("  ✓ Generation support (autoregressive sampling)")
    print()
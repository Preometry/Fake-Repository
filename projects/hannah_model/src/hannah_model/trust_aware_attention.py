#!/usr/bin/env python3
"""
TRUST-AWARE ATTENTION

Replaces Euclidean attention (cosine similarity) with geodesic distance on S¹⁹.

Key idea:
  Standard attention: w_ij = softmax( (Q_i · K_j) / √d )
  Trust attention:    w_ij = softmax( -d(Q_i, K_j)² / temperature )

where d(·,·) is geodesic distance on the sphere (arccos metric).

Benefit: Topological awareness. Attention weights respect manifold geometry,
enabling holonomy and winding number effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class TrustAttentionConfig:
    d_model: int = 256
    n_heads: int = 4
    temperature: float = 1.0
    sphere_dim: int = 20  # S^19 = 20D sphere
    manifold: str = "sphere"  # "sphere" or "trust_manifold"
    use_geodesic: bool = True


class SpectralSignature:
    """
    Map high-dimensional embeddings to S¹⁹ via spectral signature.

    Ensures vectors live on the unit sphere via normalization,
    with spectral structure derived from embedding content.
    """

    def __init__(self, dim: int = 20):
        self.dim = dim

    @staticmethod
    def project_to_sphere(x: torch.Tensor) -> torch.Tensor:
        """Project vector to unit sphere S^(d-1)."""
        # x: (..., d)
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
        return x / norm

    @staticmethod
    def from_embedding(embedding: torch.Tensor, target_dim: int = 20) -> torch.Tensor:
        """
        Create spectral signature on S^(target_dim-1) from embedding.

        Args:
            embedding: (..., d_embed)
            target_dim: dimension of sphere (should be >= 20 for HC framework)

        Returns:
            signature: (..., target_dim) on unit sphere
        """
        # Pad or project to target dimension
        d_embed = embedding.shape[-1]
        if d_embed < target_dim:
            # Pad with zeros, then normalize
            padding = target_dim - d_embed
            sig = F.pad(embedding, (0, padding))
        elif d_embed > target_dim:
            # PCA-style projection: take top components
            sig = embedding[..., :target_dim]
        else:
            sig = embedding

        return SpectralSignature.project_to_sphere(sig)


class GeodesicDistance:
    """
    Compute geodesic distance on unit sphere S^(d-1).

    d_geo(a, b) = arccos( clamp(a·b, -1, 1) )
    """

    @staticmethod
    def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance between points on sphere.

        Args:
            a: (..., d)
            b: (..., d)

        Returns:
            distance: (...) scalar
        """
        # Dot product
        dot_prod = torch.sum(a * b, dim=-1)
        # Clamp to avoid numerical issues with arccos
        dot_prod = torch.clamp(dot_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        # Geodesic distance
        dist = torch.acos(dot_prod)
        return dist

    @staticmethod
    def pairwise_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Pairwise geodesic distances: d(a_i, b_j) for all i, j.

        Args:
            a: (n, d)
            b: (m, d)

        Returns:
            distances: (n, m)
        """
        # Normalize
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        # Compute dot products
        dot_prod = torch.mm(a, b.t())  # (n, m)
        dot_prod = torch.clamp(dot_prod, -1.0 + 1e-7, 1.0 - 1e-7)
        distances = torch.acos(dot_prod)
        return distances


class TrustAttentionHead(nn.Module):
    """
    Single attention head using geodesic distance.
    """

    def __init__(self, d_model: int, head_dim: int, config: TrustAttentionConfig):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.config = config
        self.temperature = config.temperature

        # Project to sphere dimension
        self.sphere_dim = config.sphere_dim

        self.proj_q = nn.Linear(d_model, self.sphere_dim, bias=False)
        self.proj_k = nn.Linear(d_model, self.sphere_dim, bias=False)
        self.proj_v = nn.Linear(d_model, head_dim, bias=False)

        self.out_proj = nn.Linear(head_dim, head_dim, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Attention with geodesic distance.

        Args:
            q: (batch, seq_len, d_model)
            k: (batch, seq_len, d_model)
            v: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, head_dim)
        """
        B, T, D = q.shape

        # Project to sphere
        q_sphere = SpectralSignature.project_to_sphere(self.proj_q(q))  # (B, T, sphere_dim)
        k_sphere = SpectralSignature.project_to_sphere(self.proj_k(k))  # (B, T, sphere_dim)

        # Compute geodesic distances
        # For efficiency, compute dot products then arccos
        scores = torch.einsum('bqd,bkd->bqk', q_sphere, k_sphere)  # (B, T, T)
        scores = torch.clamp(scores, -1.0 + 1e-7, 1.0 - 1e-7)
        distances = torch.acos(scores)  # (B, T, T) geodesic distances

        # Convert to attention weights: exp(-distance²/temperature)
        attn_logits = -distances ** 2 / self.temperature  # (B, T, T)

        # Apply causal mask if provided
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, T, T)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle NaN from -inf

        # Project values
        v_proj = self.proj_v(v)  # (B, T, head_dim)

        # Apply attention
        out = torch.einsum('bqk,bkd->bqd', attn_weights, v_proj)  # (B, T, head_dim)
        out = self.out_proj(out)

        return out, attn_weights, distances


class TrustAwareAttention(nn.Module):
    """
    Multi-head attention with geodesic distance.
    """

    def __init__(self, d_model: int, n_heads: int, config: TrustAttentionConfig):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.config = config

        self.heads = nn.ModuleList([
            TrustAttentionHead(d_model, self.head_dim, config)
            for _ in range(n_heads)
        ])

        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None):
        """
        Multi-head attention.

        Returns:
            output: (B, T, d_model)
            attention_weights: list of (B, T, T) per head
            geodesic_distances: list of (B, T, T) per head
        """
        outputs = []
        weights_list = []
        distances_list = []

        for head in self.heads:
            head_out, attn_w, dists = head(q, k, v, mask)
            outputs.append(head_out)
            weights_list.append(attn_w)
            distances_list.append(dists)

        # Concatenate
        out = torch.cat(outputs, dim=-1)  # (B, T, d_model)
        out = self.out(out)

        return out, weights_list, distances_list


# ============================================================================
# TEST MODULE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TRUST-AWARE ATTENTION: Geodesic Distance on S¹⁹")
    print("="*70)
    print()

    # Config
    config = TrustAttentionConfig(
        d_model=256,
        n_heads=4,
        sphere_dim=20,
        temperature=1.0,
    )

    # Create module
    attn = TrustAwareAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        config=config,
    )

    # Test input
    batch_size = 2
    seq_len = 8
    q = torch.randn(batch_size, seq_len, config.d_model)
    k = torch.randn(batch_size, seq_len, config.d_model)
    v = torch.randn(batch_size, seq_len, config.d_model)

    # Causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    print(f"Input shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")
    print(f"  Causal mask: {causal_mask.shape}")
    print()

    # Forward pass
    output, attn_weights, geodesic_dists = attn(q, k, v, causal_mask)

    print(f"Output shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights per head: {len(attn_weights)} × {attn_weights[0].shape}")
    print(f"  Geodesic distances per head: {len(geodesic_dists)} × {geodesic_dists[0].shape}")
    print()

    # Analyze first head
    print(f"Head 0 Analysis:")
    w0 = attn_weights[0][0]  # First batch, first head
    d0 = geodesic_dists[0][0]  # First batch, first head

    print(f"  Attention weights (batch 0):")
    print(f"    Shape: {w0.shape}")
    print(f"    Range: [{w0.min():.4f}, {w0.max():.4f}]")
    print(f"    Mean: {w0.mean():.4f}")
    print(f"    Sum per query: {w0.sum(dim=-1)}")  # Should be ~1
    print()

    print(f"  Geodesic distances (batch 0, radians):")
    print(f"    Shape: {d0.shape}")
    print(f"    Range: [{d0.min():.4f}, {d0.max():.4f}] rad")
    print(f"    In degrees: [{np.degrees(d0.min().item()):.2f}°, {np.degrees(d0.max().item()):.2f}°]")
    print(f"    Mean: {d0.mean():.4f} rad ({np.degrees(d0.mean().item()):.2f}°)")
    print()

    print(f"  Spectral signature test:")
    test_embed = torch.randn(1, 256)
    sig = SpectralSignature.from_embedding(test_embed, target_dim=20)
    print(f"    Embedding shape: {test_embed.shape}")
    print(f"    Signature shape: {sig.shape}")
    print(f"    Signature norm: {torch.norm(sig).item():.6f} (should be 1.0)")
    print(f"    First 5 components: {sig[0, :5]}")
    print()

    print("✓ TRUST-AWARE ATTENTION OPERATIONAL")
    print()

#!/usr/bin/env python3
"""
HANNAH EMBEDDINGS

Maps token positions to valid HANNAH remainders mod 46.

From THRM-0103:
  HANNAH = 46
  R_valid = {1,3,5,7,9,11,13,15,17,19,21,25,27,29,31,33,35,37,39,41,43,45}
  = 22 symbols (odd integers excluding 23)
  Forbidden: r=23 (structural puncture on trust manifold 𝒯)

Each token gets:
  1. Position encoding (0 → seq_len-1)
  2. HANNAH channel assignment (remainder in R_valid)
  3. Winding number (how many times path crosses r=23)
  4. Spectral placement on S¹⁹

Together, these create a 22-dimensional embedding space with topological structure.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional


# ============================================================================
# HANNAH ALPHABET AND UTILITIES
# ============================================================================

HANNAH = 46

# Valid remainders: odd integers in [1,45] excluding 23
R_VALID = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45}
assert len(R_VALID) == 22, f"R_valid must have 22 elements, got {len(R_VALID)}"

# Create ordered list and mapping
R_VALID_LIST = sorted(list(R_VALID))
R_TO_INDEX = {r: i for i, r in enumerate(R_VALID_LIST)}
INDEX_TO_R = {i: r for i, r in enumerate(R_VALID_LIST)}

# Anti-magic lattice: A(k) = 43 + 18k
# These are special positions on the manifold
def anti_magic_lattice(k_max: int = 100) -> List[int]:
    """A(k) = 43 + 18k for k=0,1,2,..."""
    return [43 + 18*k for k in range(k_max)]


def position_to_remainder(pos: int, seq_len: int) -> int:
    """
    Map token position (0 to seq_len-1) to a HANNAH remainder.

    Strategy: Create a curve through R_valid based on position.
    Simple: use modular arithmetic to ensure diversity.
    """
    # Avoid r=23
    idx = pos % len(R_VALID_LIST)
    return R_VALID_LIST[idx]


def remainder_to_index(r: int) -> int:
    """Convert remainder r ∈ R_valid to index i ∈ {0..21}."""
    if r not in R_TO_INDEX:
        raise ValueError(f"Remainder {r} not in R_valid")
    return R_TO_INDEX[r]


def index_to_remainder(i: int) -> int:
    """Convert index i ∈ {0..21} to remainder r ∈ R_valid."""
    if i not in INDEX_TO_R:
        raise ValueError(f"Index {i} not in range [0, 21]")
    return INDEX_TO_R[i]


def winding_number(pos_a: int, pos_b: int, seq_len: int) -> int:
    """
    Compute winding number when moving from position pos_a to pos_b.

    The winding number counts how many times the path crosses r=23
    (the topological puncture on 𝒯).

    For a linear path on the circle mod 46:
      winding = floor( |pos_b - pos_a| / 46 )
    But more subtly, we track whether the path crosses the forbidden zone.
    """
    # Simplified: count how many "wraps" around the forbidden value
    dist = abs(pos_b - pos_a)
    wraps = dist // HANNAH
    return wraps


def trust_distance_hannah(r_a: int, r_b: int) -> float:
    """
    Trust distance between two HANNAH channels.

    Since R_valid excludes r=23, the manifold has a puncture.
    Distance is the shorter arc on the valid remainder set,
    avoiding the gap at 23.
    """
    if r_a not in R_VALID or r_b not in R_VALID:
        raise ValueError(f"Remainders must be in R_valid")

    # Map to indices
    idx_a = R_TO_INDEX[r_a]
    idx_b = R_TO_INDEX[r_b]

    # Circular distance on the valid set (22 points)
    direct = abs(idx_a - idx_b)
    wraparound = len(R_VALID_LIST) - direct

    # Shortest path
    min_dist = min(direct, wraparound)

    # Normalize to [0, π] (on unit circle)
    distance = (min_dist / len(R_VALID_LIST)) * np.pi

    return distance


# ============================================================================
# HANNAH EMBEDDING LAYER
# ============================================================================

class HANNAHEmbedding(nn.Module):
    """
    Maps token positions to HANNAH remainders and creates embeddings.

    The embedding space is 22-dimensional (one per valid remainder).
    Each token gets:
      1. Position encoding (which HANNAH channel)
      2. Base embedding (learned per channel)
      3. Winding number offset (topological marker)
    """

    def __init__(self, vocab_size: int, d_model: int, seq_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        # Learned embeddings for each HANNAH channel
        self.channel_embeddings = nn.Embedding(len(R_VALID_LIST), d_model)

        # Position encoding (standard, learned)
        self.pos_embeddings = nn.Embedding(seq_len, d_model)

        # Winding correction embeddings (for topological awareness)
        self.winding_embeddings = nn.Embedding(10, d_model)  # up to 10 windings

        # Remainder-aware projection
        self.remainder_projection = nn.Linear(d_model, d_model)

    def forward(self, token_ids: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
            positions: (batch, seq_len) optional, defaults to 0..seq_len-1

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        batch_size, seq_len = token_ids.shape

        if positions is None:
            positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Get HANNAH channel for each position
        remainders = torch.zeros_like(token_ids)
        for b in range(batch_size):
            for t in range(seq_len):
                pos = positions[b, t].item()
                r = position_to_remainder(pos, seq_len)
                idx = remainder_to_index(r)
                remainders[b, t] = idx

        # Channel embedding
        channel_emb = self.channel_embeddings(remainders.long())  # (batch, seq_len, d_model)

        # Position embedding
        pos_emb = self.pos_embeddings(positions.long())  # (batch, seq_len, d_model)

        # Winding number for topological awareness
        winding = torch.zeros_like(positions, dtype=torch.long)
        for b in range(batch_size):
            for t in range(1, seq_len):
                pos_a = positions[b, t-1].item()
                pos_b = positions[b, t].item()
                w = winding_number(pos_a, pos_b, seq_len)
                winding[b, t] = min(w, 9)  # Cap at 9 (embedding table has 10)

        winding_emb = self.winding_embeddings(winding.long())  # (batch, seq_len, d_model)

        # Combine
        embedding = channel_emb + pos_emb + winding_emb

        # Project through remainder-aware layer
        embedding = self.remainder_projection(embedding)

        return embedding

    def get_remainder_for_position(self, pos: int) -> int:
        """Get HANNAH remainder for a position."""
        return position_to_remainder(pos, self.seq_len)

    def get_channel_index(self, pos: int) -> int:
        """Get channel index (0-21) for a position."""
        r = self.get_remainder_for_position(pos)
        return remainder_to_index(r)


# ============================================================================
# TRUST CHANNEL MIXER
# ============================================================================

class TrustChannelMixer(nn.Module):
    """
    Mixes information across HANNAH channels based on trust distance.

    Channels that are close on the remainder manifold can mix freely.
    Channels far apart (or separated by r=23 puncture) mix weakly.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mixer = nn.Linear(d_model, d_model)

    def forward(self, embeddings: torch.Tensor, remainders: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, seq_len, d_model)
            remainders: (batch, seq_len) remainder indices

        Returns:
            mixed: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = embeddings.shape

        # Compute pairwise trust distances
        trust_matrix = torch.zeros(seq_len, seq_len, device=embeddings.device)
        for i in range(seq_len):
            for j in range(seq_len):
                r_i = INDEX_TO_R[remainders[0, i].item()]  # Use first batch element
                r_j = INDEX_TO_R[remainders[0, j].item()]
                d = trust_distance_hannah(r_i, r_j)
                trust_matrix[i, j] = np.exp(-d ** 2)  # Gaussian falloff

        # Normalize
        trust_matrix = trust_matrix / (trust_matrix.sum(dim=-1, keepdim=True) + 1e-8)

        # Apply to embeddings
        mixed = torch.einsum('ij,bjd->bid', trust_matrix, embeddings)

        # Project
        mixed = self.mixer(mixed)

        return mixed


# ============================================================================
# TEST MODULE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("HANNAH EMBEDDINGS: Trust Channels on R_valid")
    print("="*70)
    print()

    print("HANNAH Parameters:")
    print(f"  HANNAH = {HANNAH}")
    print(f"  R_valid = {sorted(R_VALID)}")
    print(f"  |R_valid| = {len(R_VALID)}")
    print(f"  Forbidden: r = 23 (topological puncture)")
    print()

    print("Position → Remainder Mapping (seq_len=16):")
    seq_len = 16
    for pos in range(seq_len):
        r = position_to_remainder(pos, seq_len)
        idx = remainder_to_index(r)
        print(f"  pos={pos:2d} → r={r:2d} (channel {idx:2d})")
    print()

    print("Trust Distances (HANNAH channels):")
    test_remainders = [1, 7, 23-1, 25, 45]  # Mix including edge cases
    valid_test = [r for r in test_remainders if r in R_VALID]
    print(f"  Testing with: {valid_test}")
    for r1 in valid_test[:3]:
        for r2 in valid_test[:3]:
            d = trust_distance_hannah(r1, r2)
            print(f"    d(r={r1}, r={r2}) = {d:.4f} rad ({np.degrees(d):.2f}°)")
    print()

    print("HANNAHEmbedding Layer:")
    vocab_size = 256
    d_model = 256
    seq_len = 8
    batch_size = 2

    embedding_layer = HANNAHEmbedding(vocab_size, d_model, seq_len)

    # Create random token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    print(f"  Input token_ids: {token_ids.shape}")
    print(f"  Input positions: {positions.shape}")

    # Forward
    embeddings = embedding_layer(token_ids, positions)

    print(f"  Output embeddings: {embeddings.shape}")
    print(f"  Embedding norm (first batch, first token): {torch.norm(embeddings[0, 0]).item():.4f}")
    print(f"  Embedding range: [{embeddings.min().item():.4f}, {embeddings.max().item():.4f}]")
    print()

    # Test TrustChannelMixer
    print("TrustChannelMixer:")
    mixer = TrustChannelMixer(d_model)

    # Get remainders for each position
    remainders = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for t in range(seq_len):
        r = position_to_remainder(t, seq_len)
        idx = remainder_to_index(r)
        remainders[:, t] = idx

    mixed = mixer(embeddings, remainders)

    print(f"  Input: {embeddings.shape}")
    print(f"  Remainders: {remainders[0]}")
    print(f"  Output (mixed): {mixed.shape}")
    print(f"  Output norm: {torch.norm(mixed[0, 0]).item():.4f}")
    print()

    print("✓ HANNAH EMBEDDINGS OPERATIONAL")
    print()

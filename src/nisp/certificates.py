"""Merkle root over a fixed set of leaves — pure hashlib, no external deps.

Used by PriorityArbiter to produce a 4-leaf Merkle root over (rho, D, facts, Z).
A safety officer can independently verify any verdict by re-hashing the four
logged leaves and comparing to the stored merkle_root.

Tree layout (leaves l0..l3, internal nodes h01, h23, root):

    root
   /    \\
 h01    h23
 / \\    / \\
l0  l1  l2  l3
"""
import hashlib
from typing import Sequence


def _h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def merkle_root(leaves: Sequence[bytes]) -> str:
    """Compute SHA-256 Merkle root over an even number of byte leaves.

    Always pads to the next power of two by duplicating the last leaf.
    Returns a 64-character lowercase hex string.
    """
    if not leaves:
        return hashlib.sha256(b"").hexdigest()

    layer = [_h(leaf) for leaf in leaves]

    # Pad to even length at each level
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [_h(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]

    return layer[0].hex()


def verdict_merkle_root(rho: float, D: float,
                        facts_tuple: tuple, Z_tuple: tuple) -> str:
    """Compute the canonical Merkle root for one NISP verdict.

    Four leaves, fixed order:
      0  rho   — proof completeness score
      1  D     — neural-physical KL divergence
      2  facts — tuple of (proposition, value, reading, threshold) strings
      3  Z     — predicate distribution tuple

    Deterministic: same inputs always produce the same root.
    """
    def _enc(x) -> bytes:
        return str(x).encode()

    leaves = [
        _enc(f"{rho:.15f}"),
        _enc(f"{D:.15f}"),
        _enc(facts_tuple),
        _enc(Z_tuple),
    ]
    return merkle_root(leaves)

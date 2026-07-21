"""Phase N-1b — determinism of L2_Nengo across runs with the same seed.

Nengo's decoder cache lives on disk and can introduce non-determinism on
Windows (different timestamps, partial writes). Two L2_Nengo instances built
with seed=42 against the same latent set must produce bitwise-identical Z
vectors for the same input.
"""
from pathlib import Path

import numpy as np
import pytest

from src.nisp.core.contracts import LatentVector, RiskState
from src.nisp.layers.l2_nengo import L2_Nengo


_LATENT_DIR = Path("artifacts/latents")


def _load_latent(cycle_id: int = 0) -> LatentVector:
    path = _LATENT_DIR / f"cycle_{cycle_id}_difumo.npy"
    if not path.exists():
        pytest.skip(f"Precomputed latent {path} missing.")
    return LatentVector(
        E=np.load(path).astype(np.float32),
        risk_state=RiskState.NOMINAL,
        timestamp=0.0,
    )


def test_l2_nengo_determinism_same_seed():
    """Two L2_Nengo instances with seed=42 produce identical Z on same input."""
    latent = _load_latent(cycle_id=0)

    l2_a = L2_Nengo(seed=42)
    z_a = np.array(l2_a.process(latent).Z)

    l2_b = L2_Nengo(seed=42)
    z_b = np.array(l2_b.process(latent).Z)

    np.testing.assert_allclose(
        z_a, z_b, rtol=1e-7, atol=1e-9,
        err_msg=f"Same-seed Z vectors differ:\n  a={z_a}\n  b={z_b}\n  delta={z_a - z_b}",
    )


def test_l2_nengo_determinism_repeated_calls():
    """Same L2_Nengo instance, same input twice -> same Z (sim.reset() works)."""
    latent = _load_latent(cycle_id=0)
    l2 = L2_Nengo(seed=42)

    z_first = np.array(l2.process(latent).Z)
    z_second = np.array(l2.process(latent).Z)

    np.testing.assert_allclose(
        z_first, z_second, rtol=1e-7, atol=1e-9,
        err_msg="Repeated process() on same instance must give same Z",
    )

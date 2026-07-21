"""Phase A-1 — L1_TribeV2_Real file-lookup loader tests."""
from pathlib import Path

import numpy as np
import pytest

from src.nisp.core.contracts import RiskState, TelemetryState
from src.nisp.layers.l1_tribev2_real import L1_TribeV2_Real


def _telemetry(sim_time: float = 0.0, min_dist: float = 1.5,
               tcp_velocity: float = 0.05) -> TelemetryState:
    return TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6),
        ee_pos=np.array([0.3, 0.0, 0.9]),
        min_dist=min_dist, zone_status=0,
        sim_time=sim_time, tcp_velocity=tcp_velocity,
    )


def test_loads_precomputed_latent():
    """sim_time=0 -> cycle_0_difumo.npy. Output is 1024-dim float32."""
    latent_dir = Path("artifacts/latents")
    if not (latent_dir / "cycle_0_difumo.npy").exists():
        pytest.skip("artifacts/latents/cycle_0_difumo.npy missing.")

    l1 = L1_TribeV2_Real(latent_dir=latent_dir)
    lv = l1.process(_telemetry(sim_time=0.0))

    E = np.array(lv.E)
    assert E.shape == (1024,)
    assert E.dtype in (np.float32, np.float64)


def test_sim_time_to_cycle_id_wraps():
    """With 15 cached cycles and dt=0.02, sim_time=0.30 -> cycle 15 mod 15 = 0."""
    latent_dir = Path("artifacts/latents")
    if not (latent_dir / "cycle_0_difumo.npy").exists():
        pytest.skip("artifacts/latents/cycle_0_difumo.npy missing.")

    l1 = L1_TribeV2_Real(latent_dir=latent_dir)
    lv_a = l1.process(_telemetry(sim_time=0.00))
    lv_b = l1.process(_telemetry(sim_time=0.30))   # 15 * 0.02 -> wraps to 0
    np.testing.assert_array_equal(np.array(lv_a.E), np.array(lv_b.E))


def test_risk_state_inference():
    """Heuristic risk_state should match telemetry severity."""
    latent_dir = Path("artifacts/latents")
    if not (latent_dir / "cycle_0_difumo.npy").exists():
        pytest.skip("latents missing.")

    l1 = L1_TribeV2_Real(latent_dir=latent_dir)
    assert l1.process(_telemetry(min_dist=1.5)).risk_state == RiskState.NOMINAL
    assert l1.process(_telemetry(min_dist=0.6)).risk_state == RiskState.HESITANT
    assert l1.process(_telemetry(min_dist=0.3)).risk_state == RiskState.CRITICAL


def test_fallback_when_latent_missing(tmp_path):
    """Missing .npy + fallback L1 -> uses fallback. No fallback -> raises."""
    # Empty dir, no fallback -> raises.
    l1 = L1_TribeV2_Real(latent_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        l1.process(_telemetry())

    # Same dir, fallback to TribevX -> works.
    from src.nisp.layers.l1_tribevx import TribevX
    l1_with_fb = L1_TribeV2_Real(latent_dir=tmp_path, fallback=TribevX(seed=0))
    lv = l1_with_fb.process(_telemetry())
    # TribevX outputs P=200 not 1024; the fallback path is allowed to break the
    # 1024-dim contract because the demo will route to L2_PredicateDecoder(p_dim=200) in
    # that case. We only check that we got *something* back.
    assert lv is not None

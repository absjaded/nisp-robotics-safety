"""Phase N-1a — pipeline-level smoke test for L2_Nengo.

Verifies the PCA dimension fix (k = min(_PCA_DIMS, n_latents-1) = 14 for our
15 precomputed latents) survives a full NISPPipeline.run_cycle() call.

We can't reuse `_build_pipeline` from test_pipeline.py because it uses
TribevX (200-dim output) and L2_Nengo expects 1024-dim DiFuMo latents.
Instead, we wire a tiny stub L1 that loads a precomputed cycle_*_difumo.npy
from disk — this is the same pattern L1_TribeV2_Real will use in Phase A-1.
"""
from pathlib import Path

import numpy as np
import pytest

from src.nisp.core.contracts import LatentVector, RiskState, TelemetryState
from src.nisp.core.orchestrator import NISPPipeline
from src.nisp.layers.evidence_solver import EvidenceSolver
from src.nisp.layers.l2_nengo import L2_Nengo
from src.nisp.layers.l4_bridge import LeanVerifier
from src.nisp.layers.priority_arbiter import PriorityArbiter


_LATENT_DIR = Path("artifacts/latents")


class _StubFileLookupL1:
    """Minimal L1 that reads a precomputed DiFuMo latent. Same idea as the
    forthcoming L1_TribeV2_Real (Phase A-1) but inlined here so this test
    doesn't depend on a module that hasn't been written yet."""

    def __init__(self, cycle_id: int = 0, risk_state: RiskState = RiskState.NOMINAL):
        path = _LATENT_DIR / f"cycle_{cycle_id}_difumo.npy"
        if not path.exists():
            pytest.skip(f"Precomputed latent {path} missing — run "
                        "scripts/precompute_latents.py --mock-tribe first.")
        self._E = np.load(path).astype(np.float32)
        self._risk_state = risk_state
        self.P = self._E.shape[0]  # 1024

    def process(self, telemetry: TelemetryState) -> LatentVector:
        return LatentVector(
            E=self._E,
            risk_state=self._risk_state,
            timestamp=telemetry.sim_time,
        )


def _nominal_telemetry() -> TelemetryState:
    return TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6),
        ee_pos=np.array([0.3, 0.0, 0.9]),
        min_dist=1.5, zone_status=0, sim_time=0.0, tcp_velocity=0.05,
    )


def _build_pipeline_with_nengo(cycle_id: int = 0) -> NISPPipeline:
    l1 = _StubFileLookupL1(cycle_id=cycle_id)
    l2 = L2_Nengo()  # builds Nengo simulator (~300ms one-time)
    l3 = EvidenceSolver()
    l4 = LeanVerifier()
    l5 = PriorityArbiter()
    return NISPPipeline(l0=object(), l1=l1, l2=l2, l3=l3, l4=l4, l5=l5)


def test_l2_nengo_pipeline_nominal_proceeds():
    """Full L0→L5 cycle with Nengo decoder on a nominal telemetry must:
       - return a SafetyVerdict (action == PROCEED for nominal)
       - rho == 1.0 (Lean cache hit on the all-False state)
       - Z is a length-8 distribution summing to 1
       - certificate has merkle_root and primary_trigger='nominal'
    """
    pipeline = _build_pipeline_with_nengo(cycle_id=0)
    verdict, latent, pred_dist, evidence, proof, D = pipeline.run_cycle(_nominal_telemetry())

    # Verdict structure
    assert verdict.action in {"PROCEED", "REPLAN", "ABORT"}, f"unexpected action {verdict.action}"
    assert verdict.action == "PROCEED", \
        f"nominal telemetry should PROCEED, got {verdict.action} (D={D:.3f}, rho={proof.rho})"
    assert proof.rho == 1.0
    assert verdict.certificate["primary_trigger"] == "nominal"
    assert "merkle_root" in verdict.certificate
    assert len(verdict.certificate["merkle_root"]) == 64  # SHA-256 hex

    # Predicate distribution shape and validity
    Z = np.array(pred_dist.Z)
    assert Z.shape == (8,)
    assert np.isclose(Z.sum(), 1.0, atol=1e-5), f"Z must sum to 1, got {Z.sum()}"
    assert np.all(Z >= 0)

    # Latent passed through unchanged dimensionality
    E = np.array(latent.E)
    assert E.shape == (1024,), f"DiFuMo latent should be 1024-dim, got {E.shape}"


def test_l2_nengo_pipeline_proximity_violation_aborts():
    """Real Tier 1 violation routed through Nengo decoder still aborts,
    primary_trigger pulled from evidence not from Z."""
    pipeline = _build_pipeline_with_nengo(cycle_id=0)
    tel = TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6),
        ee_pos=np.array([0.3, 0.0, 0.9]),
        min_dist=0.30, zone_status=0, sim_time=0.0, tcp_velocity=0.05,
    )
    verdict, *_ = pipeline.run_cycle(tel)
    assert verdict.action == "ABORT"
    assert verdict.certificate["primary_trigger"] == "proximity_violation"
    assert verdict.certificate["trigger_reason"] == "tier1_violation"


def test_l2_nengo_grounding_maps_shape():
    """grounding_maps must have entries for all 8 predicates, each (1024,).

    PredicateDistribution.__post_init__ converts the dict to a sorted tuple of
    (k, tuple) pairs for hashability, so we test that shape here."""
    pipeline = _build_pipeline_with_nengo(cycle_id=0)
    _, _, pred_dist, *_ = pipeline.run_cycle(_nominal_telemetry())
    gm_dict = dict(pred_dist.grounding_maps)
    assert set(gm_dict.keys()) == set(range(8))
    for k, gm in gm_dict.items():
        gm = np.asarray(gm)
        assert gm.shape == (1024,), f"grounding[{k}] shape {gm.shape} != (1024,)"

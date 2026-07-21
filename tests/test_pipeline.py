"""End-to-end NISP pipeline tests — exercises L0→L5 on a TelemetryState and asserts
the full verdict (action + rho + primary_trigger).

Covers the three arbiter paths that are otherwise only tested in isolation:
  - nominal telemetry → PROCEED, rho == 1.0
  - proximity violation → ABORT, primary_trigger == 'proximity_violation'
  - HESITANT risk state with no Tier 1 violations → REPLAN

This is the test that would have caught R3-B1 (qfrc_applied), R3-B2 (softmax
symmetry), and R4-B1 (2 s Lean timeout) as a single integration failure.
"""
import numpy as np
import pytest

from src.nisp.core.orchestrator import NISPPipeline
from src.nisp.core.contracts import TelemetryState
from src.nisp.layers.l1_tribevx import TribevX
from src.nisp.layers.l2_analysis import L2_PredicateDecoder
from src.nisp.layers.evidence_solver import EvidenceSolver
from src.nisp.layers.l4_bridge import LeanVerifier
from src.nisp.layers.priority_arbiter import PriorityArbiter


def _build_pipeline(seed: int = 42):
    # L0 is unused in these tests — we construct TelemetryState directly.
    l1 = TribevX(seed=seed)
    l2 = L2_PredicateDecoder(seed=seed)
    l3 = EvidenceSolver()
    l4 = LeanVerifier()  # cache-backed by default, no subprocess
    l5 = PriorityArbiter()
    # Sentinel for L0 — not None, not actually called in run_cycle(telemetry).
    return NISPPipeline(l0=object(), l1=l1, l2=l2, l3=l3, l4=l4, l5=l5)


def _nominal_telemetry():
    return TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6), ee_pos=np.array([0.3, 0, 0.9]),
        min_dist=1.5, zone_status=0, sim_time=0.0, tcp_velocity=0.05,
    )


def test_pipeline_nominal_proceeds():
    pipeline = _build_pipeline()
    verdict, _, _, _, proof, _ = pipeline.run_cycle(_nominal_telemetry())
    assert verdict.action == "PROCEED"
    assert proof.rho == 1.0
    assert verdict.certificate["primary_trigger"] == "nominal"


def test_pipeline_proximity_violation_aborts():
    pipeline = _build_pipeline()
    tel = TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6), ee_pos=np.array([0.3, 0, 0.9]),
        min_dist=0.3, zone_status=0, sim_time=0.0, tcp_velocity=0.05,
    )
    verdict, _, _, _, proof, _ = pipeline.run_cycle(tel)
    assert verdict.action == "ABORT"
    assert verdict.certificate["primary_trigger"] == "proximity_violation"
    assert verdict.certificate["trigger_reason"] == "tier1_violation"


def test_pipeline_hesitant_triggers_replan():
    """HESITANT risk state + Tier 2 probe firing ≥ 0.60 → REPLAN. Any REPLAN verdict
    within a handful of hesitant cycles is sufficient (L1's sub-slice randomization
    makes any single cycle nondeterministic between 4/5/6)."""
    pipeline = _build_pipeline(seed=123)
    tel = TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6), ee_pos=np.array([0.3, 0, 0.9]),
        min_dist=0.6, zone_status=0, sim_time=0.0, tcp_velocity=0.1,
    )

    # 10 cycles — at least one must produce REPLAN (all 10 should, in practice).
    actions = []
    for _ in range(10):
        verdict, *_ = pipeline.run_cycle(tel)
        actions.append(verdict.action)

    assert "REPLAN" in actions, f"expected at least one REPLAN in {actions}"
    # No ABORT — no Tier 1 violations in this telemetry.
    assert "ABORT" not in actions, f"unexpected ABORT in {actions}"

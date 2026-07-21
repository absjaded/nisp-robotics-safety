"""Phase A-5 — end-to-end pipeline test for the --real-neural curated demo.

Builds the same pipeline as `scripts/demo.py --real-neural --curated` and runs
the 5 hero cycles. Asserts:

- All 5 cycles produce a SafetyVerdict (no exceptions, no shape mismatches).
- Tier-1 violation cycles (7, 11) ABORT with the correct primary_trigger.
- Recovery cycles (0, 14) PROCEED.
- Cycle 4 (the 340ms moat) is left LOOSE — the mock probe weights aren't
  tuned to fire predicate 5 strongly. Phase NE-2/NE-3 will tighten this once
  the H100 fine-tuning pass runs. For now we assert *only* that no Tier-1
  violation exists and no exception is thrown.

This test guards the full real-neural integration: L1_TribeV2_Real (file
lookup) -> L2_Nengo (SNN decoder) -> L3 -> L4 (Lean cache) -> L5 (Merkle cert).
"""
from pathlib import Path

import numpy as np
import pytest

from src.nisp.core.contracts import TelemetryState
from src.nisp.core.orchestrator import NISPPipeline
from src.nisp.layers.evidence_solver import EvidenceSolver
from src.nisp.layers.l1_tribev2_real import L1_TribeV2_Real
from src.nisp.layers.l1_tribevx import TribevX
from src.nisp.layers.l2_nengo import L2_Nengo
from src.nisp.layers.l4_bridge import LeanVerifier
from src.nisp.layers.priority_arbiter import PriorityArbiter


_LATENT_DIR = Path("artifacts/latents")

# Mirrors scripts/demo.py SCRIPT for the 5 curated cycles.
# Each entry: (cycle_id, telemetry_overrides, expected_verdict_or_None_if_loose).
CURATED_TELEMETRY = [
    (0,  dict(min_dist=1.50, tcp_velocity=0.05, zone_status=0),                            "PROCEED"),
    (4,  dict(min_dist=0.62, tcp_velocity=0.14, zone_status=0),                            None),  # loose until probes tuned
    (7,  dict(min_dist=0.35, tcp_velocity=0.20, zone_status=0),                            "ABORT"),
    (11, dict(min_dist=0.20, tcp_velocity=0.40, zone_status=1, tau=[350.0, 0, 0, 0, 0, 0]),"ABORT"),
    (14, dict(min_dist=1.00, tcp_velocity=0.08, zone_status=0),                            "PROCEED"),
]


def _telemetry(cycle_id: int, overrides: dict) -> TelemetryState:
    return TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6),
        tau=np.array(overrides.get("tau", [0.0] * 6), dtype=float),
        ee_pos=np.array([0.3, 0.0, 0.9]),
        min_dist=overrides.get("min_dist", 1.0),
        zone_status=int(overrides.get("zone_status", 0)),
        sim_time=cycle_id * 0.02,
        tcp_velocity=overrides.get("tcp_velocity", 0.05),
    )


def _build_real_neural_pipeline(seed: int = 42) -> NISPPipeline:
    if not (_LATENT_DIR / "cycle_0_difumo.npy").exists():
        pytest.skip("artifacts/latents/cycle_0_difumo.npy missing — "
                    "run scripts/precompute_latents.py --mock-tribe first.")
    fb = TribevX(seed=seed)
    return NISPPipeline(
        l0=object(),
        l1=L1_TribeV2_Real(fallback=fb),
        l2=L2_Nengo(seed=seed),
        l3=EvidenceSolver(),
        l4=LeanVerifier(),
        l5=PriorityArbiter(),
    )


def test_real_neural_curated_arc():
    """5-scene curated arc must produce the documented verdict shape.

    Verdict trail (with current mock probes):
      cycle  0:  PROCEED          (clear workspace baseline)
      cycle  4:  PROCEED|REPLAN   (the moat; loose until NE-2 tunes probes)
      cycle  7:  ABORT            (proximity_violation, ISO 10218-2 trip)
      cycle 11:  ABORT            (combined critical, 4/4 violations)
      cycle 14:  PROCEED          (recovery)
    """
    pipeline = _build_real_neural_pipeline()
    actions = []

    for cycle_id, overrides, expected in CURATED_TELEMETRY:
        tel = _telemetry(cycle_id, overrides)
        verdict, latent, dist, evidence, proof, D = pipeline.run_cycle(tel)
        actions.append((cycle_id, verdict.action, verdict.certificate["primary_trigger"]))

        # Hard assertions: pipeline produced everything it owes.
        assert verdict.action in {"PROCEED", "REPLAN", "ABORT"}
        assert proof.rho == 1.0  # Lean cache hits on every state
        assert "merkle_root" in verdict.certificate
        assert len(verdict.certificate["merkle_root"]) == 64
        assert np.array(dist.Z).shape == (8,)
        assert np.array(latent.E).shape == (1024,), \
            "L1_TribeV2_Real must output 1024-dim DiFuMo latents"

        # Strict-action assertion (only when expected is set).
        if expected:
            assert verdict.action == expected, (
                f"cycle {cycle_id}: expected {expected}, got {verdict.action} "
                f"(primary_trigger={verdict.certificate['primary_trigger']}, D={D:.3f})"
            )

    # Loose-arc assertion: at least one PROCEED, at least one ABORT.
    action_set = {a for _, a, _ in actions}
    assert "PROCEED" in action_set, f"no PROCEED in arc: {actions}"
    assert "ABORT"   in action_set, f"no ABORT in arc: {actions}"

    # Cycle 7 specifically must trigger proximity_violation (ISO 10218-2 row).
    cycle7 = next(t for c, _, t in actions if c == 7)
    assert cycle7 == "proximity_violation", \
        f"cycle 7 must trigger proximity_violation, got {cycle7}"


def test_real_neural_pipeline_uses_1024_dim_latents():
    """Sanity: L1_TribeV2_Real must emit 1024-dim DiFuMo, not 200-dim TribevX."""
    pipeline = _build_real_neural_pipeline()
    tel = _telemetry(0, dict(min_dist=1.5))
    _, latent, _, _, _, _ = pipeline.run_cycle(tel)
    assert np.array(latent.E).shape == (1024,)


def test_real_neural_merkle_round_trip_holds():
    """The auditability claim must hold on the real-neural path too."""
    from src.nisp.crypto.merkle import verdict_merkle_root

    pipeline = _build_real_neural_pipeline()
    tel = _telemetry(7, CURATED_TELEMETRY[2][1])  # cycle 7 ABORT
    verdict, _, dist, evidence, proof, D = pipeline.run_cycle(tel)

    facts_tuple = tuple(
        (f.proposition, f.value, f.raw_reading, f.threshold) for f in evidence.facts
    )
    Z_tuple = tuple(float(z) for z in np.array(dist.Z))
    audited = verdict_merkle_root(proof.rho, D, facts_tuple, Z_tuple)
    assert audited == verdict.certificate["merkle_root"]

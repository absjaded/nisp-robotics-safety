"""Phase A-4 — Merkle round-trip: an external auditor can independently
verify any verdict's certificate from the four logged leaves.

The IEEE-paper claim "every verdict carries an auditable certificate" is only
true if a third party can reproduce the merkle_root without re-running the
pipeline. This test simulates that auditor: take only (rho, D, facts_tuple,
Z_tuple) — the fields written to the JSONL log — re-hash them, and confirm
the result matches what PriorityArbiter stored in cert.
"""
import numpy as np
import pytest

from src.nisp.core.contracts import TelemetryState
from src.nisp.core.orchestrator import NISPPipeline
from src.nisp.crypto.merkle import merkle_root, verdict_merkle_root
from src.nisp.layers.evidence_solver import EvidenceSolver
from src.nisp.layers.l1_tribevx import TribevX
from src.nisp.layers.l2_analysis import L2_PredicateDecoder
from src.nisp.layers.l4_bridge import LeanVerifier
from src.nisp.layers.priority_arbiter import PriorityArbiter


def _build_pipeline(seed: int = 42) -> NISPPipeline:
    l1 = TribevX(seed=seed)
    l2 = L2_PredicateDecoder(p_dim=l1.P, seed=seed)
    return NISPPipeline(
        l0=object(), l1=l1, l2=l2,
        l3=EvidenceSolver(), l4=LeanVerifier(), l5=PriorityArbiter(),
    )


def _audit_root(rho: float, D: float, evidence, Z) -> str:
    """Recompute the cert's merkle_root from logged primitives only.

    This mirrors what an auditor reads from the JSONL log — no objects from the
    pipeline are reused, only the persistable scalars and tuples.
    """
    facts_tuple = tuple(
        (f.proposition, f.value, f.raw_reading, f.threshold) for f in evidence.facts
    )
    Z_tuple = tuple(float(z) for z in Z)
    return verdict_merkle_root(rho, D, facts_tuple, Z_tuple)


def test_merkle_root_matches_auditor_recompute_nominal():
    """Nominal cycle: cert.merkle_root == auditor's recomputed root."""
    pipeline = _build_pipeline()
    tel = TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6),
        ee_pos=np.array([0.3, 0.0, 0.9]),
        min_dist=1.5, zone_status=0, sim_time=0.0, tcp_velocity=0.05,
    )
    verdict, _, dist, evidence, proof, D = pipeline.run_cycle(tel)
    Z = np.array(dist.Z)

    audited = _audit_root(proof.rho, D, evidence, Z)
    assert audited == verdict.certificate["merkle_root"], (
        f"merkle mismatch:\n"
        f"  cert.merkle_root = {verdict.certificate['merkle_root']}\n"
        f"  auditor recomputed = {audited}"
    )
    assert len(audited) == 64  # SHA-256 hex


def test_merkle_root_matches_auditor_recompute_proximity_abort():
    """ABORT cycle with active facts also round-trips correctly."""
    pipeline = _build_pipeline()
    tel = TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6),
        ee_pos=np.array([0.3, 0.0, 0.9]),
        min_dist=0.30, zone_status=0, sim_time=0.0, tcp_velocity=0.05,
    )
    verdict, _, dist, evidence, proof, D = pipeline.run_cycle(tel)
    Z = np.array(dist.Z)

    audited = _audit_root(proof.rho, D, evidence, Z)
    assert audited == verdict.certificate["merkle_root"]
    assert verdict.action == "ABORT"


def test_merkle_root_changes_when_any_leaf_changes():
    """Tampering with any single leaf must change the root.

    This is the property a fraudulent log would have to break. We perturb each
    of the four leaves in turn and confirm a different root.
    """
    pipeline = _build_pipeline()
    tel = TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6),
        ee_pos=np.array([0.3, 0.0, 0.9]),
        min_dist=1.5, zone_status=0, sim_time=0.0, tcp_velocity=0.05,
    )
    verdict, _, dist, evidence, proof, D = pipeline.run_cycle(tel)
    Z = np.array(dist.Z)
    original = verdict.certificate["merkle_root"]

    facts_tuple = tuple(
        (f.proposition, f.value, f.raw_reading, f.threshold) for f in evidence.facts
    )
    Z_tuple = tuple(float(z) for z in Z)

    # Perturb each leaf; root must change.
    perturbed_rho     = verdict_merkle_root(proof.rho + 1e-6, D, facts_tuple, Z_tuple)
    perturbed_D       = verdict_merkle_root(proof.rho, D + 1e-6, facts_tuple, Z_tuple)
    perturbed_facts   = verdict_merkle_root(
        proof.rho, D,
        ((facts_tuple[0][0], not facts_tuple[0][1]) + facts_tuple[0][2:],) + facts_tuple[1:],
        Z_tuple,
    )
    Z_tuple_pert = (Z_tuple[0] + 1e-6,) + Z_tuple[1:]
    perturbed_Z       = verdict_merkle_root(proof.rho, D, facts_tuple, Z_tuple_pert)

    assert perturbed_rho   != original, "perturbed rho leaf must change root"
    assert perturbed_D     != original, "perturbed D leaf must change root"
    assert perturbed_facts != original, "perturbed facts leaf must change root"
    assert perturbed_Z     != original, "perturbed Z leaf must change root"


def test_merkle_root_function_pure():
    """merkle_root is a pure function: same leaves -> same hash."""
    leaves = [b"alpha", b"beta", b"gamma", b"delta"]
    root_a = merkle_root(leaves)
    root_b = merkle_root(leaves)
    assert root_a == root_b
    assert len(root_a) == 64

    # Order matters
    root_swapped = merkle_root([leaves[1], leaves[0], leaves[2], leaves[3]])
    assert root_swapped != root_a

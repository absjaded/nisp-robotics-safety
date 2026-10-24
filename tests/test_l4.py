import pytest
from nisp.verification import LeanVerifier
from nisp.contracts import EvidenceState, SymbolicFact

def test_lean_verifier():
    verifier = LeanVerifier()
    
    facts = (
        SymbolicFact("proximity_violation", True, 0.3, 0.5, "min_dist"),
        SymbolicFact("geofence_breach", False, 0, 1, "zone_status")
    )
    
    evidence = EvidenceState(facts=facts, queried_channels=("min_dist", "zone_status"))
    
    proof = verifier.verify(evidence)
    
    # If lean is installed, rho should be 1.0.
    assert proof.rho == 1.0
    assert proof.n_goals == 1  # Only one active fact: proximity_violation
    assert proof.n_sorry == 0
    assert proof.latency_ms > 0
    assert proof.proof_term_hash is not None

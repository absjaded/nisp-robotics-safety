import pytest
import numpy as np
from nisp.policy import PriorityArbiter
from nisp.contracts import EvidenceState, SymbolicFact, ProofResult

def test_arbiter_abort():
    arbiter = PriorityArbiter()
    
    facts = (
        SymbolicFact("proximity_violation", True, 0.3, 0.5, "min_dist"),
        SymbolicFact("torque_exceedance", True, 350.0, 330.0, "tau")
    )
    evidence = EvidenceState(facts=facts, queried_channels=("min_dist", "tau"))
    
    Z = np.full(8, 0.1)
    proof = ProofResult(rho=1.0, n_sorry=0, n_goals=2, proof_term_hash="abc", latency_ms=10.0)
    
    verdict = arbiter.arbitrate(evidence, Z, proof, D=0.1, proof_latency_ms=10.0)
    
    assert verdict.action == "ABORT"
    assert verdict.certificate["primary_trigger"] == "proximity_violation"

def test_arbiter_replan():
    arbiter = PriorityArbiter()
    
    facts = (
        SymbolicFact("proximity_violation", False, 1.0, 0.5, "min_dist"),
    )
    evidence = EvidenceState(facts=facts, queried_channels=("min_dist",))
    
    Z = np.full(8, 0.1)
    Z[4] = 0.8  # hesitate_spatial >= 0.60
    
    proof = ProofResult(rho=1.0, n_sorry=0, n_goals=1, proof_term_hash="abc", latency_ms=10.0)
    
    # D >= 0.30 triggers REPLAN
    verdict = arbiter.arbitrate(evidence, Z, proof, D=0.35, proof_latency_ms=10.0)
    
    assert verdict.action == "REPLAN"
    assert verdict.certificate["trigger_reason"] == "high_dissonance"

def test_arbiter_proceed():
    arbiter = PriorityArbiter()
    
    facts = (
        SymbolicFact("proximity_violation", False, 1.0, 0.5, "min_dist"),
    )
    evidence = EvidenceState(facts=facts, queried_channels=("min_dist",))
    
    Z = np.full(8, 0.1)
    proof = ProofResult(rho=1.0, n_sorry=0, n_goals=1, proof_term_hash="abc", latency_ms=10.0)
    
    verdict = arbiter.arbitrate(evidence, Z, proof, D=0.1, proof_latency_ms=10.0)
    
    assert verdict.action == "PROCEED"
    assert verdict.certificate["trigger_reason"] == "nominal"

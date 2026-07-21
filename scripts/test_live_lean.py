
import numpy as np
from src.nisp.layers.l4_bridge import LeanVerifier
from src.nisp.core.contracts import EvidenceState, SymbolicFact

def test_live_verify_nominal():
    # Use live mode (no cache)
    verifier = LeanVerifier(use_cache=False)
    
    # Nominal evidence: no facts
    evidence = EvidenceState(facts=(), queried_channels=())
    
    print("Testing live Lean verification for nominal state...")
    result = verifier.verify(evidence)
    print(f"Result: rho={result.rho}, goals={result.n_goals}, sorry={result.n_sorry}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    
    assert result.rho == 1.0
    assert result.n_goals == 1
    assert result.n_sorry == 0

def test_live_verify_violation():
    verifier = LeanVerifier(use_cache=False)
    
    # Proximity violation
    fact = SymbolicFact(
        proposition="proximity_violation",
        value=True,
        raw_reading=0.3,
        threshold=0.5,
        source_channel="proximity"
    )
    evidence = EvidenceState(facts=(fact,), queried_channels=("proximity",))
    
    print("\nTesting live Lean verification for proximity violation...")
    result = verifier.verify(evidence)
    print(f"Result: rho={result.rho}, goals={result.n_goals}, sorry={result.n_sorry}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    
    assert result.rho == 1.0
    assert result.n_goals == 1
    assert result.n_sorry == 0

if __name__ == "__main__":
    try:
        test_live_verify_nominal()
        test_live_verify_violation()
        print("\nAll live verification tests PASSED.")
    except Exception as e:
        print(f"\nTest FAILED: {e}")

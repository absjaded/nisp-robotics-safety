import numpy as np
import pytest
from src.nisp.layers.l2_analysis import L2_PredicateDecoder
from src.nisp.core.contracts import LatentVector, RiskState, PredicateDistribution

def test_l2_predicate_decoder_probes():
    # Setup
    l2 = L2_PredicateDecoder(p_dim=200, k_probes=8)
    # Assemble the nominal-block excitation before constructing the (frozen) latent.
    E_nominal = np.zeros(200)
    E_nominal[150:200] = 1.0
    latent = LatentVector(
        E=E_nominal,
        risk_state=RiskState.NOMINAL,
        timestamp=0.0
    )
    
    # Execute
    dist = l2.process(latent)
    
    # Verify - index 7 should be dominant for nominal latent
    assert isinstance(dist, PredicateDistribution)
    assert len(dist.Z) == 8
    assert dist.Z[7] > 0.5 
    # Nominal should have highest prob at index 7
    assert np.argmax(dist.Z) == 7
    
    # 2. Test Hesitant latent (high on 50:150)
    E_hesitant = np.concatenate([np.full(50, 0.1), np.full(100, 0.8), np.full(50, 0.1)])
    latent_hesitant = LatentVector(E=E_hesitant, risk_state=RiskState.HESITANT, timestamp=1.0)
    
    dist_hesitant = l2.process(latent_hesitant)
    # Hesitant should have highest prob in indices 4-6
    assert 4 <= np.argmax(dist_hesitant.Z) <= 6
    
    # 3. Test Critical latent (high on 0:50)
    E_critical = np.concatenate([np.full(50, 0.8), np.full(150, 0.1)])
    latent_critical = LatentVector(E=E_critical, risk_state=RiskState.CRITICAL, timestamp=2.0)
    
    dist_critical = l2.process(latent_critical)
    # Critical should have highest prob in indices 0-3
    assert 0 <= np.argmax(dist_critical.Z) <= 3

if __name__ == "__main__":
    pytest.main([__file__])

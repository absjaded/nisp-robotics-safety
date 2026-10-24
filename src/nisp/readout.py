import numpy as np
from nisp.contracts import LatentVector, PredicateDistribution

class PredicateReadout:
    """
    Layer 2: linear predicate decoder.
    Maps an L1 LatentVector onto a K=8 PredicateDistribution via K linear
    probes over a P-dimensional latent, with a softmax head. Per-cycle
    grounding maps |W[k] * E| are exposed for narrator-side interpretability.
    """
    def __init__(self, p_dim: int = 200, k_probes: int = 8, seed: int = 42):
        self.p_dim = p_dim
        self.k_probes = k_probes
        self.rng = np.random.default_rng(seed)

        # K=8 linear probes on a P=200 latent. Weight regions are non-overlapping
        # across Tier 2 probes so any single Tier 2 probe can dominate softmax
        # (required for the arbiter's Z[i] >= 0.60 REPLAN gate).
        self.W = self.rng.standard_normal((self.k_probes, self.p_dim)) * 0.01

        # Tier 1 (indices 0-3): CRITICAL latent slice 0:50, shared boost.
        self.W[0:4, 0:50] += 0.5

        # Tier 2 (indices 4-6): HESITANT latent slice 100:150, each probe owns
        # its own 17-wide sub-slice so the softmax can concentrate on one probe.
        self.W[4, 100:117] += 1.2  # hesitate_spatial
        self.W[5, 117:134] += 1.2  # hesitate_kinematic
        self.W[6, 134:150] += 1.2  # dissonance_relational

        # Tier 3 (index 7): NOMINAL latent slice 150:200.
        self.W[7, 150:200] += 0.5

    def process(self, latent: LatentVector) -> PredicateDistribution:
        """Processes LatentVector to produce a probability distribution over predicates."""
        E = np.array(latent.E)
        logits = self.W @ E

        exp_logits = np.exp(logits - np.max(logits))
        Z = exp_logits / np.sum(exp_logits)

        # Input-dependent grounding maps: |W[k] * E| / max — per-cycle saliency,
        # not static weight magnitudes. Lets the Narrator explain what drove the prediction.
        grounding_maps = {}
        for k in range(self.k_probes):
            contribution = np.abs(self.W[k] * E)
            cmax = np.max(contribution)
            grounding_maps[k] = contribution / cmax if cmax > 0 else np.zeros_like(contribution)

        return PredicateDistribution(Z=Z, grounding_maps=grounding_maps)

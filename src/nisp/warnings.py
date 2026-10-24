import numpy as np
from nisp.contracts import TelemetryState, LatentVector, RiskState

class WarningFixture:
    """
    Seeded synthetic warning features for exercising the decision pipeline.

    Gaussian samples around hand-defined centroids depend on the supplied
    telemetry or an explicit scenario override. This generator uses no neural
    recordings or pretrained encoder.
    """
    def __init__(self, P: int = 200, seed: int = 42):
        self.P = P
        self.rng = np.random.default_rng(seed)

        # Hand-tuned centroids per risk state over P=200 latent parcels.
        # NOMINAL:  high on 150:200 (→ L2 probe 7)
        # CRITICAL: high on 0:50   (→ L2 probes 0-3)
        # HESITANT: three sub-slices of 100:150, one randomly chosen per sample
        #   — sub-slices map to L2 probes 4, 5, 6 (hesitate_spatial / kinematic /
        #   dissonance_relational). Sub-type selection happens in `process()`.
        self.mu_nominal  = np.concatenate([np.full(150, 0.1), np.full(50,  0.8)])
        self.mu_critical = np.concatenate([np.full(50,  0.8), np.full(150, 0.1)])
        self.hesitant_slices = [(100, 117), (117, 134), (134, 150)]

        # Shared diagonal covariance
        self.sigma = 0.05

    # Bands derived from thresholds.yaml:
    #   proximity hard threshold = 0.5 m  → CRITICAL at min_dist ≤ 0.5, HESITANT up to 0.7
    #   velocity  hard threshold = 0.25 m/s → HESITANT warning at 0.6 × threshold = 0.15 m/s
    PROX_CRITICAL = 0.5
    PROX_HESITANT_MAX = 0.7
    VEL_HESITANT = 0.15

    def _infer_risk_state(self, telemetry: TelemetryState) -> RiskState:
        """Early-warning bands anchored to thresholds.yaml hard thresholds."""
        if telemetry.min_dist <= self.PROX_CRITICAL:
            return RiskState.CRITICAL
        if telemetry.min_dist < self.PROX_HESITANT_MAX or telemetry.tcp_velocity > self.VEL_HESITANT:
            return RiskState.HESITANT
        return RiskState.NOMINAL

    def _mean_for(self, state: RiskState) -> np.ndarray:
        """Return the centroid for a given risk state. HESITANT picks one
        sub-slice per call so individual Tier 2 probes dominate L2's softmax."""
        if state == RiskState.NOMINAL:
            return self.mu_nominal
        if state == RiskState.CRITICAL:
            return self.mu_critical
        # HESITANT: pick one of 3 sub-slices this cycle
        mu = np.full(self.P, 0.1)
        lo, hi = self.hesitant_slices[self.rng.integers(0, 3)]
        mu[lo:hi] = 0.8
        return mu

    def process(self, telemetry: TelemetryState) -> LatentVector:
        """Generates a latent vector E based on current telemetry."""
        risk_state = self._infer_risk_state(telemetry)
        mu = self._mean_for(risk_state)
        E = self.rng.normal(mu, self.sigma)
        E = np.clip(E, 0, 1)
        return LatentVector(E=E, risk_state=risk_state, timestamp=telemetry.sim_time)

    def generate(self, risk_state_name: str) -> np.ndarray:
        """Utility for manual generation / scenario overrides."""
        try:
            state = RiskState(risk_state_name)
        except ValueError as exc:
            raise ValueError(f"Unknown warning fixture: {risk_state_name}") from exc
        E = self.rng.normal(self._mean_for(state), self.sigma)
        return np.clip(E, 0, 1)

import numpy as np
from nisp.contracts import EvidenceState
from nisp.validation import (
    validate_telemetry, validate_distribution, validate_evidence, validate_proof,
)

PROPOSITION_TO_PREDICATE_INDEX = {
    "proximity_violation": 0,
    "torque_exceedance":   1,
    "geofence_breach":     2,
    "velocity_limit":      3,
    # Tier 2 predicates are not driven by hard facts — left at baseline
}
EPSILON = 1e-8
BASELINE = 0.01  # small non-zero mass for all predicates

def kl_divergence(P_intent: np.ndarray, P_action: np.ndarray, epsilon: float = EPSILON) -> float:
    p = np.clip(P_intent, epsilon, 1.0)
    q = np.clip(P_action, epsilon, 1.0)
    return float(np.sum(p * np.log(p / q)))

class NISPPipeline:
    """
    Core orchestrator that sequences the NISP Engine layers.

    L0/L3/L4/L5 are required. L1/L2 may be None to run the explicit ablation
    path — in that case Z falls back to uniform and the verdict certificate is
    tagged with `tribevx_active` / `neuronax_active` flags.
    """
    def __init__(self, l0, l1, l2, l3, l4, l5):
        if l0 is None or l3 is None or l4 is None or l5 is None:
            raise ValueError(
                "NISPPipeline requires l0, l3, l4, l5. "
                "l1/l2 may be None only for explicit ablation."
            )
        self.l0_ingestion = l0
        self.l1_tribevx = l1
        self.l2_neuronax = l2
        self.l3_evidence_solver = l3
        self.l4_formal_verifier = l4
        self.l5_priority_arbiter = l5

    def _construct_p_action(self, evidence: EvidenceState) -> np.ndarray:
        P = np.full(8, BASELINE)
        any_violation = False
        for fact in evidence.facts:
            idx = PROPOSITION_TO_PREDICATE_INDEX.get(fact.proposition)
            if idx is not None:
                if fact.value:
                    P[idx] = 0.99
                    any_violation = True
                else:
                    P[idx] = BASELINE
        
        # If no hard violations, the "action" is semantically nominal
        if not any_violation:
            P[7] = 0.99
            
        P = np.clip(P, EPSILON, 1.0)
        return P / P.sum()

    def run_cycle(self, telemetry):
        """
        Executes a single pass of the NISP architecture against a pre-ingested
        TelemetryState. The caller is responsible for L0 ingestion — that
        decouples the pipeline from MuJoCo so scenario YAMLs with
        `telemetry_overrides:` can bypass physics entirely.
        """
        validate_telemetry(telemetry)
        # L1 -> WarningFixture (synthetic warning fixture)
        latent_vector = self.l1_tribevx.process(telemetry) if self.l1_tribevx else None

        # L2 -> linear predicate decoder
        predicate_dist = self.l2_neuronax.process(latent_vector) if (self.l2_neuronax and latent_vector) else None

        # Uniform fallback when L1/L2 are ablated
        Z = np.array(predicate_dist.Z) if predicate_dist else np.full(8, 0.125)

        verdict, evidence_state, proof_result, D = self.evaluate_distribution(telemetry, Z)
        return verdict, latent_vector, predicate_dist, evidence_state, proof_result, D

    def evaluate_distribution(self, telemetry, Z):
        """Common validated decision path for generated or supplied warnings.

        Invalid input raises ValueError before a normal verdict is emitted.
        The demo's boundary converts that rejection into an explicit ABORT.
        """
        validate_telemetry(telemetry)
        Z = validate_distribution(Z)
        # L3 -> Evidence Solver
        evidence_state = self.l3_evidence_solver.process(Z, telemetry)
        validate_evidence(evidence_state)

        # Dissonance
        P_action = self._construct_p_action(evidence_state)
        D = kl_divergence(Z, P_action)

        # L4 -> Formal Verifier
        proof_result = self.l4_formal_verifier.verify(evidence_state)
        validate_proof(proof_result)

        # L5 -> Priority Arbiter
        verdict = self.l5_priority_arbiter.arbitrate(
            evidence_state, Z, proof_result, D, proof_result.latency_ms
        )
        return verdict, evidence_state, proof_result, D

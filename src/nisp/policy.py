import numpy as np
import hashlib
from types import MappingProxyType
from nisp.contracts import ProofResult, SafetyVerdict, EvidenceState
from nisp.certificates import verdict_merkle_root

class PriorityArbiter:
    def __init__(self):
        # Inherited prototype priority ordering (lower number = higher priority).
        self.priority_map = {
            "proximity_violation": 1,
            "geofence_breach": 2,
            "torque_exceedance": 3,
            "velocity_limit": 4
        }

        # Z Indices for Tier 2 predicates
        self.TIER_2_INDICES = [4, 5, 6]  # hesitate_spatial, hesitate_kinematic, dissonance_relational

        # Prototype lookup budget; timing is measured separately from offline compilation.
        self.PROOF_TIMEOUT_MS = 50.0

    def arbitrate(self, evidence: EvidenceState, Z: np.ndarray, proof: ProofResult, D: float, proof_latency_ms: float) -> SafetyVerdict:
        active_violations = []
        for fact in evidence.facts:
            if fact.value:
                active_violations.append(fact.proposition)

        # Sort by priority
        active_violations.sort(key=lambda x: self.priority_map.get(x, 99))

        primary_trigger = active_violations[0] if active_violations else "nominal"

        # 1. Check for ABORT conditions
        if active_violations:
            action = "ABORT"
            trigger_reason = "tier1_violation"
        elif proof_latency_ms > self.PROOF_TIMEOUT_MS:
            action = "ABORT"
            trigger_reason = "proof_timeout"
        else:
            # 2. Check for REPLAN conditions
            # All Tier 1 False AND at least one Tier 2 predicate Z[i] >= 0.60 AND (rho < 1.0 OR D >= 0.30)
            has_tier2_activation = any(Z[i] >= 0.60 for i in self.TIER_2_INDICES)

            rho_incomplete = abs(proof.rho - 1.0) >= 1e-9
            if has_tier2_activation and (rho_incomplete or D >= 0.30):
                action = "REPLAN"
                trigger_reason = "incomplete_proof" if rho_incomplete else "high_dissonance"
            else:
                # 3. PROCEED condition
                if not rho_incomplete:
                    action = "PROCEED"
                    trigger_reason = "nominal"
                else:
                    # Fallback precautionary
                    action = "ABORT"
                    trigger_reason = "incomplete_proof"

        # Generate Certificate — SHA-256 Merkle root over (rho, D, facts, Z)
        facts_tuple = tuple(
            (f.proposition, f.value, f.raw_reading, f.threshold)
            for f in evidence.facts
        )
        Z_tuple = tuple(float(z) for z in np.asarray(Z))
        merkle_root = verdict_merkle_root(proof.rho, D, facts_tuple, Z_tuple)

        cert_dict = {
            "active_violations": tuple(active_violations),
            "primary_trigger": primary_trigger,
            "proof_term_hash": proof.proof_term_hash,
            "latency_ms": proof_latency_ms,
            "trigger_reason": trigger_reason,
            "merkle_root": merkle_root,
        }

        return SafetyVerdict(
            action=action,
            D=D,
            rho=proof.rho,
            certificate=MappingProxyType(cert_dict)
        )


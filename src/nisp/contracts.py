from dataclasses import dataclass, field
import numpy as np
from types import MappingProxyType
from typing import Any, Tuple, Mapping
from enum import Enum

class RiskState(Enum):
    NOMINAL = "nominal"
    HESITANT = "hesitant"
    CRITICAL = "critical"

@dataclass(frozen=True, unsafe_hash=True)
class TelemetryState:
    q: Any  # Converted to tuple[float, ...] in post_init
    q_dot: Any
    tau: Any
    ee_pos: Any
    min_dist: float
    zone_status: int
    sim_time: float
    closing_velocity: float = 0.0
    human_aabb_center: Any = field(default_factory=lambda: tuple([0.0, 0.0, 0.0]))
    tcp_velocity: float = 0.0
    schema_version: int = 1

    def __post_init__(self):
        # Convert any ndarrays to tuples to ensure robust hashing and equality
        for field_name in ["q", "q_dot", "tau", "ee_pos", "human_aabb_center"]:
            val = getattr(self, field_name)
            if isinstance(val, np.ndarray):
                object.__setattr__(self, field_name, tuple(val.tolist()))
            elif isinstance(val, list):
                object.__setattr__(self, field_name, tuple(val))

@dataclass(frozen=True, unsafe_hash=True)
class LatentVector:
    E: Any  # Converted to tuple[float, ...] in post_init
    risk_state: RiskState
    timestamp: float
    schema_version: int = 1

    def __post_init__(self):
        if isinstance(self.E, np.ndarray):
            object.__setattr__(self, 'E', tuple(self.E.tolist()))
        elif isinstance(self.E, list):
            object.__setattr__(self, 'E', tuple(self.E))

@dataclass(frozen=True, unsafe_hash=True)
class PredicateDistribution:
    Z: Any  # Converted to tuple[float, ...] in post_init
    grounding_maps: Any  # Converted to tuple of (int, tuple) pairs
    schema_version: int = 1

    def __post_init__(self):
        if isinstance(self.Z, np.ndarray):
            object.__setattr__(self, 'Z', tuple(self.Z.tolist()))
        
        # MappingProxyType is not hashable, so we convert to a sorted tuple of items
        maps = self.grounding_maps
        if isinstance(maps, (dict, MappingProxyType)):
            new_maps = []
            for k in sorted(maps.keys()):
                v = maps[k]
                if isinstance(v, np.ndarray):
                    new_maps.append((k, tuple(v.tolist())))
                else:
                    new_maps.append((k, v))
            object.__setattr__(self, 'grounding_maps', tuple(new_maps))

@dataclass(frozen=True)
class SymbolicFact:
    proposition: str
    value: bool
    raw_reading: float
    threshold: float
    source_channel: str
    schema_version: int = 1

@dataclass(frozen=True)
class EvidenceState:
    facts: tuple[SymbolicFact, ...]
    queried_channels: tuple[str, ...]
    schema_version: int = 1

@dataclass(frozen=True)
class ProofResult:
    rho: float
    n_sorry: int
    n_goals: int
    proof_term_hash: str
    latency_ms: float
    schema_version: int = 1

@dataclass(frozen=True)
class ProposedAction:
    kinematic_target: Any        # tuple[float, ...] — target joint positions (6-DOF)
    risk_state_hint: RiskState   # WM's own risk classification
    schema_version: int = 1

@dataclass(frozen=True)
class NarrationRecord:
    sim_time: float
    cycle_id: int
    verdict: str
    trigger: str
    top_predicate_name: str
    top_predicate_confidence: float
    top_predicate_tier: int
    primary_evidence_proposition: str
    primary_evidence_value: bool
    primary_evidence_reading: float
    primary_evidence_threshold: float
    primary_evidence_channel: str
    rho: float
    D: float
    certificate_hash: str
    schema_version: int = 1

@dataclass(frozen=True)
class SafetyVerdict:
    action: str  # "PROCEED" | "ABORT" | "REPLAN"
    D: float
    rho: float
    certificate: MappingProxyType[str, Any]
    schema_version: int = 1

@dataclass(frozen=True)
class CyclePayload:
    """Typed payload for per-cycle JSONL logging. Required for IEEE reproducibility."""
    cycle_id: int
    sim_time: float
    wall_timestamp: float
    scenario_name: str
    seed: int
    risk_state: str
    E_norm: float
    Z: tuple
    facts: tuple  # tuple[(str, bool, float, float), ...]
    rho: float
    D: float
    verdict: str
    primary_trigger: str
    trigger_reason: str
    wall_latency_ms: float
    sim_latency_ms: float
    lean_latency_ms: float
    certificate_hash: str
    tribevx_active: bool
    neuronax_active: bool
    narration: str = ""
    schema_version: int = 1
    proof_term_hash: str = ""

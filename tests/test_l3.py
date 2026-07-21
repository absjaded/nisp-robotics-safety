import pytest
import numpy as np
from src.nisp.layers.evidence_solver import EvidenceSolver
from src.nisp.core.contracts import TelemetryState

def test_evidence_solver_nominal():
    solver = EvidenceSolver()
    
    telemetry = TelemetryState(
        q=np.zeros(6),
        q_dot=np.zeros(6),
        tau=np.zeros(6),
        ee_pos=np.zeros(3),
        min_dist=1.0,  # > 0.5 (safe)
        zone_status=0, # == 0 (safe)
        sim_time=0.0,
        tcp_velocity=0.1 # < 0.25 (safe)
    )
    
    Z = [0.1] * 8
    evidence = solver.process(Z, telemetry)
    
    for fact in evidence.facts:
        assert fact.value is False

def test_evidence_solver_violation():
    solver = EvidenceSolver()

    telemetry = TelemetryState(
        q=np.zeros(6),
        q_dot=np.zeros(6),
        tau=np.array([350.0, 0, 0, 0, 0, 0]), # > 330.0 (violation)
        ee_pos=np.zeros(3),
        min_dist=0.3,  # < 0.5 (violation)
        zone_status=0,
        sim_time=0.0,
        tcp_velocity=0.1
    )

    Z = [0.1] * 8
    evidence = solver.process(Z, telemetry)

    facts_dict = {f.proposition: f.value for f in evidence.facts}
    assert facts_dict["proximity_violation"] is True
    assert facts_dict["torque_exceedance"] is True
    assert facts_dict["geofence_breach"] is False
    assert facts_dict["velocity_limit"] is False


def test_evidence_solver_geofence_breach():
    """Post R1-B1 fix: zone_status=1 must trigger geofence_breach."""
    solver = EvidenceSolver()
    telemetry = TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6), tau=np.zeros(6), ee_pos=np.zeros(3),
        min_dist=1.0, zone_status=1, sim_time=0.0, tcp_velocity=0.1,
    )
    evidence = solver.process([0.1] * 8, telemetry)
    facts = {f.proposition: f.value for f in evidence.facts}
    assert facts["geofence_breach"] is True
    assert facts["proximity_violation"] is False

"""Validate the recovered pipeline's input boundary before making a decision."""
import numpy as np

PROPOSITIONS = {"proximity_violation", "torque_exceedance", "geofence_breach", "velocity_limit"}


def _finite(value, name, shape=None):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: numeric values required") from exc
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: non-finite value")
    return array


def validate_telemetry(telemetry):
    for name, shape in (("q", (6,)), ("q_dot", (6,)), ("tau", (6,)),
                        ("ee_pos", (3,)), ("human_aabb_center", (3,))):
        _finite(getattr(telemetry, name), name, shape)
    for name in ("min_dist", "sim_time", "closing_velocity", "tcp_velocity", "zone_status"):
        _finite(getattr(telemetry, name), name, ())
    if telemetry.min_dist < 0 or telemetry.tcp_velocity < 0 or telemetry.sim_time < 0:
        raise ValueError("distance, speed and simulation time must be nonnegative")
    if telemetry.zone_status not in (0, 1):
        raise ValueError("zone_status: expected 0 or 1")


def validate_distribution(values):
    z = _finite(values, "predicate distribution", (8,))
    if np.any(z < 0) or np.any(z > 1) or not np.isclose(z.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("predicate distribution: expected probabilities summing to one")
    return z


def validate_evidence(evidence):
    names = [fact.proposition for fact in evidence.facts]
    if len(names) != 4 or set(names) != PROPOSITIONS:
        raise ValueError("evidence: expected each of the four hard constraints exactly once")
    for fact in evidence.facts:
        if not isinstance(fact.value, (bool, np.bool_)):
            raise ValueError("evidence: violation values must be booleans")
        _finite(fact.raw_reading, "evidence reading", ())
        _finite(fact.threshold, "evidence threshold", ())


def validate_proof(proof):
    _finite(proof.rho, "proof score", ())
    _finite(proof.latency_ms, "proof latency", ())
    if not 0 <= proof.rho <= 1 or proof.latency_ms < 0:
        raise ValueError("proof: invalid score or latency")
    if (not isinstance(proof.n_goals, int) or not isinstance(proof.n_sorry, int)
            or proof.n_goals < 1 or not 0 <= proof.n_sorry <= proof.n_goals):
        raise ValueError("proof: invalid goal counts")
    expected = 1.0 - proof.n_sorry / proof.n_goals
    if not np.isclose(proof.rho, expected, rtol=0, atol=1e-12):
        raise ValueError("proof: score disagrees with goal counts")

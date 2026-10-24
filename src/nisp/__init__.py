"""NISP: validated evidence, an explicit decision, and its inspectable trace."""
from .contracts import TelemetryState

__version__ = "0.1.0"
__all__ = ["TelemetryState", "decide"]


def decide(telemetry, warning=None, *, name="decision"):
    """Return a decision trace; malformed inputs yield an explicit ABORT record.

    warning is an eight-probability vector. None supplies a uniform ablation.
    The caller remains responsible for acting on the returned decision.
    """
    from .demo import decision_trace
    return decision_trace(name, telemetry, [0.125] * 8 if warning is None else warning,
                          warning_source="uniform ablation" if warning is None else "supplied warning")

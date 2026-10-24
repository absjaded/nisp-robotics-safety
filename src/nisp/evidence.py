import yaml
import os
from nisp.contracts import TelemetryState, SymbolicFact, EvidenceState

class EvidenceSolver:
    def __init__(self, thresholds_path: str = None):
        if thresholds_path is None:
            thresholds_path = os.path.join(os.path.dirname(__file__), "assets", "config", "thresholds.yaml")
        
        with open(thresholds_path, 'r') as f:
            self.thresholds = yaml.safe_load(f)["channels"]

    def _factify(self, prop_name: str, yaml_key: str, channel: str, reading: float) -> SymbolicFact:
        config = self.thresholds[yaml_key]
        limit = config["threshold"]
        direction = config["direction"]
        
        # We don't implement full hysteresis logic here yet, just basic threshold
        is_violation = False
        if isinstance(limit, list):
            # For torque which has per-joint limits
            for r, l in zip(reading, limit):
                if direction == "above" and abs(r) > l:
                    is_violation = True
                    break
        else:
            if direction == "below" and reading < limit:
                is_violation = True
            elif direction == "above" and reading > limit:
                is_violation = True
                
        # If it's torque, reading is a tuple, but raw_reading expects float, so we just pass max absolute for scalar representation
        raw_val = float(max([abs(x) for x in reading])) if isinstance(reading, (list, tuple)) else float(reading)
        thresh_val = float(max(limit)) if isinstance(limit, list) else float(limit)

        return SymbolicFact(
            proposition=prop_name,
            value=is_violation,
            raw_reading=raw_val,
            threshold=thresh_val,
            source_channel=channel
        )

    def process(self, Z: list, telemetry: TelemetryState) -> EvidenceState:
        # Z acts as attention prior, but for Tier 1 we must evaluate all hard facts
        facts = []
        
        # 1. Proximity
        facts.append(self._factify("proximity_violation", "proximity", "min_dist", telemetry.min_dist))
        
        # 2. Torque
        facts.append(self._factify("torque_exceedance", "torque", "tau", telemetry.tau))
        
        # 3. Geofence
        facts.append(self._factify("geofence_breach", "geofence", "zone_status", float(telemetry.zone_status)))
        
        # 4. Velocity
        facts.append(self._factify("velocity_limit", "velocity", "tcp_velocity", telemetry.tcp_velocity))
        
        queried = ("min_dist", "tau", "zone_status", "tcp_velocity")
        return EvidenceState(facts=tuple(facts), queried_channels=queried)

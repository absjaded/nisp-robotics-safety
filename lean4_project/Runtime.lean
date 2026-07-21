import NISPAxioms.Basic

def currentState : IntentState := {
  proximity_violation := true,
  torque_violation := false,
  geofence_violation := false,
  velocity_violation := false
}

theorem runtime_proximity_violation : safety_check currentState = SafetyVerdict.abort := by decide
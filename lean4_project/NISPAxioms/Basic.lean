inductive SafetyVerdict where
  | proceed : SafetyVerdict
  | abort   : SafetyVerdict
  | replan  : SafetyVerdict
  deriving Repr, DecidableEq

structure IntentState where
  proximity_violation : Bool
  torque_violation    : Bool
  geofence_violation  : Bool
  velocity_violation  : Bool
  deriving Repr, DecidableEq

def safety_check (s : IntentState) : SafetyVerdict :=
  if s.proximity_violation || s.torque_violation ||
     s.geofence_violation  || s.velocity_violation then
    SafetyVerdict.abort
  else
    SafetyVerdict.proceed

theorem safety_sound (s : IntentState)
    (h : s.proximity_violation = true ∨ s.torque_violation = true ∨
         s.geofence_violation = true ∨ s.velocity_violation = true) :
    safety_check s = SafetyVerdict.abort := by
  unfold safety_check
  simp only [Bool.or_eq_true]
  rcases h with h | h | h | h <;> simp [h]

theorem safety_nominal (s : IntentState)
    (h : s.proximity_violation = false ∧ s.torque_violation = false ∧
         s.geofence_violation = false ∧ s.velocity_violation = false) :
    safety_check s = SafetyVerdict.proceed := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  unfold safety_check
  simp [h1, h2, h3, h4]

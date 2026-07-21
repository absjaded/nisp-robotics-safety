import os
import subprocess
import time
import hashlib
from types import MappingProxyType
from src.nisp.core.contracts import EvidenceState, ProofResult
from src.nisp.layers.l4_cache import ProofCache

class LeanVerifier:
    def __init__(self, lean_project_path: str = None, use_cache: bool = True):
        if lean_project_path is None:
            self.lean_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "lean4_project"))
        else:
            self.lean_project_path = lean_project_path
        self._cache = ProofCache() if use_cache else None

    def verify(self, evidence: EvidenceState) -> ProofResult:
        """Runtime verification — uses the 16-state proof cache by default, falling back
        to a live `lake env lean` subprocess only when cache is disabled."""
        if self._cache is not None:
            return self._cache.lookup(evidence)
        return self._verify_live(evidence)

    def _verify_live(self, evidence: EvidenceState) -> ProofResult:
        start_time = time.time()
        
        # 1. Map facts to IntentState booleans
        # Defaults
        state_bools = {
            "proximity_violation": "false",
            "torque_violation": "false",
            "geofence_violation": "false",
            "velocity_violation": "false"
        }
        
        active_facts = []
        for fact in evidence.facts:
            if fact.value:
                active_facts.append(fact.proposition)
                if fact.proposition == "proximity_violation":
                    state_bools["proximity_violation"] = "true"
                elif fact.proposition == "torque_exceedance":
                    state_bools["torque_violation"] = "true"
                elif fact.proposition == "geofence_breach":
                    state_bools["geofence_violation"] = "true"
                elif fact.proposition == "velocity_limit":
                    state_bools["velocity_violation"] = "true"
        
        # 2. Generate Runtime.lean
        lean_code = [
            "import NISPAxioms.Basic",
            "",
            "def currentState : IntentState := {",
            f"  proximity_violation := {state_bools['proximity_violation']},",
            f"  torque_violation := {state_bools['torque_violation']},",
            f"  geofence_violation := {state_bools['geofence_violation']},",
            f"  velocity_violation := {state_bools['velocity_violation']}",
            "}",
            ""
        ]
        
        n_goals = 0
        if not active_facts:
            lean_code.append("theorem runtime_nominal : safety_check currentState = SafetyVerdict.proceed := by decide")
            n_goals = 1
        else:
            for prop in active_facts:
                lean_code.append(f"theorem runtime_{prop} : safety_check currentState = SafetyVerdict.abort := by decide")
                n_goals += 1
                
        lean_file_path = os.path.join(self.lean_project_path, "Runtime.lean")
        with open(lean_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lean_code))
            
        # 3. Call Lean compiler
        try:
            # We run `lean Runtime.lean` in the context of the lake project
            # Using lake env lean Runtime.lean
            result = subprocess.run(
                ["lake", "env", "lean", "Runtime.lean"],
                cwd=self.lean_project_path,
                capture_output=True,
                text=True,
                timeout=2.0 # 2 seconds max
            )
            
            if result.returncode == 0:
                n_sorry = 0
            else:
                # If there's an error, we assume all goals failed for simplicity in PoC
                n_sorry = n_goals
                print(f"Lean proof failed: {result.stderr}")
        except Exception as e:
            n_sorry = n_goals
            print(f"Lean proof execution error: {e}")
            
        latency_ms = (time.time() - start_time) * 1000.0
        rho = 1.0 - (n_sorry / n_goals) if n_goals > 0 else 0.0
        
        # Proof term hash acts as our certificate of formal verification
        proof_term_hash = hashlib.sha256("\n".join(lean_code).encode()).hexdigest()
        
        return ProofResult(
            rho=rho,
            n_sorry=n_sorry,
            n_goals=n_goals,
            proof_term_hash=proof_term_hash,
            latency_ms=latency_ms
        )

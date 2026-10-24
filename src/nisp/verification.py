import os
import subprocess
import time
import hashlib
import tempfile
from nisp.contracts import EvidenceState, ProofResult
from nisp.proof_cache import ProofCache, canonical_lean_source, evidence_to_key

class LeanVerifier:
    def __init__(self, lean_project_path: str = None, use_cache: bool = True,
                 compilation_timeout_seconds: float = 30.0):
        if lean_project_path is None:
            if not use_cache:
                raise ValueError("Live verification needs a writable qualified Lean project; run nisp verify-policy first.")
            self.lean_project_path = None
        else:
            self.lean_project_path = lean_project_path
        self._cache = ProofCache() if use_cache else None
        self.compilation_timeout_seconds = compilation_timeout_seconds

    def verify(self, evidence: EvidenceState) -> ProofResult:
        """Runtime verification — uses the 16-state proof cache by default, falling back
        to a live `lake env lean` subprocess only when cache is disabled."""
        if self._cache is not None:
            return self._cache.lookup(evidence)
        return self._verify_live(evidence)

    def _verify_live(self, evidence: EvidenceState) -> ProofResult:
        """Offline/debug compilation using exactly the source hashed by the cache.

        Compilation time is reported honestly. The runtime arbiter still applies
        its inherited 50 ms budget; live compilation is not the default path.
        """
        start_time = time.perf_counter()
        flags = evidence_to_key(evidence)
        source = canonical_lean_source(flags, any(flags))
        n_sorry = 1
        with tempfile.TemporaryDirectory(prefix=".runtime-", dir=self.lean_project_path) as tmp:
            lean_file_path = os.path.join(tmp, "Runtime.lean")
            with open(lean_file_path, "w", encoding="utf-8") as handle:
                handle.write(source)
            try:
                result = subprocess.run(
                    ["lake", "env", "lean", lean_file_path], cwd=self.lean_project_path,
                    capture_output=True, text=True, timeout=self.compilation_timeout_seconds,
                )
                n_sorry = int(result.returncode != 0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        return ProofResult(
            rho=float(1 - n_sorry), n_sorry=n_sorry, n_goals=1,
            proof_term_hash=hashlib.sha256(source.encode()).hexdigest(),
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )

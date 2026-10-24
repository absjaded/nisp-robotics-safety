"""Offline Lean qualification in an output directory, including installed wheels."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import time

from .contracts import EvidenceState, SymbolicFact
from .proof_cache import ProofCache, canonical_lean_source
from .verification import LeanVerifier

PROPOSITIONS = ("proximity_violation", "torque_exceedance", "geofence_breach", "velocity_limit")


def qualify(output):
    output = Path(output).resolve()
    source = Path(__file__).resolve().parent / "assets/lean"
    project = output / "lean"
    for name in ("lean-toolchain", "lakefile.lean", "NISPAxioms.lean", "NISPAxioms/Basic.lean"):
        target = project / name
        data = (source / name).read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite a different Lean source: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
    lake, elan = shutil.which("lake"), shutil.which("elan")
    if not lake or not elan:
        raise RuntimeError("Install Lean 4.16.0 with elan before verification; no download was attempted.")
    toolchain = (project / "lean-toolchain").read_text().strip()
    installed = subprocess.run([elan, "toolchain", "list"], capture_output=True, text=True, check=True)
    if toolchain not in installed.stdout.splitlines():
        raise RuntimeError(f"Required toolchain is not installed: {toolchain}")

    def compile_command(arguments):
        started = time.perf_counter()
        proc = subprocess.run([lake] + arguments, cwd=project, capture_output=True, text=True, timeout=120)
        record = {"command": ["lake"] + arguments, "returncode": proc.returncode,
                  "seconds": time.perf_counter() - started, "stdout": proc.stdout, "stderr": proc.stderr}
        if proc.returncode:
            raise RuntimeError(json.dumps(record, indent=2))
        return record

    version = compile_command(["env", "lean", "--version"])
    build = compile_command(["build"])
    cases, sections = [], ["import NISPAxioms.Basic\n"]
    cache = ProofCache()
    for flags in itertools.product((False, True), repeat=4):
        name = "State" + "".join(str(int(flag)) for flag in flags)
        canonical = canonical_lean_source(flags, any(flags))
        sections.append(f"namespace {name}\n" + canonical.split("\n", 1)[1] + f"\nend {name}\n")
        evidence = EvidenceState(tuple(SymbolicFact(p, f, float(f), .5, p)
                                       for p, f in zip(PROPOSITIONS, flags)), PROPOSITIONS)
        cached = cache.lookup(evidence)
        source_hash = hashlib.sha256(canonical.encode()).hexdigest()
        if cached.rho != 1.0 or cached.proof_term_hash != source_hash:
            raise RuntimeError("Cache entry differs from canonical policy source")
        cases.append({"state": list(flags), "policy_verdict": "ABORT" if any(flags) else "PROCEED",
                      "source_sha256": source_hash, "cache_matches_source": True,
                      "cache_lookup_ms": cached.latency_ms})
    generated = project / "AllStates.lean"
    generated.write_text("\n".join(sections), encoding="utf-8")
    compilation = compile_command(["env", "lean", generated.name])
    checks = []
    for flags in ((False,) * 4, (True,) * 4):
        evidence = EvidenceState(tuple(SymbolicFact(p, f, float(f), .5, p)
                                       for p, f in zip(PROPOSITIONS, flags)), PROPOSITIONS)
        live = LeanVerifier(str(project), use_cache=False).verify(evidence)
        if live.rho != 1.0 or live.proof_term_hash != cache.lookup(evidence).proof_term_hash:
            raise RuntimeError("Live compiler bridge differs from the qualified cache")
        checks.append({"state": list(flags), "rho": live.rho, "source_matches_cache": True,
                       "compilation_ms": live.latency_ms})
    report = {"toolchain": toolchain, "lean_version": version["stdout"].strip(),
              "library_build": build, "all_states_compilation": compilation,
              "states_verified": len(cases), "cases": cases, "live_bridge_checks": checks,
              "compiled_file_sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
              "basic_policy_sha256": hashlib.sha256((project / "NISPAxioms/Basic.lean").read_bytes()).hexdigest(),
              "scope": "Four supplied hard-constraint booleans. Warning-driven REPLAN is tested in Python."}
    (output / "policy_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

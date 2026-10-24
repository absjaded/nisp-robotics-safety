"""Replay the original scenarios and write compact decision/fault traces."""
from __future__ import annotations

from collections import Counter
import contextlib
from dataclasses import asdict, replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parent

from nisp.audit import audit_cycle, audit_trace, seal
from nisp.scenarios import run_scenario, _build_telemetry
from nisp.pipeline import NISPPipeline
from nisp.evidence import EvidenceSolver
from nisp.warnings import WarningFixture
from nisp.readout import PredicateReadout
from nisp.verification import LeanVerifier
from nisp.policy import PriorityArbiter


def build_pipeline(seed=42, ablated=False):
    return NISPPipeline(object(), None if ablated else WarningFixture(seed=seed),
                        None if ablated else PredicateReadout(seed=seed),
                        EvidenceSolver(), LeanVerifier(), PriorityArbiter())


def json_safe(value):
    """Preserve rejected non-finite input visibly without writing invalid JSON."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def decision_trace(name, telemetry, z, *, warning_source="supplied warning fixture"):
    pipeline = build_pipeline(ablated=True)
    record = {"name": name, "telemetry": json_safe(asdict(telemetry)),
              "warning_source": warning_source, "supplied_Z": json_safe(z),
              "policy_sha256": hashlib.sha256((PACKAGE / "policy.py").read_bytes()).hexdigest(),
              "thresholds_sha256": hashlib.sha256((PACKAGE / "assets/config/thresholds.yaml").read_bytes()).hexdigest()}
    try:
        verdict, evidence, proof, divergence = pipeline.evaluate_distribution(telemetry, z)
    except ValueError as exc:
        # This demo boundary refuses to authorize motion on malformed input.
        record.update(status="rejected", verdict="ABORT", reason="invalid_input",
                      error=str(exc), proof=None, cycle=None)
        return seal(record)
    facts = [(f.proposition, bool(f.value), float(f.raw_reading), float(f.threshold))
             for f in evidence.facts]
    cycle = {"rho": proof.rho, "D": divergence, "facts": facts, "Z": list(map(float, z)),
             "verdict": verdict.action, "lean_latency_ms": proof.latency_ms,
             "proof_term_hash": proof.proof_term_hash,
             "certificate_hash": verdict.certificate["merkle_root"]}
    record.update(status="evaluated", verdict=verdict.action,
                  reason=verdict.certificate["trigger_reason"],
                  primary_trigger=verdict.certificate["primary_trigger"],
                  proof=asdict(proof), cycle=cycle,
                  active_constraints=[f.proposition for f in evidence.facts if f.value])
    return seal(record)


def make_traces():
    nominal = _build_telemetry({"min_dist": 1.2, "tcp_velocity": 0.05}, 0.0)
    no_warning = np.full(8, 0.125)
    warning = np.array([0.02, 0.02, 0.02, 0.02, 0.86, 0.02, 0.02, 0.02])
    low_warning = np.array([0.02, 0.02, 0.02, 0.02, 0.59, 0.02, 0.02, 0.29])
    rows = [
        decision_trace("nominal_without_warning", nominal, no_warning, warning_source="uniform ablation"),
        decision_trace("same_telemetry_with_warning", nominal, warning),
        decision_trace("warning_below_gate", nominal, low_warning),
        decision_trace("hard_violation_without_warning", replace(nominal, min_dist=0.3), no_warning),
        decision_trace("hard_violation_with_warning", replace(nominal, min_dist=0.3), warning),
        decision_trace("distance_at_inherited_limit", replace(nominal, min_dist=0.5), no_warning),
        decision_trace("nonfinite_distance", replace(nominal, min_dist=float("nan")), no_warning),
        decision_trace("missing_torque_joint", replace(nominal, tau=(0.0,) * 5), no_warning),
        decision_trace("nonfinite_warning", nominal, [float("nan")] + [0.125] * 7),
        decision_trace("unnormalized_warning", nominal, [0.1] * 8),
    ]
    assert all(audit_trace(row) for row in rows)
    expected = ["PROCEED", "REPLAN", "PROCEED", "ABORT", "ABORT", "PROCEED"] + ["ABORT"] * 4
    assert [row["verdict"] for row in rows] == expected
    return rows


def historical_checks(config, rows):
    checks = {}
    mean_rho = float(np.mean([r["rho"] for r in rows]))
    mean_d = float(np.mean([r["D"] for r in rows]))
    for key, value in config.get("expected", {}).items():
        if key == "verdict":
            checks[key] = all(r["verdict"] == value for r in rows)
        elif key == "primary_trigger":
            checks[key] = all(r["primary_trigger"] == value for r in rows)
        elif key == "rho_mean":
            checks[key] = bool(np.isclose(mean_rho, value))
        elif key == "rho_mean_lower":
            checks[key] = mean_rho >= value
        elif key == "D_mean_upper":
            checks[key] = mean_d <= value
        elif key == "D_mean_lower":
            checks[key] = mean_d >= value
        else:
            raise ValueError(f"Unrecognized historical expectation: {key}")
    return checks


def replay_one(path, seed, ablated, destination):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="nisp-replay-") as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            log = Path(run_scenario(str(path), seed, tmp, ablated))
        lines = log.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[1:]]
    assert len(rows) == config["cycles"]
    assert all(all(audit_cycle(row).values()) for row in rows)
    assert all(row["certificate_hash"][:16] in row["narration"] for row in rows)
    name = f"{path.stem}_seed{seed}_{'ablated' if ablated else 'fixture'}.jsonl.gz"
    with gzip.GzipFile(filename=str(destination / name), mode="wb", mtime=0) as handle:
        handle.write(("\n".join(lines) + "\n").encode())
    return {"scenario": path.stem, "seed": seed, "warning_enabled": not ablated,
            "cycles": len(rows), "actions": dict(Counter(r["verdict"] for r in rows)),
            "primary_triggers": dict(Counter(r["primary_trigger"] for r in rows)),
            "rho_mean": float(np.mean([r["rho"] for r in rows])),
            "D_mean": float(np.mean([r["D"] for r in rows])),
            "cache_lookup_ms_mean": float(np.mean([r["lean_latency_ms"] for r in rows])),
            "historical_expectations": config["expected"],
            "historical_checks": historical_checks(config, rows) if not ablated else None,
            "all_input_certificates_verified": True, "all_policy_replays_matched": True,
            "log": "replays/" + name}


def run(output, *, traces_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    traces = make_traces()
    (output / "decision_traces.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False, sort_keys=True) + "\n" for r in traces), encoding="utf-8")
    for row in traces:
        print(f"{row['name']:36s} {row['verdict']:7s} {row['reason']}", flush=True)
    if traces_only:
        return
    logdir = output / "replays"
    logdir.mkdir(exist_ok=True)
    runs = []
    for path in sorted((PACKAGE / "assets/scenarios").glob("*.yaml")):
        for seed in (7, 42, 123):
            for ablated in (False, True):
                row = replay_one(path, seed, ablated, logdir)
                runs.append(row)
        print(f"Replayed {path.stem}: 3 seeds, with/without warning fixture", flush=True)
    differences = [{"scenario": r["scenario"], "seed": r["seed"], "checks": r["historical_checks"]}
                   for r in runs if r["historical_checks"] and not all(r["historical_checks"].values())]
    summary = {"implementation": "NISP standalone v0", "runs": runs,
               "total_cycles": sum(r["cycles"] for r in runs), "traces": len(traces),
               "historical_differences": differences,
               "scope": "Deterministic telemetry replay and synthetic warnings; no physics rollout or learned-warning benefit is measured."}
    (output / "replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "cycles": summary["total_cycles"],
                      "historical_expectation_differences": len(differences)}, indent=2))

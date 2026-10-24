"""Recovery regressions: malformed input, policy boundaries and persisted audits."""
import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
from nisp.audit import audit_cycle, audit_trace, seal
from nisp.demo import build_pipeline, decision_trace, make_traces
from nisp.scenarios import _build_telemetry, run_scenario
from nisp.contracts import EvidenceState, ProofResult, SymbolicFact
from nisp.warnings import WarningFixture
from nisp.verification import LeanVerifier
from nisp.proof_cache import canonical_lean_source


def nominal():
    return _build_telemetry({"min_dist": 1.2, "tcp_velocity": 0.05}, 0)


@pytest.mark.parametrize("changes", [
    {"min_dist": float("nan")}, {"min_dist": float("inf")}, {"min_dist": -1},
    {"tau": (0,) * 5}, {"tau": (0, 0, 0, 0, 0, float("nan"))},
    {"q": (0,) * 7}, {"q_dot": (0,) * 5}, {"ee_pos": (0,) * 2},
    {"zone_status": 0.5}, {"zone_status": 2}, {"tcp_velocity": -0.1},
    {"tcp_velocity": float("nan")}, {"sim_time": -1},
    {"human_aabb_center": (0,) * 2}, {"closing_velocity": float("inf")},
])
def test_invalid_telemetry_rejected_before_decoding(changes):
    pipeline = build_pipeline()
    pipeline.l1_tribevx = SimpleNamespace(process=lambda _: pytest.fail("decoder must not run"))
    with pytest.raises(ValueError):
        pipeline.run_cycle(replace(nominal(), **changes))


@pytest.mark.parametrize("z", [
    [0.1] * 8, [0.125] * 7, [float("nan")] + [0.125] * 7,
    [-0.125] + [0.125] * 6 + [0.375], [float("inf")] + [0.125] * 7,
    [[0.125] * 8],
])
def test_invalid_warning_rejected(z):
    with pytest.raises(ValueError):
        build_pipeline(ablated=True).evaluate_distribution(nominal(), z)


@pytest.mark.parametrize("bits", list(itertools.product((False, True), repeat=4)))
def test_all_hard_constraint_combinations_override_warning(bits):
    proximity, torque, geofence, velocity = bits
    tel = replace(nominal(), min_dist=0.3 if proximity else 1.2,
                  tau=(350.0 if torque else 0.0,) + (0.0,) * 5,
                  zone_status=int(geofence), tcp_velocity=0.4 if velocity else 0.05)
    pipeline = build_pipeline(ablated=True)
    for warning in (np.full(8, 0.125), [0.02] * 4 + [0.86] + [0.02] * 3):
        verdict, evidence, proof, _ = pipeline.evaluate_distribution(tel, warning)
        assert verdict.action == ("ABORT" if any(bits) else
                                  "REPLAN" if warning[4] >= 0.60 else "PROCEED")
        assert tuple(f.value for f in evidence.facts) == bits
        import hashlib
        assert proof.proof_term_hash == hashlib.sha256(canonical_lean_source(bits, any(bits)).encode()).hexdigest()


@pytest.mark.parametrize("probability,expected", [(0.59, "PROCEED"), (0.60, "REPLAN"), (0.86, "REPLAN")])
def test_warning_boundary(probability, expected):
    z = np.zeros(8)
    z[4], z[7] = probability, 1 - probability
    verdict, *_ = build_pipeline(ablated=True).evaluate_distribution(nominal(), z)
    assert verdict.action == expected


@pytest.mark.parametrize("distance,expected", [(0.499999, "ABORT"), (0.5, "PROCEED"), (0.500001, "PROCEED")])
def test_inherited_distance_boundary(distance, expected):
    verdict, *_ = build_pipeline(ablated=True).evaluate_distribution(
        replace(nominal(), min_dist=distance), np.full(8, 0.125))
    assert verdict.action == expected


@pytest.mark.parametrize("latency,expected", [(50.0, "PROCEED"), (50.0001, "ABORT")])
def test_inherited_proof_timeout_boundary(latency, expected):
    pipeline = build_pipeline(ablated=True)
    pipeline.l4_formal_verifier = SimpleNamespace(verify=lambda _: ProofResult(1., 0, 1, "fixture", latency))
    verdict, *_ = pipeline.evaluate_distribution(nominal(), np.full(8, 0.125))
    assert verdict.action == expected


@pytest.mark.parametrize("proof", [ProofResult(float("nan"), 0, 1, "x", 0),
                                    ProofResult(1., 0, 1, "x", float("nan")),
                                    ProofResult(1., 1, 1, "x", 0),
                                    ProofResult(1., 0, 0, "x", 0)])
def test_invalid_proof_rejected(proof):
    pipeline = build_pipeline(ablated=True)
    pipeline.l4_formal_verifier = SimpleNamespace(verify=lambda _: proof)
    with pytest.raises(ValueError):
        pipeline.evaluate_distribution(nominal(), np.full(8, 0.125))


def test_failed_proof_cannot_authorize_nominal_motion():
    pipeline = build_pipeline(ablated=True)
    pipeline.l4_formal_verifier = SimpleNamespace(verify=lambda _: ProofResult(0., 1, 1, "fixture", 0.))
    verdict, *_ = pipeline.evaluate_distribution(nominal(), np.full(8, 0.125))
    assert verdict.action == "ABORT"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unknown"])
def test_incomplete_or_ambiguous_evidence_rejected(kind):
    pipeline = build_pipeline(ablated=True)
    evidence = pipeline.l3_evidence_solver.process(np.full(8, 0.125), nominal())
    facts = list(evidence.facts)
    if kind == "missing":
        facts.pop()
    elif kind == "duplicate":
        facts[-1] = facts[0]
    else:
        facts[-1] = replace(facts[-1], proposition="unrecognized")
    pipeline.l3_evidence_solver = SimpleNamespace(process=lambda *args: EvidenceState(tuple(facts), ()))
    with pytest.raises(ValueError):
        pipeline.evaluate_distribution(nominal(), np.full(8, 0.125))


def test_json_roundtrip_and_tampering():
    for record in make_traces():
        assert audit_trace(json.loads(json.dumps(record, allow_nan=False)))
        corrupted = copy.deepcopy(record)
        corrupted["verdict"] = "PROCEED" if record["verdict"] != "PROCEED" else "ABORT"
        assert not audit_trace(corrupted)
    record = decision_trace("normal", nominal(), np.full(8, 0.125))
    tampered = copy.deepcopy(record["cycle"])
    tampered["verdict"] = "ABORT"
    checks = audit_cycle(tampered)
    assert checks["legacy_input_root_matches"]  # The inherited root binds inputs only.
    assert not checks["policy_replay_matches"]
    changed = copy.deepcopy(record)
    changed["cycle"]["verdict"] = changed["verdict"] = "ABORT"
    assert not audit_trace(seal(changed))


@pytest.mark.parametrize("scenario", sorted((ROOT / "src/nisp/assets/scenarios").glob("*.yaml")), ids=lambda p: p.stem)
@pytest.mark.parametrize("ablated", [False, True])
def test_scenario_runner_roundtrip(tmp_path, scenario, ablated):
    config = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    config["cycles"] = 2
    miniature = tmp_path / "scenario.yaml"
    miniature.write_text(yaml.safe_dump(config), encoding="utf-8")
    log = Path(run_scenario(str(miniature), 42, str(tmp_path), ablated))
    rows = [json.loads(line) for line in log.read_text().splitlines()][1:]
    assert len(rows) == 2
    assert all(all(audit_cycle(row).values()) for row in rows)
    assert all(row["certificate_hash"][:16] in row["narration"] for row in rows)


def test_unknown_warning_fixture_is_rejected():
    with pytest.raises(ValueError, match="Unknown warning"):
        WarningFixture().generate("typo")


def test_fractional_zone_not_silently_truncated():
    tel = _build_telemetry({"zone_status": 0.5}, 0)
    with pytest.raises(ValueError, match="zone_status"):
        build_pipeline().run_cycle(tel)


def test_live_verifier_uses_same_source_as_cache(monkeypatch, tmp_path):
    import nisp.verification as bridge
    seen = []
    def fake_compile(command, **kwargs):
        seen.append(Path(command[-1]).read_text(encoding="utf-8"))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(bridge.subprocess, "run", fake_compile)
    facts = tuple(SymbolicFact(p, i == 0, float(i == 0), .5, p) for i, p in enumerate(
        ("proximity_violation", "torque_exceedance", "geofence_breach", "velocity_limit")))
    evidence = EvidenceState(facts, ())
    live = LeanVerifier(str(tmp_path), use_cache=False).verify(evidence)
    cached = LeanVerifier().verify(evidence)
    assert seen == [canonical_lean_source((True, False, False, False), True)]
    assert live.proof_term_hash == cached.proof_term_hash
    assert not list(tmp_path.iterdir())

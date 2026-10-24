"""Standalone API, CLI and package-resource regressions."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from nisp import TelemetryState, decide
from nisp.audit import audit_trace, check_paths
from nisp.cli import main


def telemetry():
    return TelemetryState(q=(0.,) * 6, q_dot=(0.,) * 6, tau=(0.,) * 6,
                          ee_pos=(.3, 0., .9), min_dist=1.2, zone_status=0,
                          sim_time=0., tcp_velocity=.05)


def test_public_api_uses_supplied_warning_and_uniform_ablation():
    assert decide(telemetry())["verdict"] == "PROCEED"
    warning = np.array([.02] * 4 + [.86] + [.02] * 3)
    result = decide(telemetry(), warning)
    assert result["verdict"] == "REPLAN"
    assert result["reason"] == "high_dissonance"
    assert audit_trace(json.loads(json.dumps(result, allow_nan=False)))
    assert decide(replace(telemetry(), min_dist=.3), warning)["verdict"] == "ABORT"


@pytest.mark.parametrize("warning", [np.ones((2, 4)), np.array(float("nan")), 0.5,
                                    [float("nan")] * 8, np.full(8, .1)])
def test_public_boundary_handles_malformed_numeric_arrays(warning):
    result = decide(telemetry(), warning)
    assert result["verdict"] == "ABORT"
    assert result["status"] == "rejected"
    assert audit_trace(json.loads(json.dumps(result, allow_nan=False)))


def test_cli_demo_and_directory_audit(tmp_path, capsys):
    main(["demo", "--output", str(tmp_path)])
    assert "REPLAN" in capsys.readouterr().out
    result = check_paths([tmp_path])
    assert result == {"records_checked": 10, "failed": 0}
    main(["audit", str(tmp_path)])
    assert json.loads(capsys.readouterr().out) == result


def test_cli_reports_corrupted_record(tmp_path, capsys):
    record = decide(telemetry())
    record["verdict"] = "ABORT"
    (tmp_path / "corrupt.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["audit", str(tmp_path)])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["failed"] == 1


def test_assets_are_package_relative():
    import nisp
    root = Path(nisp.__file__).parent / "assets"
    assert len(list((root / "scenarios").glob("*.yaml"))) == 6
    assert (root / "config/thresholds.yaml").is_file()
    assert (root / "lean/NISPAxioms/Basic.lean").is_file()

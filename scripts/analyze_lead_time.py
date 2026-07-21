"""Compute the cognitive-substrate lead time from JSONL logs.

The lead-time metric formalizes the substrate's structural property: the
cognitive substrate produces a precautionary safety response at the cycle of
activation, before any sensor-grounded violation manifests.

For each S4-Hesitant-Lead log we compute:
    t_precautionary  := first cycle where verdict in {REPLAN, ABORT}
    t_sensor_violation := first cycle where any Tier 1 fact is True
    lead_time_cycles := t_sensor_violation - t_precautionary
    lead_time_ms     := lead_time_cycles * dt * 1000

If the sensor never violates within the observation window, lead_time is
reported as a lower bound: ">= window_seconds" (the precautionary response
covers the entire observation window with no sensor-grounded backup needed).

Usage:
    python scripts/analyze_lead_time.py logs_m3/s4_hesitant_lead_*.jsonl
    python scripts/analyze_lead_time.py logs_m3/  # all matching logs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np

DT_SECONDS = 0.02  # cycle_time_step from run_scenario.py (50 Hz)
TIER1_FACTS = {
    "proximity_violation",
    "torque_exceedance",
    "geofence_breach",
    "velocity_limit",
}


def _expand_paths(args: list[str]) -> list[str]:
    out = []
    for a in args:
        if os.path.isdir(a):
            out.extend(sorted(glob.glob(os.path.join(a, "*.jsonl"))))
        else:
            out.extend(sorted(glob.glob(a)))
    return [p for p in out if "_lead_" in os.path.basename(p) and "_ablation" not in os.path.basename(p)]


def _load(path: str):
    rows = []
    header = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_header"):
                header = obj
            else:
                rows.append(obj)
    return header, rows


def _first_cycle(rows: list[dict], pred) -> Optional[int]:
    for r in rows:
        if pred(r):
            return r["cycle_id"]
    return None


def _is_precautionary(row: dict) -> bool:
    return row.get("verdict") in {"REPLAN", "ABORT"}


def _has_tier1_violation(row: dict) -> bool:
    facts = row.get("facts", []) or []
    for f in facts:
        # facts is a list of [proposition, value, raw_reading, threshold]
        if len(f) >= 2 and f[0] in TIER1_FACTS and bool(f[1]):
            return True
    return False


def analyze(path: str) -> dict:
    header, rows = _load(path)
    if not rows:
        return {"path": path, "error": "empty log"}

    cycles = len(rows)
    window_s = cycles * DT_SECONDS

    t_prec = _first_cycle(rows, _is_precautionary)
    t_sens = _first_cycle(rows, _has_tier1_violation)

    result = {
        "path": os.path.basename(path),
        "scenario": header.get("scenario") if header else "?",
        "seed": header.get("seed") if header else "?",
        "cycles": cycles,
        "window_s": window_s,
        "t_precautionary_cycle": t_prec,
        "t_sensor_violation_cycle": t_sens,
    }

    if t_prec is None:
        result["lead_time_ms"] = None
        result["note"] = "no precautionary response in window"
    elif t_sens is None:
        # Cognitive substrate fired; sensor never did.
        # Lead time is at least the remaining observation window.
        result["lead_time_ms"] = (cycles - t_prec) * DT_SECONDS * 1000.0
        result["lead_lower_bound"] = True
        result["note"] = (
            f"sensor never violated within {window_s:.2f}s window; "
            f"cognitive substrate is exclusive precautionary source"
        )
    else:
        result["lead_time_ms"] = (t_sens - t_prec) * DT_SECONDS * 1000.0
        result["lead_lower_bound"] = False
        result["note"] = "lead time directly measured"

    # Aggregate D over the window for reference
    Ds = [float(r.get("D", 0.0)) for r in rows]
    result["D_mean"] = float(np.mean(Ds))
    result["D_std"] = float(np.std(Ds))

    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="JSONL files, dirs, or globs")
    args = p.parse_args()

    paths = _expand_paths(args.paths)
    if not paths:
        sys.exit(f"no S4-lead logs matched: {args.paths}")

    results = [analyze(p) for p in paths]

    print("# S4 Cognitive-Substrate Lead-Time Analysis\n")
    print("| Seed | Cycles | t_prec | t_sens | Lead (ms) | Bound | D mean |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        lt = r.get("lead_time_ms")
        lt_str = f"{lt:.0f}" if lt is not None else "n/a"
        bound = ">=" if r.get("lead_lower_bound") else "="
        sens = r.get("t_sensor_violation_cycle")
        sens_str = str(sens) if sens is not None else "never"
        print(f"| {r['seed']} | {r['cycles']} | "
              f"{r['t_precautionary_cycle']} | {sens_str} | "
              f"{lt_str} | {bound} | {r['D_mean']:.3f} |")

    # Aggregate
    leads = [r["lead_time_ms"] for r in results if r.get("lead_time_ms") is not None]
    bounded = all(r.get("lead_lower_bound") for r in results if r.get("lead_time_ms") is not None)
    if leads:
        print()
        prefix = "≥ " if bounded else ""
        print(f"**Aggregate across {len(leads)} seeds:** "
              f"lead = {prefix}{np.mean(leads):.0f} ms "
              f"(min {prefix}{min(leads):.0f}, max {prefix}{max(leads):.0f})")
        if bounded:
            print(f"\n_Sensor channels never violate within the observation window in any seed; "
                  f"the cognitive substrate is the exclusive precautionary source._")


if __name__ == "__main__":
    main()

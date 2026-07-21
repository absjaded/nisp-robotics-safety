"""Aggregate CyclePayload JSONL logs into a Table II markdown summary.

Usage:
    python scripts/analyze_logs.py logs/*.jsonl
    python scripts/analyze_logs.py logs/  # all .jsonl in a directory

Output: prints a Markdown table per scenario with mean ± std across seeds.
Pair rows: normal runs separately from `--no-tribevx` ablation runs (the IEEE
ablation comparison).
"""
import os
import sys
import glob
import json
import argparse
from collections import defaultdict

import numpy as np


def load_log(path: str):
    """Read a JSONL log. Returns (header_dict, list_of_rows)."""
    header = {}
    rows = []
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


def aggregate(paths):
    """Group logs by (scenario, ablation_flag); collect per-seed summaries.

    Returns: dict[(scenario, ablation_flag)] -> dict with mean/std across seeds.
    """
    groups = defaultdict(list)  # (scenario, ablated) -> list of per-seed summaries
    for p in paths:
        hdr, rows = load_log(p)
        if not rows:
            continue
        scenario = hdr.get("scenario", rows[0].get("scenario_name", "UNKNOWN"))
        seed = hdr.get("seed", rows[0].get("seed", -1))
        ablated = hdr.get("ablate_tribevx", rows[0].get("tribevx_active") is False)

        rhos = np.array([r["rho"] for r in rows])
        Ds = np.array([r["D"] for r in rows])
        actions = [r["verdict"] for r in rows]
        lean_lat = np.array([r["lean_latency_ms"] for r in rows])
        wall_lat = np.array([r["wall_latency_ms"] for r in rows])

        groups[(scenario, bool(ablated))].append({
            "seed": seed,
            "cycles": len(rows),
            "rho_mean": float(rhos.mean()),
            "D_mean": float(Ds.mean()),
            "abort_rate": sum(1 for a in actions if a == "ABORT") / len(actions),
            "replan_rate": sum(1 for a in actions if a == "REPLAN") / len(actions),
            "proceed_rate": sum(1 for a in actions if a == "PROCEED") / len(actions),
            "lean_lat_mean": float(lean_lat.mean()),
            "wall_lat_mean": float(wall_lat.mean()),
            "log_file": os.path.basename(p),
        })
    return groups


def _scenario_sort_key(key):
    scenario, ablated = key
    # S1, S2, S3, S4, S5 — sort by the leading digit
    digits = "".join(c for c in scenario if c.isdigit())
    idx = int(digits) if digits else 99
    return (ablated, idx)


def format_table(groups):
    lines = []
    lines.append("# NISP Table II — Aggregated Scenario Results\n")

    # Scan groups to pull per-seed rows
    all_seeds = sorted({s["seed"] for runs in groups.values() for s in runs})
    lines.append(f"Seeds used: {all_seeds}. Cycles/scenario: "
                 f"{sorted({s['cycles'] for runs in groups.values() for s in runs})}.\n")

    def _emit_section(title, filter_ablated):
        lines.append(f"## {title}\n")
        lines.append("| Scenario | Seeds | rho (mean ± std) | D (mean ± std) | "
                     "PROCEED | REPLAN | ABORT | Lean lat (µs) |")
        lines.append("|---|---|---|---|---|---|---|---|")

        for key in sorted(groups.keys(), key=_scenario_sort_key):
            scenario, ablated = key
            if ablated != filter_ablated:
                continue
            runs = groups[key]
            rhos = np.array([r["rho_mean"] for r in runs])
            Ds = np.array([r["D_mean"] for r in runs])
            abort_rates = np.array([r["abort_rate"] for r in runs])
            replan_rates = np.array([r["replan_rate"] for r in runs])
            proceed_rates = np.array([r["proceed_rate"] for r in runs])
            lean_lats = np.array([r["lean_lat_mean"] for r in runs])
            lines.append(
                f"| **{scenario}** | {len(runs)} | "
                f"{rhos.mean():.3f} ± {rhos.std():.3f} | "
                f"{Ds.mean():.3f} ± {Ds.std():.3f} | "
                f"{proceed_rates.mean()*100:.1f}% | "
                f"{replan_rates.mean()*100:.1f}% | "
                f"{abort_rates.mean()*100:.1f}% | "
                f"{lean_lats.mean()*1000:.1f} |"
            )
        lines.append("")

    _emit_section("Main runs (L1 + L2 decoder active)", filter_ablated=False)
    if any(ablated for (_, ablated) in groups.keys()):
        _emit_section("Ablation runs (`--no-tribevx`: L1/L2 disabled, Z → uniform)",
                      filter_ablated=True)

    # Per-seed detail
    lines.append("## Per-seed detail\n")
    lines.append("| Scenario | Ablated | Seed | rho | D | PROCEED | REPLAN | ABORT | Cycles | Log |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups.keys(), key=_scenario_sort_key):
        scenario, ablated = key
        for r in sorted(groups[key], key=lambda x: x["seed"]):
            lines.append(
                f"| {scenario} | {'Y' if ablated else 'N'} | {r['seed']} | "
                f"{r['rho_mean']:.3f} | {r['D_mean']:.3f} | "
                f"{r['proceed_rate']*100:.1f}% | {r['replan_rate']*100:.1f}% | {r['abort_rate']*100:.1f}% | "
                f"{r['cycles']} | `{r['log_file']}` |"
            )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSONL files or directories containing them")
    parser.add_argument("--output", default=None,
                        help="Write markdown to this file (default: stdout)")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.jsonl"))))
        elif "*" in p or "?" in p:
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)

    if not files:
        print("No .jsonl files found.", file=sys.stderr)
        sys.exit(1)

    groups = aggregate(files)
    report = format_table(groups)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()

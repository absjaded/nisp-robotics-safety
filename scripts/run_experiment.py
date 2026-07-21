"""Full experiment sweep: every scenario × every seed, plus one ablation run per scenario.

Produces the 7,500+ data points (5 scenarios × 3 seeds × 500 cycles) required for
the IEEE submission's reproducibility claim. Ablation runs are separate so
`scripts/analyze_logs.py` can tabulate them side-by-side.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --seeds 42 --no-ablation   # quick run
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_scenario import run_scenario

SCENARIOS = [
    "scenarios/s1_nominal.yaml",
    "scenarios/s2_geofence.yaml",
    "scenarios/s3_torque.yaml",
    "scenarios/s4_hesitant.yaml",
    "scenarios/s5_combined_critical.yaml",
]

DEFAULT_SEEDS = [42, 123, 999]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output", default="logs/")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip the --no-tribevx ablation sweep")
    parser.add_argument("--ablation-only", action="store_true",
                        help="Only run the ablation sweep")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t_start = time.time()
    run_count = 0

    if not args.ablation_only:
        print(f"\n=== MAIN SWEEP: {len(SCENARIOS)} scenarios x {len(args.seeds)} seeds ===\n")
        for scenario in SCENARIOS:
            for seed in args.seeds:
                run_scenario(scenario, seed, args.output, ablate_tribevx=False)
                run_count += 1

    if not args.no_ablation:
        print(f"\n=== ABLATION SWEEP (--no-tribevx): {len(SCENARIOS)} scenarios x "
              f"{len(args.seeds)} seeds ===\n")
        for scenario in SCENARIOS:
            for seed in args.seeds:
                run_scenario(scenario, seed, args.output, ablate_tribevx=True)
                run_count += 1

    elapsed = time.time() - t_start
    print(f"\n=== EXPERIMENT COMPLETE ===")
    print(f"Runs: {run_count}  |  Cycles: {run_count * 500}  |  Wall time: {elapsed:.1f}s")
    print(f"Next: python scripts/analyze_logs.py {args.output} --output results/table_ii.md")


if __name__ == "__main__":
    main()

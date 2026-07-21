"""Compute Information Gain on logged JSONL cycles.

    python scripts/compute_ig.py logs/
    python scripts/compute_ig.py logs/*.jsonl --output results/ig.md
    python scripts/compute_ig.py logs/ --n-bootstrap 100
"""
import argparse
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nisp.learning.ig_estimator import load_and_compute, format_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSONL files or directories")
    parser.add_argument("--n-neighbors", type=int, default=5,
                        help="kNN parameter for Kraskov estimator (default 5)")
    parser.add_argument("--n-bootstrap", type=int, default=30,
                        help="Bootstrap resamples for CI (default 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Write markdown report to this path (default: stdout)")
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

    print(f"Processing {len(files)} log files...", file=sys.stderr)
    main_result, ablation_result = load_and_compute(
        files, n_neighbors=args.n_neighbors,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )

    report = format_report(main_result, ablation_result)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()

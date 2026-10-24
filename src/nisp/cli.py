"""Command-line entry points for the standalone decision prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nisp", description="Inspect decisions from sensor facts and warning signals.")
    parser.add_argument("--version", action="version", version="nisp-decision 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("demo", "Ten controlled decision and invalid-input examples"),
                            ("replay", "Replay six scenarios across three seeds and warning ablations")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output", type=Path, default=Path("runs") / name)
    audit = commands.add_parser("audit", help="Check saved traces, proof sources and decision-rule consistency")
    audit.add_argument("paths", nargs="+", type=Path)
    verify = commands.add_parser("verify-policy", help="Compile the hard-constraint policy and all 16 states in Lean")
    verify.add_argument("--output", type=Path, default=Path("runs/verification"))
    args = parser.parse_args(argv)
    if args.command in ("demo", "replay"):
        from .demo import run
        run(args.output, traces_only=args.command == "demo")
    elif args.command == "audit":
        from .audit import check_paths
        result = check_paths(args.paths)
        print(json.dumps(result, indent=2))
        if not result["records_checked"] or result["failed"]:
            raise SystemExit(1)
    else:
        from .qualification import qualify
        report = qualify(args.output)
        print(json.dumps({"states_verified": report["states_verified"],
                          "lean_version": report["lean_version"],
                          "live_bridge_checks": len(report["live_bridge_checks"])}, indent=2))


if __name__ == "__main__":
    main()

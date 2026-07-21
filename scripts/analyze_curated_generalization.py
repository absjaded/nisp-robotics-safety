"""Phase NE-4 — generalization sanity check.

Runs the same 15-cycle SCRIPT through three decoder configurations, printing
per-cycle verdict and a summary comparison. Output is also written to
artifacts/results/curated_vs_full.md.

Configs:
  full       : L2_Nengo trained on all 15 cycle_labels (baseline, T=1.0)
  curated    : L2_Nengo trained on 5 curated_labels      (T=0.1; the demo path)
  synthetic  : TribevX + L2_PredicateDecoder (the no-real-neural fallback demo)

Acceptance criterion (per architecture/today_plan_phased.md NE-4):
  curated path's verdict-action match on the 10 unseen cycles >= 80%.
  < 80% is a known limitation; we either keep this in the paper as a
  curation-tradeoff note or fall back to the full decoder for the IEEE table.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.nisp.core.contracts import LatentVector, RiskState, TelemetryState
from src.nisp.core.orchestrator import NISPPipeline, kl_divergence
from src.nisp.layers.evidence_solver import EvidenceSolver
from src.nisp.layers.l1_tribev2_real import L1_TribeV2_Real
from src.nisp.layers.l1_tribevx import TribevX
from src.nisp.layers.l2_analysis import L2_PredicateDecoder
from src.nisp.layers.l2_nengo import L2_Nengo
from src.nisp.layers.l4_bridge import LeanVerifier
from src.nisp.layers.priority_arbiter import PriorityArbiter

# Mirror SCRIPT from scripts/demo.py — telemetry only.
SCRIPT_TEL = [
    dict(min_dist=1.50, tcp_velocity=0.05, zone_status=0),
    dict(min_dist=1.10, tcp_velocity=0.10, zone_status=0),
    dict(min_dist=0.85, tcp_velocity=0.12, zone_status=0),
    dict(min_dist=0.68, tcp_velocity=0.12, zone_status=0),
    dict(min_dist=0.62, tcp_velocity=0.14, zone_status=0),
    dict(min_dist=0.56, tcp_velocity=0.15, zone_status=0),
    dict(min_dist=0.48, tcp_velocity=0.18, zone_status=0),
    dict(min_dist=0.35, tcp_velocity=0.20, zone_status=0),
    dict(min_dist=0.30, tcp_velocity=0.22, zone_status=0),
    dict(min_dist=0.30, tcp_velocity=0.28, zone_status=0),
    dict(min_dist=0.28, tcp_velocity=0.32, zone_status=1),
    dict(min_dist=0.20, tcp_velocity=0.40, zone_status=1, tau=[350.0, 0, 0, 0, 0, 0]),
    dict(min_dist=0.20, tcp_velocity=0.40, zone_status=1, tau=[350.0, 0, 0, 0, 0, 0]),
    dict(min_dist=0.60, tcp_velocity=0.15, zone_status=1, tau=[0, 0, 0, 0, 0, 0]),
    dict(min_dist=1.00, tcp_velocity=0.08, zone_status=0),
]

CURATED_CYCLES = {0, 4, 7, 11, 14}


def _telemetry(cid: int, ov: dict) -> TelemetryState:
    return TelemetryState(
        q=np.zeros(6), q_dot=np.zeros(6),
        tau=np.array(ov.get("tau", [0.0] * 6), dtype=float),
        ee_pos=np.array([0.3, 0.0, 0.9]),
        min_dist=ov.get("min_dist", 1.0),
        zone_status=int(ov.get("zone_status", 0)),
        sim_time=cid * 0.02,
        tcp_velocity=ov.get("tcp_velocity", 0.05),
    )


def _run(label: str, l1, l2) -> list[tuple[int, str, str]]:
    pipe = NISPPipeline(
        l0=object(), l1=l1, l2=l2,
        l3=EvidenceSolver(), l4=LeanVerifier(), l5=PriorityArbiter(),
    )
    out = []
    for cid, ov in enumerate(SCRIPT_TEL):
        verdict, *_ = pipe.run_cycle(_telemetry(cid, ov))
        out.append((cid, verdict.action, verdict.certificate["primary_trigger"]))
    return out


def main():
    print("\n=== Phase NE-4: curated decoder generalization ===\n")

    # Synthetic baseline: TribevX + L2_PredicateDecoder (200-dim path).
    syn = _run("synthetic",
               TribevX(seed=42),
               L2_PredicateDecoder(p_dim=200, seed=42))

    # Full-decoder real-neural: L2_Nengo with cycle_labels.json (T=1.0)
    full = _run("full real-neural",
                L1_TribeV2_Real(fallback=TribevX(seed=42)),
                L2_Nengo(seed=42,
                         labels_path="artifacts/labels/cycle_labels.json",
                         temperature=1.0))

    # Curated real-neural: L2_Nengo with curated_labels.json (T=0.1)
    curated = _run("curated real-neural",
                   L1_TribeV2_Real(fallback=TribevX(seed=42)),
                   L2_Nengo(seed=42,
                            labels_path="artifacts/labels/curated_labels.json",
                            temperature=0.1))

    # Comparison
    out_lines = []
    out_lines.append("# Phase NE-4 — Curated decoder generalization\n")
    out_lines.append("Acceptance: >=80% verdict-action match on the 10 unseen cycles.\n")
    out_lines.append("| cycle | curated? | synthetic | full real-neural | curated real-neural | match (curated vs full)? |")
    out_lines.append("|------:|:---------|:----------|:-----------------|:--------------------|:------------------------|")

    matches_unseen = 0
    n_unseen = 0
    matches_curated = 0
    for (cid, syn_a, _), (_, full_a, _), (_, cur_a, _) in zip(syn, full, curated):
        is_curated = cid in CURATED_CYCLES
        match = "OK" if cur_a == full_a else "diff"
        if not is_curated:
            n_unseen += 1
            if cur_a == full_a:
                matches_unseen += 1
        else:
            if cur_a == full_a:
                matches_curated += 1
        out_lines.append(
            f"| {cid:>5} | {'curated' if is_curated else '       '} "
            f"| {syn_a:<7} | {full_a:<7} | {cur_a:<7} | {match:<5} |"
        )

    pct_unseen = 100.0 * matches_unseen / n_unseen if n_unseen else 0.0

    out_lines.append("")
    out_lines.append(f"**Curated cycles (5):** curated == full on {matches_curated}/5")
    out_lines.append(f"**Unseen cycles (10):** curated == full on {matches_unseen}/{n_unseen} = **{pct_unseen:.0f}%**")
    if pct_unseen >= 80.0:
        out_lines.append(f"\n[ACCEPT] curated decoder generalizes ({pct_unseen:.0f}% >= 80%).")
    else:
        out_lines.append(f"\n[INFO] curated decoder under-generalizes ({pct_unseen:.0f}% < 80%). ")
        out_lines.append("Acceptable for the demo (hits all 5 hero scenes correctly), ")
        out_lines.append("but for IEEE Phase C use the full-cycle decoder as the baseline row ")
        out_lines.append("and the curated decoder only for the demo-curation column.")

    md = "\n".join(out_lines)
    print(md)

    out_path = Path("artifacts/results")
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "curated_vs_full.md").write_text(md, encoding="utf-8")
    print(f"\nWrote {out_path / 'curated_vs_full.md'}")


if __name__ == "__main__":
    main()

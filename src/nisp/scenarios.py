"""Run one scenario YAML against the NISP pipeline. Publication-grade logging.

Output: logs/<scenario>_seed<N>[_ablation]_<TIMESTAMP>.jsonl, one CyclePayload per line.

CLI:
    python scripts/run_scenario.py scenarios/s4_hesitant.yaml [--seed N] [--no-tribevx]

`--no-tribevx` is the IEEE ablation run: L1 and L2 are disabled so Z falls back
to uniform. The verdict certificate is tagged so downstream analysis can tell
which run was the ablation.
"""
import os
import sys
import time
import yaml
import argparse
import hashlib
import numpy as np
from dataclasses import asdict
from datetime import datetime


from nisp.pipeline import NISPPipeline, kl_divergence
from nisp.contracts import TelemetryState, CyclePayload, LatentVector, RiskState
from nisp.logging import NISPLogger
from nisp.warnings import WarningFixture
from nisp.readout import PredicateReadout
from nisp.evidence import EvidenceSolver
from nisp.verification import LeanVerifier
from nisp.policy import PriorityArbiter
from nisp.narrator import Narrator  # free, no network


def _build_telemetry(overrides: dict, sim_time: float) -> TelemetryState:
    """Construct TelemetryState directly from scenario YAML overrides — bypasses
    MuJoCo physics so scenario replays are deterministic and cheap."""
    tau = overrides.get("tau_override", [0.0] * 6)
    return TelemetryState(
        q=np.zeros(6),
        q_dot=np.zeros(6),
        tau=np.array(tau, dtype=float),
        ee_pos=np.array(overrides.get("ee_pos", [0.3, 0.0, 0.9])),
        min_dist=overrides.get("min_dist", 1.0),
        zone_status=overrides.get("zone_status", 0),
        sim_time=sim_time,
        closing_velocity=overrides.get("closing_velocity", 0.0),
        human_aabb_center=np.array(overrides.get("human_aabb_center", [1.0, 0.0, 0.9])),
        tcp_velocity=overrides.get("tcp_velocity", 0.05),
    )


def run_scenario(scenario_path: str, seed: int, output_dir: str, ablate_tribevx: bool,
                 narrate: bool = True):
    with open(scenario_path, "r") as f:
        scenario = yaml.safe_load(f)

    scenario_name = scenario["name"]
    cycles = int(scenario["cycles"])
    if cycles < 1:
        raise ValueError("A scenario must contain at least one cycle")
    overrides = scenario.get("telemetry_overrides", {}) or {}
    latent_override = scenario.get("latent_risk_override")

    # Config hash for reproducibility — stable under YAML dict ordering
    cfg_hash = hashlib.sha256(
        yaml.safe_dump(scenario, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Build pipeline. L1/L2 nullable for ablation.
    l1 = None if ablate_tribevx else WarningFixture(seed=seed)
    l2 = None if ablate_tribevx else PredicateReadout(seed=seed)
    l3 = EvidenceSolver()
    l4 = LeanVerifier()
    l5 = PriorityArbiter()
    pipeline = NISPPipeline(l0=object(), l1=l1, l2=l2, l3=l3, l4=l4, l5=l5)

    # Narrator is modular — free template rendering, no API calls.
    narrator = Narrator() if narrate else None

    # Output path: scenario name + seed + ablation flag + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = scenario_name.replace(" ", "_").lower()
    ablation_tag = "_ablation" if ablate_tribevx else ""
    log_name = f"{slug}_seed{seed}{ablation_tag}_{timestamp}.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_name)

    print(f"Running {scenario_name} (seed={seed}, ablate_tribevx={ablate_tribevx})")
    print(f"  config hash: {cfg_hash}  ->  {log_path}")

    results = []
    with NISPLogger(log_path) as logger:
        # Header line: scenario metadata so every log is self-describing.
        logger.log({
            "_header": True,
            "scenario": scenario_name,
            "seed": seed,
            "ablate_tribevx": ablate_tribevx,
            "cycles": cycles,
            "config_hash": cfg_hash,
            "cycle_time_step": 0.02,
            "warning_source": "uniform ablation" if ablate_tribevx else "synthetic fixture",
            "proof_mode": "boolean policy cache; see separate Lean verification report",
        })

        for i in range(cycles):
            sim_time = i * 0.02  # 50 Hz control loop
            telemetry = _build_telemetry(overrides, sim_time)

            # Latent override: force L1 to emit a specific risk state's latent.
            t_wall_start = time.perf_counter()
            if latent_override and l1 is not None:
                E = l1.generate(latent_override)
                latent = LatentVector(
                    E=E, risk_state=RiskState(latent_override), timestamp=sim_time
                )
                # Feed through L2 + L3 + L4 + L5 manually (skip L1.process)
                dist = l2.process(latent)
                Z = np.array(dist.Z)
                verdict, evidence, proof, D = pipeline.evaluate_distribution(telemetry, Z)
            else:
                verdict, latent, dist, evidence, proof, D = pipeline.run_cycle(telemetry)
                Z = np.array(dist.Z) if dist is not None else np.full(8, 0.125)

            wall_latency_ms = (time.perf_counter() - t_wall_start) * 1000.0

            # Build typed payload
            facts_tuple = tuple(
                (f.proposition, bool(f.value), float(f.raw_reading), float(f.threshold))
                for f in evidence.facts
            )
            narration_text = ""
            if narrator is not None:
                _, narration_text = narrator.narrate(verdict, Z, evidence, proof, i, sim_time)

            payload = CyclePayload(
                cycle_id=i,
                sim_time=sim_time,
                wall_timestamp=time.time(),
                scenario_name=scenario_name,
                seed=seed,
                risk_state=latent.risk_state.value if latent is not None else "ablated",
                E_norm=float(np.linalg.norm(latent.E)) if latent is not None else 0.0,
                Z=tuple(float(z) for z in Z),
                facts=facts_tuple,
                rho=float(proof.rho),
                D=float(D),
                verdict=verdict.action,
                primary_trigger=verdict.certificate["primary_trigger"],
                trigger_reason=verdict.certificate["trigger_reason"],
                wall_latency_ms=float(wall_latency_ms),
                sim_latency_ms=0.0,  # Phase 2b §2.4 sim-pause not wired yet
                lean_latency_ms=float(proof.latency_ms),
                certificate_hash=verdict.certificate["merkle_root"],
                tribevx_active=(l1 is not None),
                neuronax_active=(l2 is not None),
                narration=narration_text,
                proof_term_hash=proof.proof_term_hash,
            )
            logger.log(payload)
            results.append(payload)

            if i > 0 and i % 100 == 0:
                print(f"  cycle {i}/{cycles}")

    # Summary — also the data for Table II.
    rhos = np.array([r.rho for r in results])
    Ds = np.array([r.D for r in results])
    actions = [r.verdict for r in results]
    abort_rate = sum(1 for a in actions if a == "ABORT") / len(actions)
    replan_rate = sum(1 for a in actions if a == "REPLAN") / len(actions)
    proceed_rate = sum(1 for a in actions if a == "PROCEED") / len(actions)

    print(f"\n{scenario_name}  (seed={seed}{', ABLATED' if ablate_tribevx else ''})")
    print(f"  rho      mean={rhos.mean():.4f}  std={rhos.std():.4f}")
    print(f"  D        mean={Ds.mean():.4f}  std={Ds.std():.4f}")
    print(f"  PROCEED  {proceed_rate*100:5.1f}%")
    print(f"  REPLAN   {replan_rate*100:5.1f}%")
    print(f"  ABORT    {abort_rate*100:5.1f}%")
    print(f"  log -> {log_path}\n")

    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="Path to scenario YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="logs/")
    parser.add_argument("--no-tribevx", action="store_true",
                        help="IEEE ablation: disable L1/L2, Z -> uniform.")
    parser.add_argument("--no-narrate", action="store_true",
                        help="Skip the free template narrator (otherwise on by default).")
    args = parser.parse_args()
    run_scenario(args.scenario, args.seed, args.output, args.no_tribevx,
                 narrate=not args.no_narrate)

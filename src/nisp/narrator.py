"""NISP Narrator — GATE-04 template rendering of safety verdicts.

Free by construction: no network calls, no API keys, zero external dependencies
beyond the standard library + PyYAML (already in requirements). Pure function
of the per-cycle data contracts.

Modularity invariant: this module does not import from `src.nisp.layers` or
`nisp.pipeline`. It only reads the core dataclasses (contracts.py)
and the predicate vocabulary (predicates.yaml). If anything else imports from
here, the pipeline's runtime path must stay uncoupled — we only want the
runner and explicit interpretability tools calling in.

Usage:
    from nisp.narrator import Narrator
    narrator = Narrator()
    record, text = narrator.narrate(verdict, Z, evidence, proof, cycle_id, sim_time)
"""
import os
from typing import Tuple

import numpy as np
import yaml

from nisp.contracts import (
    EvidenceState,
    NarrationRecord,
    PredicateDistribution,
    ProofResult,
    SafetyVerdict,
)


class Narrator:
    def __init__(self, predicates_path: str = None):
        if predicates_path is None:
            predicates_path = os.path.join(
                os.path.dirname(__file__), "assets", "config", "predicates.yaml"
            )
        with open(predicates_path, "r") as f:
            cfg = yaml.safe_load(f)
        # index -> (name, display, tier)
        self.predicates = {
            p["index"]: (p["name"], p["display"], p["tier"]) for p in cfg["predicates"]
        }

    def _top_predicate(self, Z) -> Tuple[int, str, str, int, float]:
        Z_arr = np.asarray(Z)
        idx = int(np.argmax(Z_arr))
        name, display, tier = self.predicates[idx]
        return idx, name, display, tier, float(Z_arr[idx])

    def _primary_evidence(self, evidence: EvidenceState, verdict: SafetyVerdict):
        """Pick the fact that drove the verdict. Matches verdict.primary_trigger
        when the trigger is a proposition name; otherwise falls back to the
        first violating fact, or the first fact at all."""
        trigger = verdict.certificate.get("primary_trigger", "")
        if trigger and trigger != "nominal":
            for f in evidence.facts:
                if f.proposition == trigger:
                    return f
        for f in evidence.facts:
            if f.value:
                return f
        return evidence.facts[0] if evidence.facts else None

    def narrate_record(
        self,
        verdict: SafetyVerdict,
        Z,
        evidence: EvidenceState,
        proof: ProofResult,
        cycle_id: int,
        sim_time: float,
    ) -> NarrationRecord:
        _, top_name, _, top_tier, top_conf = self._top_predicate(Z)
        fact = self._primary_evidence(evidence, verdict)
        return NarrationRecord(
            sim_time=sim_time,
            cycle_id=cycle_id,
            verdict=verdict.action,
            trigger=verdict.certificate.get("trigger_reason", "nominal"),
            top_predicate_name=top_name,
            top_predicate_confidence=top_conf,
            top_predicate_tier=top_tier,
            primary_evidence_proposition=fact.proposition if fact else "",
            primary_evidence_value=bool(fact.value) if fact else False,
            primary_evidence_reading=float(fact.raw_reading) if fact else 0.0,
            primary_evidence_threshold=float(fact.threshold) if fact else 0.0,
            primary_evidence_channel=fact.source_channel if fact else "",
            rho=float(proof.rho),
            D=float(verdict.D),
            certificate_hash=str(verdict.certificate["merkle_root"]),
        )

    def render(self, r: NarrationRecord) -> str:
        """GATE-04 template. Frozen format — log consumers parse this."""
        display_map = {idx: disp for idx, (_, disp, _) in self.predicates.items()}
        top_display = next(
            (disp for (name, disp, _) in self.predicates.values() if name == r.top_predicate_name),
            r.top_predicate_name,
        )
        evidence_line = (
            f"{r.primary_evidence_proposition} = {r.primary_evidence_value} "
            f"(reading {r.primary_evidence_reading:.3f} vs threshold "
            f"{r.primary_evidence_threshold:.3f} on channel '{r.primary_evidence_channel}')"
        ) if r.primary_evidence_proposition else "no evidence"
        return (
            f"[t={r.sim_time:6.2f}s | cycle {r.cycle_id:4d}] {r.verdict}  ({r.trigger})\n"
            f"  Warning: top predicate = '{top_display}' "
            f"(Tier {r.top_predicate_tier}, confidence {r.top_predicate_confidence:.2%})\n"
            f"  Evidence: {evidence_line}\n"
            f"  Metrics: rho = {r.rho:.3f}, D = {r.D:.3f}\n"
            f"  Cert   : {r.certificate_hash[:16]}..."
        )

    def narrate(
        self,
        verdict: SafetyVerdict,
        Z,
        evidence: EvidenceState,
        proof: ProofResult,
        cycle_id: int,
        sim_time: float,
    ) -> Tuple[NarrationRecord, str]:
        record = self.narrate_record(verdict, Z, evidence, proof, cycle_id, sim_time)
        return record, self.render(record)

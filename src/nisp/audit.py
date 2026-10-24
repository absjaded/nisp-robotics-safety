"""Independent checks on persisted decision records, including decision binding."""
from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path

from nisp.certificates import verdict_merkle_root
from nisp.proof_cache import canonical_lean_source

PROPOSITIONS = ("proximity_violation", "torque_exceedance", "geofence_breach", "velocity_limit")


def record_digest(record):
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seal(record):
    record = dict(record)
    record["record_sha256"] = record_digest(record)
    return record


def expected_action(row):
    """Independent Python replay of the inherited arbiter, not a Lean theorem."""
    if any(fact[1] for fact in row["facts"]):
        return "ABORT"
    if row["lean_latency_ms"] > 50.0:
        return "ABORT"
    incomplete = abs(row["rho"] - 1.0) >= 1e-9
    if any(row["Z"][i] >= 0.60 for i in (4, 5, 6)) and (incomplete or row["D"] >= 0.30):
        return "REPLAN"
    return "ABORT" if incomplete else "PROCEED"


def audit_cycle(row):
    facts = tuple((str(p), bool(v), float(r), float(t)) for p, v, r, t in row["facts"])
    if len(facts) != 4 or {f[0] for f in facts} != set(PROPOSITIONS):
        raise ValueError("Incomplete or duplicated hard-constraint evidence")
    z = tuple(float(v) for v in row["Z"])
    root = verdict_merkle_root(float(row["rho"]), float(row["D"]), facts, z)
    values = {f[0]: f[1] for f in facts}
    bits = tuple(values[p] for p in PROPOSITIONS)
    proof_hash = hashlib.sha256(canonical_lean_source(bits, any(bits)).encode()).hexdigest()
    return {"legacy_input_root_matches": root == row["certificate_hash"],
            "proof_source_matches": row["proof_term_hash"] == proof_hash,
            "policy_replay_matches": expected_action(row) == row["verdict"]}


def audit_trace(record):
    if record_digest(record) != record.get("record_sha256"):
        return False
    if record["status"] == "rejected":
        return (record["verdict"] == "ABORT" and record["reason"] == "invalid_input"
                and record["proof"] is None and record["cycle"] is None)
    checks = audit_cycle(record["cycle"])
    return (all(checks.values()) and record["verdict"] == record["cycle"]["verdict"]
            and record["proof"]["rho"] == record["cycle"]["rho"]
            and record["proof"]["proof_term_hash"] == record["cycle"]["proof_term_hash"])


def check_paths(given_paths):
    checked = failed = 0
    for given in map(Path, given_paths):
        paths = (sorted(given.glob("*.jsonl")) + sorted(given.glob("*.jsonl.gz"))
                 + sorted(given.glob("replays/*.jsonl.gz"))) if given.is_dir() else [given]
        if not paths:
            raise ValueError(f"No audit records found: {given}")
        for path in paths:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        if row.get("_header"):
                            continue
                        valid = audit_trace(row) if "record_sha256" in row else all(audit_cycle(row).values())
                    except (ValueError, KeyError, TypeError, IndexError):
                        valid = False
                    checked += 1
                    failed += int(not valid)
    return {"records_checked": checked, "failed": failed}

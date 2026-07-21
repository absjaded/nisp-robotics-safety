# In-Silico Robotics Safety Stack

NISP is on robotics safety. It uses a simulated cognitive-side signal, typed evidence contracts, Lean 4 proof discharge, and per-cycle certificate hashes to make robot safety decisions auditable.

## What It Shows

- A layered safety pipeline with typed contracts from input evidence to verdict.
- Lean 4 proof checks for safety obligations.
- Per-cycle Merkle-style certificate hashes for auditability.
- Scenario harnesses for nominal, geofence, torque, hesitant, and combined-critical regimes.
- A real-data anchor using UCF-Crime road-accident clips through the same contract surface.

## Main Results

- Cobot scenario suite: 5 scenarios x 3 seeds x 500 cycles.
- Fault isolation: 100% across the reported scenario suite.
- Cognitive-side lead regime: at least 10 seconds in the S4 lead-time stress test, with sensors held in-envelope.
- UCF-Crime road-accident probe: selected SVM-RBF AUC about 0.827 on 53 clips.
- Certificate audit: 53/53 UCF-Crime certificates re-derived from committed evidence.

## Files

- `src/`: typed contracts, layer implementations, verifier bridge, certificate logic.
- `lean4_project/`: Lean 4 proof layer.
- `scenarios/`: stress-test scenario definitions.
- `results/` and `artifacts/results/`: retained result summaries.
- `tests/`: unit and pipeline tests.

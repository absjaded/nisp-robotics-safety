# CODEBASE STATUS
_Last updated: 2026-04-25 (Iteration 9 - Integration Push)_

## Current state (short)

Pipeline is **fully integrated** with both synthetic and real-neural paths.
All 37 tests pass, including Nengo SNN decoder, file-lookup L1, and Merkle integrity.
The system is hackathon-ready with a `--real-neural` and `--curated` demo.

---

## What this project is

**NISP Engine v0.2** - a 6-layer neuro-symbolic safety pipeline for collaborative robotics.
A UR10e robot arm operates near a human. NISP intercepts unsafe actions 340ms before
physical failure by reading a simulated expert brain response to the visual scene,
decoding it into formal safety predicates, and running a Lean 4 proof per cycle.

---

## Layer map

| Layer | File | Status | Notes |
|-------|------|--------|-------|
| L0 MuJoCo Adapter | `src/nisp/layers/l0_ingestion.py` | OK OK | Correctly named and integrated. |
| L1 TribevX | `src/nisp/layers/l1_tribevx.py` | OK OK | Gaussian Sampler fallback for synthetic path. |
| L1 TRIBE v2 Real | `src/nisp/layers/l1_tribev2_real.py` | OK OK | File-lookup for precomputed TRIBE v2 latents. |
| L2 NEURONAX | `src/nisp/layers/l2_analysis.py` | OK OK | Linear Probes (K=8) with 1024-dim DiFuMo support. |
| L2 Nengo | `src/nisp/layers/l2_nengo.py` | OK OK | SNN decoder: DiFuMo -> PCA(14) -> 256 LIF neurons -> Z. |
| L3 Evidence Solver | `src/nisp/layers/evidence_solver.py` | OK OK | Rule-based factification using `thresholds.yaml`. |
| L4 Lean 4 Verifier | `src/nisp/layers/l4_bridge.py` | OK OK | Live Lean 4 + 16-state cache (`l4_cache.py`). |
| L5 Priority Arbiter | `src/nisp/layers/priority_arbiter.py` | OK OK | ISO 10218-2 state machine + Merkle Root integrity. |
| Orchestrator | `src/nisp/core/orchestrator.py` | OK OK | Supports sequential execution of all 6 layers. |

---

## Lean 4 project

| File | Status | Notes |
|------|--------|-------|
| `lean4_project/NISPAxioms/Basic.lean` | OK OK | `safety_check` + `safety_sound` certified. |
| `lean4_project/Runtime.lean` | WARNING GENERATED | Auto-generated each cycle. Not source. |
| `lean4_project/lakefile.lean` | OK OK | |

---

## Integrity & Crypto

| File | Purpose |
|------|---------|
| `src/nisp/crypto/merkle.py` | 4-leaf Merkle tree over `(rho, D, facts, Z)` for auditability. |

---

## Configuration

| File | Purpose |
|------|---------|
| `src/nisp/config/thresholds.yaml` | Sensor violation thresholds. |
| `src/nisp/config/predicates.yaml` | K=8 predicate vocabulary definition. |

---

## Tests (current state)

```
37 tests passed.
Includes:
- test_l2_nengo_pipeline.py
- test_l1_tribev2_real.py
- test_merkle.py
- test_pipeline_real_neural.py
- test_ig_estimator.py
```

Run: `python -m pytest tests/ -v`

---

## What to build next (Lane C)

```
1. Create src/nisp/learning/oas.py  ->  Outcome-Anchored Supervision replay
2. Create tests/test_oas.py
3. Run 3-seed ablation experiments for IEEE
4. Compute final IG report
```

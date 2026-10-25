# NISP

An inspectable robot-decision component for combining sensor constraints with supplied warning signals.

NISP asks: **given a learned warning, how should a robot act on it inspectably?** This package implements the receiving machinery. It turns sensor readings and an eight-predicate warning distribution into **PROCEED, REPLAN or ABORT**, together with the evidence and rule behind that decision. Hard constraints take priority; an activated warning can request reconsideration under the declared policy while the sensors remain within their limits.

$$
x\xrightarrow{E}e,\qquad (e,z)\xrightarrow{\pi}(a,\text{decision trace}).
$$

Here $x$ is telemetry, $z$ is the supplied warning distribution, and $e$ records the four hard constraints. The Python implementation applies the decision rules; Lean checks the underlying hard-constraint policy over supplied booleans. The bundled synthetic warnings let you inspect how changing a warning changes a decision.

The original NISP architecture also motivated a neural-eDSL decoder. Its partial integration exposed a separate translation question: could a learned expression faithfully describe its input? This release isolates the decision component. The [methods](METHODS.md#role-and-interface-boundary) distinguish that component from the unfinished learned-warning and compiler integration.

## Run

```bash
python -m pip install .
nisp demo
```

The demo keeps the sensor readings fixed while changing the warning, then introduces a hard violation and malformed inputs:

```text
nominal_without_warning              PROCEED nominal
same_telemetry_with_warning          REPLAN  high_dissonance
hard_violation_without_warning       ABORT   tier1_violation
hard_violation_with_warning          ABORT   tier1_violation
nonfinite_distance                   ABORT   invalid_input
missing_torque_joint                 ABORT   invalid_input
```

The ten complete examples are written to `runs/demo/decision_traces.jsonl`. They include readings, warning probabilities, constraint facts, decision, reason and verification metadata. Input rejection is recorded explicitly.

```bash
nisp audit runs/demo
```

The demo runs on a CPU with NumPy and PyYAML. Fixed telemetry and controlled synthetic warnings isolate the behavior of the decision policy. The ten traces demonstrate warning-driven reconsideration, hard-constraint priority, threshold behavior and rejection of malformed inputs.

## Use your own inputs

```python
from nisp import TelemetryState, decide

telemetry = TelemetryState(
    q=(0.0,) * 6, q_dot=(0.0,) * 6, tau=(0.0,) * 6,
    ee_pos=(0.3, 0.0, 0.9), min_dist=1.2,
    zone_status=0, sim_time=0.0, tcp_velocity=0.05,
)
warning = [0.02, 0.02, 0.02, 0.02, 0.86, 0.02, 0.02, 0.02]
trace = decide(telemetry, warning)
print(trace["verdict"], trace["reason"])  # REPLAN high_dissonance
```

`warning` contains eight probabilities summing to one. Indices 0–3 correspond to hard-constraint predicates, 4–6 to additional warnings, and 7 to nominal activity. Sensor facts are evaluated independently of these supplied probabilities. Omitting the warning uses a uniform ablation. The returned trace is a software decision; the caller controls how it affects the robot.

## Results

| Check | Result |
|---|---|
| Python tests | **93 passed**, including API, CLI, input rejection and policy-boundary regressions |
| Scenario replay | **18,000 cycles**, six scenarios × three seeds × two warning conditions |
| Controlled warning intervention | Same in-limit telemetry: **PROCEED → REPLAN** when the warning is supplied |
| Hard-constraint policy | **16/16** boolean states compiled in Lean 4.16.0 |
| Finite-input comparison with the historical implementation | **18,000/18,000** decisions match, including divergence scores and input certificates |
| Malformed-input regression | NaN distance and incomplete torque vectors are rejected before a normal decision |

The [saved records](results/), [package verification](results/package_verification.json) and [tests](tests/) support these software checks. The 18,000 cycles repeat six procedural scenarios; they are replay coverage rather than independent safety events. Two historical scenario targets differ from the implementation: the warning-only example requests ABORT while the rule produces REPLAN; the combined-critical example requests high divergence even though its warning agrees with the sensors. Both are retained in the [replay summary](results/replay_summary.json) and explained in [Methods](METHODS.md).

## Reproduce

```bash
python -m pip install '.[test]'
python -m pytest -q
nisp audit results
nisp replay --output runs/replay
nisp audit runs/replay
```

The replay uses the same six supplied scenarios and seeds 7, 42 and 123. It saves compressed per-cycle records and a summary, checking input certificates, canonical proof sources and decision-rule consistency as it runs.

With Lean 4.16.0 installed through elan:

```bash
nisp verify-policy --output runs/verification
```

This compiles all 16 hard-constraint cases and checks the live compiler bridge against the runtime cache. The Lean source is [included in the package](src/nisp/assets/lean/NISPAxioms/Basic.lean). Its theorem covers the four supplied booleans; warning-driven REPLAN is covered by Python tests. Offline compilation and cache-lookup timings are reported separately.

## Implementation

The core paths are [input validation](src/nisp/validation.py), [sensor evidence](src/nisp/evidence.py), [decision policy](src/nisp/policy.py), [proof cache](src/nisp/proof_cache.py) and [record audit](src/nisp/audit.py). [Methods](METHODS.md) describes the probability interface, divergence, policy boundaries and evidence provenance.

The implemented contribution is an inspectable evidence-to-decision path: a warning can affect the decision, hard constraints retain priority, and the rule can be replayed against the trace. Establishing the warning's additional predictive value is a separate empirical question. Configured limits and timing budgets are prototype assumptions; physical deployment requires application-specific validation.

The [relation to existing work](METHODS.md#relation-to-existing-work) places this decision interface alongside supervisory control and runtime assurance, with an explicit account of which part of the policy is checked.

Code is available under the [MIT license](LICENSE).

The [earlier integrated research snapshot](https://github.com/absjaded/nisp-robotics-safety/tree/d2ebea67707fa7c51bb91149f839d886b9c5f9d0)
remains accessible in Git history, including the original scenarios and integration scripts.

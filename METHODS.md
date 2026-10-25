# Decision policy and evaluation

## Role and interface boundary

NISP's systems question is how to handle a learned warning once supplied. Implementing the receiving machinery made the warning's information content and faithful translation explicit dependencies. The neural-eDSL decoder was developed within the broader project and only partly integrated. This package extracts the runnable decision and audit component from that work.

The public interface accepts telemetry and an eight-predicate probability vector. It evaluates hard sensor facts, compares the warning with a sensor-derived reference distribution, applies ordered decision rules and records the result. The included warning generator and predicate readout are controlled fixtures, providing a repeatable way to exercise that interface.

The recovered compiler's four scene/robot expressions and its richer typed scene programs are different interfaces from this probability vector. This release supplies no adapter that establishes a completed compiler-to-robot pipeline. Accepting a warning says how the policy handles it; whether a learned warning adds useful information requires its own held-out evaluation.

## Inputs and evidence

Telemetry supplies six joint positions, velocities and torques, a three-coordinate end-effector position, distance to a person, a zone flag, speed and time. The input boundary checks shape, finiteness and flag/range constraints. Invalid input raises a validation error in the low-level pipeline; `decide()` records an explicit ABORT with `reason="invalid_input"` and no proof attached.

The evidence solver evaluates distance, joint torque, geofence and speed independently of the warning vector. The configured prototype limits are in [thresholds.yaml](src/nisp/assets/config/thresholds.yaml). The inherited inequalities are strict: distance below 0.5 m violates the limit, while exactly 0.5 m does not. The replay preserves this behavior.

The eight warning predicates are defined in [predicates.yaml](src/nisp/assets/config/predicates.yaml). Indices 4–6 are the additional-warning branch. A value at least 0.60 activates that branch. These values are probabilities over predicates, not calibrated probabilities of a future collision.

## Disagreement and action

The pipeline forms a reference distribution $p$ over the same eight predicates. Each active hard fact receives mass 0.99 and each remaining predicate receives 0.01. If there are no active hard facts, the nominal predicate receives 0.99 instead. The vector is normalized.

For warning distribution $z$, disagreement is

$$
D=\sum_{k=1}^{8}\widetilde z_k\log\frac{\widetilde z_k}{\widetilde p_k},
\qquad \widetilde v_k=\min(1,\max(10^{-8},v_k)).
$$

This preserves the implementation's epsilon clipping. Its score measures distributional disagreement. The uniform ablation can have high disagreement without activating a warning predicate.

The arbiter applies these rules in order:

1. Any active hard fact gives ABORT, with the primary trigger ordered distance, geofence, torque, speed.
2. Proof-cache latency over 50 ms gives ABORT.
3. An activated additional warning plus either $D\geq0.30$ or incomplete proof metadata gives REPLAN.
4. Otherwise complete proof metadata gives PROCEED, and incomplete metadata gives ABORT.

The proof score is a goal-completion ratio, with incompleteness detected at tolerance $10^{-9}$. It is not a perception-confidence estimate. The configured priorities, limits and latency budget are inherited prototype choices.

## Formal policy

Lean's `safety_check` takes four supplied booleans. It returns ABORT when any hard constraint is violated and PROCEED otherwise. `safety_sound` and `safety_nominal` prove those implications. The qualification command builds the library and compiles the canonical cached theorem for every one of the $2^4=16$ states. It also exercises live compilation for the nominal and all-violation cases.

For the four violation flags $e_j$, the policy checked by Lean is

$$
\mathrm{safety\_check}(e)=
\begin{cases}
\mathrm{ABORT}, & \bigvee_{j=1}^{4}e_j,\\
\mathrm{PROCEED}, & \text{otherwise}.
\end{cases}
$$

The Python arbiter wraps this hard-constraint policy with warning-driven REPLAN, incomplete-proof handling and a cache-latency budget. Thus a nominal Lean result can coexist with a Python REPLAN. Qualification checks the policy's behavior on supplied facts; sensor truth, warning accuracy and physical-system behavior remain separate validation questions.

At runtime the default verifier looks up the corresponding canonical source hash. The qualifier compiles that same source. Hashes identify the source used; they are separate from the claim that its compilation succeeded, recorded in the qualification report. The learned warning and REPLAN rule are outside the Lean theorem and are tested separately.

## Scenarios and comparisons

The six YAML scenarios supply fixed telemetry for 500 cycles. Three random seeds are used, with the synthetic warning fixture enabled or replaced by a uniform distribution: 36 runs and 18,000 cycles. Replay time is a scenario clock with 0.02 s steps. There is no physics rollout or measured future event in these scenarios.

The cycle count describes repeatable software coverage. It should not be read as 18,000 independent robot trials, an estimate of collision risk or a learned-warning benchmark. The paired warning-on/uniform runs test how the receiving rules respond to the intervention.

The fixture samples features around hand-defined centroids. The predicate readout uses hand-structured weights. This tests the routing and arbitration of warnings. It supplies no independent evidence of an advantage over sensor history.

The nominal case proceeds. The geofence, torque, near-distance and combined cases abort, with and without the fixture. The additional warning-only case uses nominal telemetry and a forced warning: all 1,500 warning-enabled cycles replan, while their uniform-ablation counterparts proceed.

Two historical expectations are preserved for inspection:

- `s4_hesitant_lead` requests ABORT. The implemented additional-warning rule returns REPLAN.
- `s5_combined_critical` requests mean divergence above 0.30. Its observed mean is around 0.01 because the critical fixture agrees with the violated hard constraints. Its ABORT outcome and trigger priority match the policy.

The ordinary hesitant scenario has a distance of 0.35 m. That hard violation explains its ABORT even without the additional warning. No warning lead-time claim is inferred from this replay.

## Relation to existing work

Supervisory architectures provide a useful comparison for integrating learned
components into a decision process. The
[Neural Simplex Architecture](https://arxiv.org/abs/1908.00528)
(Phan et al., 2020) uses a decision module to switch between a neural controller
and a baseline controller, with switching conditions tied to recoverability of
the physical system. It also studies retraining and return to the neural
controller. This establishes a substantial precedent for keeping a learned
component within an explicitly checked supervisory architecture.

[A Formal Verification Framework for Runtime Assurance](https://ntrs.nasa.gov/citations/20230017350)
(Slagel et al., 2024) verifies a Simplex framework using hybrid programs in PVS,
including discrete decisions, continuous dynamics and monitor sampling.
NISP's released formal object is the hard-constraint policy on four supplied
booleans. The warning-driven REPLAN rule and trace consistency have their
separate Python checks. The supplied-input policy and a closed-loop physical
system have different proof obligations.

NISP contributes a compact implementation for inspecting that decision
boundary. The controlled warning intervention holds telemetry fixed and
changes PROCEED to REPLAN; a hard violation takes priority in either warning
condition. The package also rejects malformed inputs and replays decisions
against their evidence records. These cases let a developer inspect the
receiving policy before attaching a learned warning source.

The original application sought warnings with useful information beyond
telemetry. A matched held-out comparison of telemetry-only, warning-only and
combined predictors would test that informational premise. The current replay
qualifies the decision interface on supplied inputs.

## Records and provenance

The inherited input certificate hashes `(rho, D, facts, Z)`. It binds those inputs but omits the selected action. The audit additionally replays the Python decision rule and checks the canonical proof-source hash. Compact decision traces also carry a digest of the entire record. These are consistency checks with no signer authentication.

The standalone package is extracted from the tested recovery of historical snapshot `d2ebea67707fa7c51bb91149f839d886b9c5f9d0`. Module names and resource paths were made portable. The original and recovery trees remain separate. The [provenance record](results/provenance.json) maps extracted modules to recovery-file hashes.

Before packaging, two isolated subprocesses compared 18,000 finite-input decisions against the unchanged historical snapshot; the decision streams, divergence values and input hashes matched exactly. The [saved comparison](results/legacy_comparison.json) records that result and the two malformed-input counterexamples. The current package is checked against the saved replay records as part of repository verification.

Only source code, procedural scenarios and their generated traces are included. UCF-derived features, surveillance clips, TRIBE weights and simulator assets remain outside this package. The MIT notice from the original code is retained.

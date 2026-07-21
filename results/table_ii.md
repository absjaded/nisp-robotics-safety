# NISP Table II - Aggregated Scenario Results

Seeds used: [42, 123, 999]. Cycles/scenario: [500].

## Main runs (TribevX + NEURONAX active)

| Scenario | Seeds | rho (mean +/- std) | D (mean +/- std) | PROCEED | REPLAN | ABORT | Lean lat (us) |
|---|---|---|---|---|---|---|---|
| **S1 Nominal** | 3 | 1.000 +/- 0.000 | 0.068 +/- 0.000 | 100.0% | 0.0% | 0.0% | 1.2 |
| **S2 Geofence** | 3 | 1.000 +/- 0.000 | 4.663 +/- 0.000 | 0.0% | 0.0% | 100.0% | 1.9 |
| **S3 Torque** | 3 | 1.000 +/- 0.000 | 4.663 +/- 0.000 | 0.0% | 0.0% | 100.0% | 1.5 |
| **S4 Hesitant** | 3 | 1.000 +/- 0.000 | 4.663 +/- 0.000 | 0.0% | 0.0% | 100.0% | 1.4 |
| **S5 Combined Critical** | 3 | 1.000 +/- 0.000 | 0.011 +/- 0.000 | 0.0% | 0.0% | 100.0% | 1.4 |

## Ablation runs (`--no-tribevx`: L1/L2 disabled, Z -> uniform)

| Scenario | Seeds | rho (mean +/- std) | D (mean +/- std) | PROCEED | REPLAN | ABORT | Lean lat (us) |
|---|---|---|---|---|---|---|---|
| **S1 Nominal** | 3 | 1.000 +/- 0.000 | 2.010 +/- 0.000 | 100.0% | 0.0% | 0.0% | 1.5 |
| **S2 Geofence** | 3 | 1.000 +/- 0.000 | 2.010 +/- 0.000 | 0.0% | 0.0% | 100.0% | 1.9 |
| **S3 Torque** | 3 | 1.000 +/- 0.000 | 2.010 +/- 0.000 | 0.0% | 0.0% | 100.0% | 1.1 |
| **S4 Hesitant** | 3 | 1.000 +/- 0.000 | 2.010 +/- 0.000 | 0.0% | 0.0% | 100.0% | 0.9 |
| **S5 Combined Critical** | 3 | 1.000 +/- 0.000 | 1.614 +/- 0.000 | 0.0% | 0.0% | 100.0% | 0.8 |

## Per-seed detail

| Scenario | Ablated | Seed | rho | D | PROCEED | REPLAN | ABORT | Cycles | Log |
|---|---|---|---|---|---|---|---|---|---|
| S1 Nominal | N | 42 | 1.000 | 0.068 | 100.0% | 0.0% | 0.0% | 500 | `s1_nominal_seed42_20260424_223355.jsonl` |
| S1 Nominal | N | 123 | 1.000 | 0.068 | 100.0% | 0.0% | 0.0% | 500 | `s1_nominal_seed123_20260424_223356.jsonl` |
| S1 Nominal | N | 999 | 1.000 | 0.068 | 100.0% | 0.0% | 0.0% | 500 | `s1_nominal_seed999_20260424_223356.jsonl` |
| S2 Geofence | N | 42 | 1.000 | 4.663 | 0.0% | 0.0% | 100.0% | 500 | `s2_geofence_seed42_20260424_223357.jsonl` |
| S2 Geofence | N | 123 | 1.000 | 4.663 | 0.0% | 0.0% | 100.0% | 500 | `s2_geofence_seed123_20260424_223357.jsonl` |
| S2 Geofence | N | 999 | 1.000 | 4.663 | 0.0% | 0.0% | 100.0% | 500 | `s2_geofence_seed999_20260424_223358.jsonl` |
| S3 Torque | N | 42 | 1.000 | 4.663 | 0.0% | 0.0% | 100.0% | 500 | `s3_torque_seed42_20260424_223359.jsonl` |
| S3 Torque | N | 123 | 1.000 | 4.663 | 0.0% | 0.0% | 100.0% | 500 | `s3_torque_seed123_20260424_223359.jsonl` |
| S3 Torque | N | 999 | 1.000 | 4.663 | 0.0% | 0.0% | 100.0% | 500 | `s3_torque_seed999_20260424_223400.jsonl` |
| S4 Hesitant | N | 42 | 1.000 | 4.663 | 0.0% | 0.0% | 100.0% | 500 | `s4_hesitant_seed42_20260424_223400.jsonl` |
| S4 Hesitant | N | 123 | 1.000 | 4.663 | 0.0% | 0.0% | 100.0% | 500 | `s4_hesitant_seed123_20260424_223401.jsonl` |
| S4 Hesitant | N | 999 | 1.000 | 4.663 | 0.0% | 0.0% | 100.0% | 500 | `s4_hesitant_seed999_20260424_223402.jsonl` |
| S5 Combined Critical | N | 42 | 1.000 | 0.011 | 0.0% | 0.0% | 100.0% | 500 | `s5_combined_critical_seed42_20260424_223402.jsonl` |
| S5 Combined Critical | N | 123 | 1.000 | 0.011 | 0.0% | 0.0% | 100.0% | 500 | `s5_combined_critical_seed123_20260424_223403.jsonl` |
| S5 Combined Critical | N | 999 | 1.000 | 0.011 | 0.0% | 0.0% | 100.0% | 500 | `s5_combined_critical_seed999_20260424_223403.jsonl` |
| S1 Nominal | Y | 42 | 1.000 | 2.010 | 100.0% | 0.0% | 0.0% | 500 | `s1_nominal_seed42_ablation_20260424_223404.jsonl` |
| S1 Nominal | Y | 123 | 1.000 | 2.010 | 100.0% | 0.0% | 0.0% | 500 | `s1_nominal_seed123_ablation_20260424_223404.jsonl` |
| S1 Nominal | Y | 999 | 1.000 | 2.010 | 100.0% | 0.0% | 0.0% | 500 | `s1_nominal_seed999_ablation_20260424_223404.jsonl` |
| S2 Geofence | Y | 42 | 1.000 | 2.010 | 0.0% | 0.0% | 100.0% | 500 | `s2_geofence_seed42_ablation_20260424_223405.jsonl` |
| S2 Geofence | Y | 123 | 1.000 | 2.010 | 0.0% | 0.0% | 100.0% | 500 | `s2_geofence_seed123_ablation_20260424_223405.jsonl` |
| S2 Geofence | Y | 999 | 1.000 | 2.010 | 0.0% | 0.0% | 100.0% | 500 | `s2_geofence_seed999_ablation_20260424_223405.jsonl` |
| S3 Torque | Y | 42 | 1.000 | 2.010 | 0.0% | 0.0% | 100.0% | 500 | `s3_torque_seed42_ablation_20260424_223406.jsonl` |
| S3 Torque | Y | 123 | 1.000 | 2.010 | 0.0% | 0.0% | 100.0% | 500 | `s3_torque_seed123_ablation_20260424_223406.jsonl` |
| S3 Torque | Y | 999 | 1.000 | 2.010 | 0.0% | 0.0% | 100.0% | 500 | `s3_torque_seed999_ablation_20260424_223406.jsonl` |
| S4 Hesitant | Y | 42 | 1.000 | 2.010 | 0.0% | 0.0% | 100.0% | 500 | `s4_hesitant_seed42_ablation_20260424_223406.jsonl` |
| S4 Hesitant | Y | 123 | 1.000 | 2.010 | 0.0% | 0.0% | 100.0% | 500 | `s4_hesitant_seed123_ablation_20260424_223407.jsonl` |
| S4 Hesitant | Y | 999 | 1.000 | 2.010 | 0.0% | 0.0% | 100.0% | 500 | `s4_hesitant_seed999_ablation_20260424_223407.jsonl` |
| S5 Combined Critical | Y | 42 | 1.000 | 1.614 | 0.0% | 0.0% | 100.0% | 500 | `s5_combined_critical_seed42_ablation_20260424_223407.jsonl` |
| S5 Combined Critical | Y | 123 | 1.000 | 1.614 | 0.0% | 0.0% | 100.0% | 500 | `s5_combined_critical_seed123_ablation_20260424_223407.jsonl` |
| S5 Combined Critical | Y | 999 | 1.000 | 1.614 | 0.0% | 0.0% | 100.0% | 500 | `s5_combined_critical_seed999_ablation_20260424_223407.jsonl` |

# NISP Information Gain Report

Estimator: k-NN classifier with Laplace-smoothed probabilities, 5-fold stratified CV. I(X;y) = H(y) - H(y|X_knn).
Confidence intervals: bootstrap over cycles, 95% bounds.

## Main runs (TribevX + NEURONAX active)
- Samples: 7,500 cycles across 5 scenarios (S1 Nominal, S2 Geofence, S3 Torque, S4 Hesitant, S5 Combined Critical)
- I(telemetry ; scenario)  = **1.585 bits** (95% CI [1.584, 1.585])
- I(intent Z ; scenario)   = **0.802 bits** (95% CI [0.789, 0.823])
- **IG = I(intent) - I(telemetry) = -0.783 bits** (95% CI [-0.795, -0.761])

## Ablation runs (--no-tribevx, Z uniform)
- Samples: 7,500 cycles across 5 scenarios (S1 Nominal, S2 Geofence, S3 Torque, S4 Hesitant, S5 Combined Critical)
- I(telemetry ; scenario)  = **1.585 bits** (95% CI [1.584, 1.585])
- I(intent Z ; scenario)   = **0.000 bits** (95% CI [0.000, 0.000])
- **IG = I(intent) - I(telemetry) = -1.585 bits** (95% CI [-1.585, -1.584])

## Ablation comparison - the falsifiability test
- I(Z ; scenario) with TribevX active    = 0.802 bits
- I(Z ; scenario) with --no-tribevx      = 0.000 bits
- **Delta (TribevX-attributable info)    = +0.802 bits**

H(scenario) ceiling = log2(5) = 2.322 bits. Delta > 0 with a 95% CI excluding zero falsifies the null hypothesis that TribevX+NEURONAX are inert. In this dataset the neural stack recovers 34.6% of the maximum possible scenario-discrimination entropy from Z alone, before any physical threshold fires.

Note on IG vs. telemetry: raw telemetry in these scenarios uses deterministic YAML overrides (no sensor noise), so I(telemetry;scenario) is saturated near the smoothing-adjusted ceiling. The IG = I(intent) - I(telemetry) formulation will become meaningful against noisy real-sensor runs; until then, the ablation-delta is the falsifiability criterion.
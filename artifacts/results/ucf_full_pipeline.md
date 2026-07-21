# UCF-Crime Full Pipeline Evaluation - NISP L1->L5

N=53 clips (26 RoadAccidents, 27 Normal). Real `facebook/tribev2` forward passes. Probe: SVM-RBF, 5-fold CV AUC 0.827 +/- 0.037.

## Table: Per-class pipeline metrics

| Class | N | rho (mean) | D (mean +/- sigma) | ALERT% | PROCEED% | Cert auditable |
|---|---|---|---|---|---|---|
| RoadAccident | 26 | 1.000 | 1.013 +/- 0.927 | 58% | 23% | 26/26 |
| Normal | 27 | 1.000 | 0.615 +/- 0.902 | 4% | 85% | 27/27 |
| **Combined** | **53** | **1.000** | - | - | - | **53/53** |

## Neural Dissonance (D) discrimination

- D(RoadAccident): mean=1.013, sigma=0.927, median=0.380
- D(Normal):       mean=0.615, sigma=0.902, median=0.057
- Mann-Whitney U=529, p=0.0008, rank-biserial r=0.754
- Lean 4 latency: mean=1.9 us, max=3.1 us (all < 8 ms)

## Calibration

- Expected Calibration Error (ECE, 10 bins): 0.1751
- Tier-2 threshold (0.60) and Tier-1 threshold (0.95) are justified by a moderately-calibrated probe (ECE=0.175).

## Merkle certificate audit

- 53/53 certificates independently verified by re-deriving SHA-256 Merkle root from committed evidence.

## Decoder architecture comparison (5-fold CV on N=53 UCF-Crime)

| Architecture | AUC (mean +/- sigma) | Recall | Precision |
|---|---|---|---|
| **NISP SVM-RBF (selected)** | **0.827 +/- 0.037** | 0.813 | 0.678 |
| GradBoost | 0.805 +/- 0.131 | 0.660 | 0.758 |
| MLP (64 hidden) | 0.767 +/- 0.058 | 0.713 | 0.830 |
| LogReg-L2 | 0.755 +/- 0.085 | 0.747 | 0.836 |
| LogReg-L1 | 0.679 +/- 0.123 | 0.668 | 0.677 |
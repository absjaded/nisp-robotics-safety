"""IG estimator tests — verify the kNN MI machinery on synthetic known-answer data."""
import numpy as np
import pytest

from src.nisp.learning.ig_estimator import compute_ig


def test_ig_perfect_signal_high_mi():
    """X perfectly determines y → MI should saturate near H(y) = log2(K)."""
    rng = np.random.default_rng(0)
    n_per_class = 200
    K = 5
    X_signal = np.concatenate([
        rng.normal(loc=k * 5.0, scale=0.1, size=(n_per_class, 2))  # well-separated
        for k in range(K)
    ])
    y = np.concatenate([np.full(n_per_class, k) for k in range(K)])
    X_noise = rng.standard_normal(X_signal.shape)

    scenarios = tuple(f"S{k}" for k in range(K))
    result = compute_ig(X_noise, X_signal, y, scenarios,
                       n_neighbors=5, n_bootstrap=5, seed=0)

    # Signal features should carry near-maximal MI.
    H_y = float(np.log2(K))
    assert result.i_intent > 0.5 * H_y, f"expected I(signal;y) >> noise, got {result.i_intent:.3f}"
    # Noise features should carry little MI.
    assert result.i_telem < 0.2 * H_y, f"expected I(noise;y) ~ 0, got {result.i_telem:.3f}"
    # IG positive.
    assert result.ig > 0
    # 95% CI on IG should not cross zero.
    assert result.ig_ci[0] > 0


def test_ig_no_signal_both_features():
    """Neither feature set carries information about y → IG ~ 0."""
    rng = np.random.default_rng(1)
    X1 = rng.standard_normal((500, 3))
    X2 = rng.standard_normal((500, 3))
    y = rng.integers(0, 4, size=500)

    result = compute_ig(X1, X2, y, ("A", "B", "C", "D"),
                        n_neighbors=5, n_bootstrap=5, seed=1)
    assert abs(result.ig) < 0.3, f"expected IG ~ 0 when neither side carries info, got {result.ig:.3f}"


def test_ig_result_shape():
    rng = np.random.default_rng(2)
    X1 = rng.standard_normal((100, 2))
    X2 = rng.standard_normal((100, 2))
    y = rng.integers(0, 3, size=100)
    r = compute_ig(X1, X2, y, ("a", "b", "c"), n_neighbors=3, n_bootstrap=5, seed=2)
    assert r.n_samples == 100
    assert r.n_scenarios == 3
    assert len(r.scenarios) == 3
    assert len(r.ig_ci) == 2
    # Bootstrap CI from small-sample MI estimates has known bias; just check it's a valid interval.
    assert r.ig_ci[0] <= r.ig_ci[1]

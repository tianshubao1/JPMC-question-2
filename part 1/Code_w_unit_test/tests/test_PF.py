import numpy as np
import pytest

from sv_ssm import simulate_sv
from PF import (
    run_pf,
    log_likelihood_y_given_x,
    effective_sample_size,
    systematic_resample,
)


def test_loglikelihood_finite():
    # simple sanity: loglik returns finite values
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    ll = log_likelihood_y_given_x(y_t=0.5, x_t=x, beta=0.65)
    assert ll.shape == x.shape
    assert np.all(np.isfinite(ll))


def test_ess_bounds():
    w = np.ones(10) / 10
    ess = effective_sample_size(w)
    assert ess == pytest.approx(10.0)

    w2 = np.zeros(10); w2[0] = 1.0
    ess2 = effective_sample_size(w2)
    assert ess2 == pytest.approx(1.0)


def test_systematic_resample_valid_indices():
    rng = np.random.default_rng(0)
    w = np.array([0.1, 0.2, 0.7])
    idx = systematic_resample(w, rng)
    assert idx.shape == (len(w),)
    assert np.all(idx >= 0) and np.all(idx < len(w))


def test_pf_output_shapes_and_normal_sanity():
    x, y = simulate_sv(T=200, alpha=0.98, sigma=0.15, beta=0.65, seed=0)
    y_obs = y.numpy().astype(np.float64)

    out = run_pf(y_obs, alpha=0.98, sigma=0.15, beta=0.65, N=2000, seed=0)

    T = len(y_obs)
    assert out["pf_mean"].shape == (T,)
    assert out["pf_var"].shape == (T,)
    assert out["ess_hist"].shape == (T,)
    assert out["resampled"].shape == (T,)

    # variances must be nonnegative
    assert np.all(out["pf_var"] >= 0)

    # ESS between 1 and N
    assert np.all(out["ess_hist"] >= 1.0)
    assert np.all(out["ess_hist"] <= out["N"] + 1e-6)


def test_pf_determinism_same_seed():
    x, y = simulate_sv(T=150, alpha=0.98, sigma=0.15, beta=0.65, seed=1)
    y_obs = y.numpy().astype(np.float64)

    out1 = run_pf(y_obs, alpha=0.98, sigma=0.15, beta=0.65, N=1000, seed=123)
    out2 = run_pf(y_obs, alpha=0.98, sigma=0.15, beta=0.65, N=1000, seed=123)

    np.testing.assert_allclose(out1["pf_mean"], out2["pf_mean"], atol=1e-12)
    np.testing.assert_allclose(out1["ess_hist"], out2["ess_hist"], atol=1e-12)


def test_pf_resampling_triggers_on_spike():
    # spike-y data should cause at least one resample
    x, y = simulate_sv(T=300, alpha=0.98, sigma=0.15, beta=0.65, seed=2)
    y_obs = y.numpy().astype(np.float64)

    out = run_pf(y_obs, alpha=0.98, sigma=0.15, beta=0.65,
                 N=1500, ess_frac=0.5, seed=2)

    assert np.any(out["resampled"])  # at least one resampling event


def test_weight_snapshots_exist():
    x, y = simulate_sv(T=200, alpha=0.98, sigma=0.15, beta=0.65, seed=3)
    y_obs = y.numpy().astype(np.float64)

    out = run_pf(y_obs, alpha=0.98, sigma=0.15, beta=0.65, N=1000, seed=3)

    assert out["w_calm"] is not None
    assert out["w_spike"] is not None

    # snapshots should be valid prob distributions
    assert np.isclose(np.sum(out["w_calm"]), 1.0)
    assert np.isclose(np.sum(out["w_spike"]), 1.0)
    assert np.all(out["w_calm"] >= 0)
    assert np.all(out["w_spike"] >= 0)


def test_spike_weights_more_degenerate_than_calm():
    x, y = simulate_sv(T=300, alpha=0.98, sigma=0.15, beta=0.65, seed=4)
    y_obs = y.numpy().astype(np.float64)

    out = run_pf(y_obs, alpha=0.98, sigma=0.15, beta=0.65, N=2000, seed=4)

    calm = out["w_calm"]
    spike = out["w_spike"]

    ess_calm = effective_sample_size(calm)
    ess_spike = effective_sample_size(spike)

    # by definition of calm/spike (min/max y^2),
    # spike ESS should be smaller (more degeneracy)
    assert ess_spike <= ess_calm

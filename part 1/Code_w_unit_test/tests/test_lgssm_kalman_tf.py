import numpy as np
import tensorflow as tf
import pytest
from lgssm_kalman_tf import (
    make_tracking_lgssm,
    simulate_lgssm,
    kalman_filter,
    cov_diagnostics,
    compute_condition_numbers,
)

@pytest.fixture(scope="module")
def params():
    tf.keras.backend.set_floatx("float64")
    return make_tracking_lgssm(dt=1.0, q=0.1, r=0.5)

def test_make_tracking_lgssm_shapes(params):
    assert params.A.shape == (4, 4)
    assert params.C.shape == (2, 4)
    assert params.Q.shape == (4, 4)
    assert params.R.shape == (2, 2)
    assert params.m0.shape == (4,)
    assert params.P0.shape == (4, 4)

def test_simulate_shapes(params):
    T = 50
    x, y = simulate_lgssm(params, T=T, seed=0)
    assert x.shape == (T, 4)
    assert y.shape == (T, 2)

def test_simulate_reproducible_seed(params):
    x1, y1 = simulate_lgssm(params, T=30, seed=123)
    x2, y2 = simulate_lgssm(params, T=30, seed=123)

    np.testing.assert_allclose(x1.numpy(), x2.numpy(), atol=1e-12)
    np.testing.assert_allclose(y1.numpy(), y2.numpy(), atol=1e-12)

def test_kalman_filter_shapes(params):
    x, y = simulate_lgssm(params, T=40, seed=0)
    m, P, S = kalman_filter(y, params, use_joseph=True)

    assert m.shape == (40, 4)
    assert P.shape == (40, 4, 4)
    assert S.shape == (40, 2, 2)

def test_cov_psd_and_symmetric_joseph(params):
    _, y = simulate_lgssm(params, T=80, seed=1)
    _, Pj, _ = kalman_filter(y, params, use_joseph=True)

    min_eigs, sym_err = cov_diagnostics(Pj)

    # symmetry (numerical)
    assert np.max(sym_err) < 1e-8

    # PSD preserved (allow microscopic negative from float noise)
    assert np.min(min_eigs) > -1e-10

def test_joseph_and_standard_means_close(params):
    _, y = simulate_lgssm(params, T=100, seed=2)
    mj, _, _ = kalman_filter(y, params, use_joseph=True)
    mstd, _, _ = kalman_filter(y, params, use_joseph=False)

    # for this well-conditioned model, means should match tightly
    np.testing.assert_allclose(mj.numpy(), mstd.numpy(), atol=1e-6)

def test_filtering_beats_raw_observations_on_position(params):
    x, y = simulate_lgssm(params, T=120, seed=0)
    m, _, _ = kalman_filter(y, params, use_joseph=True)

    pos_true = x.numpy()[:, [0, 2]]
    pos_obs  = y.numpy()
    pos_filt = m.numpy()[:, [0, 2]]

    obs_rmse  = np.sqrt(np.mean((pos_obs - pos_true) ** 2))
    filt_rmse = np.sqrt(np.mean((pos_filt - pos_true) ** 2))

    assert filt_rmse < obs_rmse

def test_condition_numbers_finite(params):
    _, y = simulate_lgssm(params, T=60, seed=3)
    _, P, S = kalman_filter(y, params, use_joseph=True)

    condP, condS = compute_condition_numbers(P, S)

    assert np.all(np.isfinite(condP))
    assert np.all(np.isfinite(condS))
    assert np.all(condP >= 1.0)
    assert np.all(condS >= 1.0)

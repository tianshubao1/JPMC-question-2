import numpy as np
import tensorflow as tf
import pytest

from sv_ssm import simulate_sv


def test_sv_shapes_and_finite():
    x, y = simulate_sv(T=100, alpha=0.98, sigma=0.15, beta=0.65, seed=0)
    assert x.shape == (100,)
    assert y.shape == (100,)
    assert np.all(np.isfinite(x.numpy()))
    assert np.all(np.isfinite(y.numpy()))


def test_sv_reproducibility_with_seed():
    x1, y1 = simulate_sv(T=80, alpha=0.98, sigma=0.15, beta=0.65, seed=123)
    x2, y2 = simulate_sv(T=80, alpha=0.98, sigma=0.15, beta=0.65, seed=123)

    np.testing.assert_allclose(x1.numpy(), x2.numpy(), atol=1e-7)
    np.testing.assert_allclose(y1.numpy(), y2.numpy(), atol=1e-7)


def test_sv_mean_std_reasonable_stationary_prior():
    """
    x_t is AR(1) with stationary variance sigma^2/(1-alpha^2).
    For long T, sample variance should be in the right ballpark.
    """
    T = 2000
    alpha = 0.98
    sigma = 0.15
    beta = 0.65

    x, y = simulate_sv(T=T, alpha=alpha, sigma=sigma, beta=beta, seed=0)
    x_np = x.numpy()

    var_stationary = (sigma**2) / (1 - alpha**2)
    sample_var = x_np.var()

    # loose tolerance because of finite sample + randomness
    assert sample_var > 0.3 * var_stationary
    assert sample_var < 3.0 * var_stationary


def test_sv_observation_variance_increases_with_state():
    """
    Conditional on x_t, Var(y_t | x_t) = beta^2 * exp(x_t).
    So y^2 should be positively correlated with exp(x).
    """
    T = 1500
    alpha = 0.98
    sigma = 0.15
    beta = 0.65

    x, y = simulate_sv(T=T, alpha=alpha, sigma=sigma, beta=beta, seed=1)
    x_np = x.numpy()
    y_np = y.numpy()

    corr = np.corrcoef(np.exp(x_np), y_np**2)[0, 1]

    assert corr > 0.2

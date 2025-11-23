import numpy as np
import pytest

from sv_ssm import simulate_sv
from EKF_UKF import (
    make_z_observation,
    stationary_prior,
    run_ekf_z,
    run_ukf_z,
    run_naive_ukf_y,
)


def test_make_z_observation_shapes():
    y = np.random.randn(100).astype(np.float32)
    z, c, Rz = make_z_observation(y, beta=0.65, eps=1e-6)
    assert z.shape == (100,)
    assert np.isscalar(c)
    assert np.isscalar(Rz)
    assert Rz > 0


def test_stationary_prior_positive():
    m0, P0 = stationary_prior(alpha=0.98, sigma=0.15)
    assert np.isfinite(m0)
    assert P0 > 0


def test_ekf_ukf_z_equivalence():
    x, y = simulate_sv(T=300, alpha=0.98, sigma=0.15, beta=0.65, seed=0)
    y_obs = y.numpy().astype(np.float32)

    z_obs, c, Rz = make_z_observation(y_obs, beta=0.65, eps=1e-6)
    m0, P0 = stationary_prior(alpha=0.98, sigma=0.15)

    m_ekf, P_ekf = run_ekf_z(z_obs, alpha=0.98, sigma=0.15, c=c, Rz=Rz, m0=m0, P0=P0)
    m_ukf, P_ukf = run_ukf_z(z_obs, alpha=0.98, sigma=0.15, c=c, Rz=Rz, m0=m0, P0=P0)

    max_diff_m = np.max(np.abs(m_ekf - m_ukf))
    max_diff_P = np.max(np.abs(P_ekf - P_ukf))

    # sigma-point numerical drift in float32 is expected, but should be tiny
    assert max_diff_m < 1e-2
    assert max_diff_P < 1e-2



def test_naive_ukf_gain_collapses():
    x, y = simulate_sv(T=200, alpha=0.98, sigma=0.15, beta=0.65, seed=1)
    y_obs = y.numpy().astype(np.float32)

    m0, P0 = stationary_prior(alpha=0.98, sigma=0.15)
    Ry = np.var(y_obs)

    _, _, K_hist = run_naive_ukf_y(y_obs, alpha=0.98, sigma=0.15, m0=m0, P0=P0, Ry=Ry)

    # naive UKF should have very small gain almost everywhere
    assert np.median(np.abs(K_hist)) < 1e-3


def test_ekf_beats_naive_ukf_rmse():
    x, y = simulate_sv(T=400, alpha=0.98, sigma=0.15, beta=0.65, seed=2)
    x_true = x.numpy()
    y_obs  = y.numpy().astype(np.float32)

    z_obs, c, Rz = make_z_observation(y_obs, beta=0.65)
    m0, P0 = stationary_prior(alpha=0.98, sigma=0.15)

    m_ekf, _ = run_ekf_z(z_obs, alpha=0.98, sigma=0.15, c=c, Rz=Rz, m0=m0, P0=P0)
    m_nu, _, _ = run_naive_ukf_y(y_obs, alpha=0.98, sigma=0.15, m0=m0, P0=P0, Ry=np.var(y_obs))

    rmse_ekf = np.sqrt(np.mean((m_ekf - x_true)**2))
    rmse_nu  = np.sqrt(np.mean((m_nu  - x_true)**2))

    assert rmse_ekf < rmse_nu

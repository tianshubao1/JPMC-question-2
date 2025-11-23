import numpy as np
import pytest

from sv_ssm import simulate_sv
from EKF_UKF_PF_comparison import (
    run_ekf_z, run_ukf_z, run_naive_ukf_y, run_pf,
    rmse, mae, coverage, run_all_benchmarks
)


def _make_short_sv(seed=0, T=200):
    x, y = simulate_sv(T=T, alpha=0.98, sigma=0.15, beta=0.65, seed=seed)
    return x.numpy().astype(np.float64), y.numpy().astype(np.float64)


def test_filters_return_shapes():
    x_true, y_obs = _make_short_sv(seed=0, T=150)

    ekf_m, ekf_P = run_ekf_z(y_obs, 0.98, 0.15, 0.65)
    ukf_m, ukf_P = run_ukf_z(y_obs, 0.98, 0.15, 0.65)
    nu_m, nu_P   = run_naive_ukf_y(y_obs, 0.98, 0.15, 0.65)
    pf_m, pf_P, ess = run_pf(y_obs, 0.98, 0.15, 0.65, N=1000, seed=0)

    T = len(y_obs)
    for mhat, Phat in [(ekf_m, ekf_P), (ukf_m, ukf_P), (nu_m, nu_P), (pf_m, pf_P)]:
        assert mhat.shape == (T,)
        assert Phat.shape == (T,)
        assert np.all(Phat >= 0.0)

    assert ess.shape == (T,)
    assert np.all(ess >= 1.0) and np.all(ess <= 1000 + 1e-6)


def test_ekf_ukf_close_on_z():
    x_true, y_obs = _make_short_sv(seed=1, T=250)

    ekf_m, ekf_P = run_ekf_z(y_obs, 0.98, 0.15, 0.65)
    ukf_m, ukf_P = run_ukf_z(y_obs, 0.98, 0.15, 0.65)

    # z-model is linear => EKF and UKF should be very close
    np.testing.assert_allclose(ekf_m, ukf_m, atol=1e-2)
    np.testing.assert_allclose(ekf_P, ukf_P, atol=1e-2)


def test_naive_ukf_is_worse_than_ekf():
    x_true, y_obs = _make_short_sv(seed=2, T=300)

    ekf_m, ekf_P = run_ekf_z(y_obs, 0.98, 0.15, 0.65)
    nu_m, nu_P   = run_naive_ukf_y(y_obs, 0.98, 0.15, 0.65)

    # naive UKF on raw y should be worse
    assert rmse(nu_m, x_true) > rmse(ekf_m, x_true)
    assert mae(nu_m, x_true)  > mae(ekf_m, x_true)

    # coverage should collapse for naive UKF
    assert coverage(nu_m, nu_P, x_true) < 0.5


def test_pf_reasonable_coverage_and_resampling():
    x_true, y_obs = _make_short_sv(seed=3, T=300)

    pf_m, pf_P, ess = run_pf(y_obs, 0.98, 0.15, 0.65, N=1500, seed=3)

    cov_pf = coverage(pf_m, pf_P, x_true)
    assert cov_pf > 0.8  # PF credible band should cover most truth

    # ESS should dip sometimes enough to cause resampling in SV data
    assert np.min(ess) < 0.9 * 1500


def test_run_all_benchmarks_returns_rows():
    x_true, y_obs = _make_short_sv(seed=4, T=200)

    rows, ess = run_all_benchmarks(x_true, y_obs, 0.98, 0.15, 0.65, seed=4, N_pf=1000)

    assert len(rows) == 4
    names = [r[0] for r in rows]
    assert "EKF(z)" in names
    assert "UKF(z)" in names
    assert "Naive UKF(y)" in names
    assert any("PF" in n for n in names)

    for name, mhat, Phat, rt, mb in rows:
        assert rt > 0.0
        assert mb >= 0.0
        assert mhat.shape == (len(y_obs),)
        assert Phat.shape == (len(y_obs),)

    assert ess.shape == (len(y_obs),)

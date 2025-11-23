# test_prob_2_c_sv.py
import numpy as np
import pytest

from EDH_LEDH_kernel_PFF_comparison_on_SSM import (
    SVParams,
    simulate_sv_np,
    sv_loglik,
    sv_score_and_hessian,
    edh_flow_update,
    ledh_flow_update,
    kernel_pff_update,
    run_sv_pf,
)



# ---------------------------
# 1) SV simulation tests
# ---------------------------

def test_simulate_sv_np_shapes_and_reproducible():
    params = SVParams(T=80, alpha=0.98, sigma=0.15, beta=0.65, seed=123)
    x1, y1 = simulate_sv_np(params)
    x2, y2 = simulate_sv_np(params)

    assert x1.shape == (params.T,)
    assert y1.shape == (params.T,)
    np.testing.assert_allclose(x1, x2, atol=1e-12)
    np.testing.assert_allclose(y1, y2, atol=1e-12)


# ---------------------------
# 2) SV loglik + score/hess sanity
# ---------------------------

def test_sv_loglik_vectorized_and_finite():
    params = SVParams(T=10, beta=0.65, seed=0)
    rng = np.random.default_rng(0)

    X = rng.normal(size=50)
    y = float(rng.normal())

    logw = sv_loglik(X, y, params.beta)
    assert logw.shape == (50,)
    assert np.all(np.isfinite(logw))


def test_sv_score_and_hessian_shapes_and_finite():
    params = SVParams(T=10, beta=0.65, seed=0)
    rng = np.random.default_rng(1)

    X = rng.normal(size=60)
    y = float(rng.normal())

    score, hess = sv_score_and_hessian(X, y, params.beta)
    assert score.shape == (60,)
    assert hess.shape == (60,)
    assert np.all(np.isfinite(score))
    assert np.all(np.isfinite(hess))
    # Hessian should be <= 0 for SV likelihood
    assert np.all(hess <= 1e-12)


# ---------------------------
# 3) Flow update unit tests
# ---------------------------

@pytest.mark.parametrize("flow_fn", [edh_flow_update, ledh_flow_update])
def test_edh_ledh_flow_update_shapes_and_finite(flow_fn):
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=120)
    y = 0.5
    beta = 0.65

    X_new, diag = flow_fn(X0, y, beta, flow_steps=5, flow_dt=0.01, max_flow=2.0)

    assert X_new.shape == X0.shape
    assert np.all(np.isfinite(X_new))
    assert "flow_mag" in diag and "jac_mag" in diag
    assert diag["flow_mag"].shape == (5,)
    assert diag["jac_mag"].shape == (5,)
    assert np.all(np.isfinite(diag["flow_mag"]))
    assert np.all(np.isfinite(diag["jac_mag"]))


def test_kernel_flow_update_shapes_and_finite():
    rng = np.random.default_rng(2)
    X0 = rng.normal(size=100)
    y = -1.0
    beta = 0.65

    X_new, diag = kernel_pff_update(
        X0, y, beta, alpha_k=1.0, flow_steps=6, flow_dt=0.01, max_flow=2.0
    )

    assert X_new.shape == X0.shape
    assert np.all(np.isfinite(X_new))
    assert diag["flow_mag"].shape == (6,)
    assert diag["jac_mag"].shape == (6,)
    assert np.all(np.isfinite(diag["flow_mag"]))
    assert np.all(np.isfinite(diag["jac_mag"]))


# ---------------------------
# 4) End-to-end PF tests
# ---------------------------

@pytest.mark.parametrize("method", ["baseline", "edh", "ledh", "kernel"])
def test_run_sv_pf_shapes_and_finite(method):
    params = SVParams(T=70, alpha=0.98, sigma=0.15, beta=0.65, seed=0)

    out = run_sv_pf(
        params=params,
        N=200,
        method=method,
        obs_gap=1,
        flow_steps=4,   # small for speed
        flow_dt=0.01,
        alpha_k=1.0,
        max_flow=2.0,
        seed=0
    )

    rmse = out["rmse"]
    ess  = out["ess"]

    assert rmse.shape == (params.T,)
    assert ess.shape == (params.T,)
    assert np.all(np.isfinite(rmse))
    assert np.all(np.isfinite(ess))
    assert np.all(rmse >= 0)
    assert np.all(ess > 0)
    assert np.all(ess <= 200 + 1e-6)


def test_run_sv_pf_sparse_obs_still_finite():
    params = SVParams(T=60, alpha=0.98, sigma=0.15, beta=0.65, seed=1)

    out = run_sv_pf(
        params=params,
        N=180,
        method="kernel",
        obs_gap=5,
        flow_steps=4,
        flow_dt=0.01,
        alpha_k=1.0,
        max_flow=2.0,
        seed=1
    )

    assert np.all(np.isfinite(out["rmse"]))
    assert np.all(np.isfinite(out["ess"]))
    assert out["flow_mag"].ndim == 2  # (n_assim, flow_steps)
    assert out["jac_mag"].ndim == 2

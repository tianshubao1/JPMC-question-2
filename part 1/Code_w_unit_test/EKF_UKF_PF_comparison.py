#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonlinear/Non-Gaussian SSM with EKF/UKF and Particle Filter
Problem d
Comparison EKF(z), UKF(z), naive UKF(y), PF on SV model
Measures:
  RMSE, MAE, AvgVar, Cov@2σ, Runtime(s), PeakCPU(MB)
"""

import time
import tracemalloc
import numpy as np


# -----------------------
# Metrics / utilities
# -----------------------
def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean(np.abs(a - b)))

def coverage(m, P, truth, k=2.0):
    m = np.asarray(m); P = np.asarray(P); truth = np.asarray(truth)
    std = np.sqrt(P)
    lo, hi = m - k * std, m + k * std
    return float(np.mean((truth >= lo) & (truth <= hi)))

def effective_sample_size(w):
    w = np.asarray(w)
    return 1.0 / np.sum(w ** 2)

def systematic_resample(w, rng):
    w = np.asarray(w)
    Nloc = len(w)
    positions = (rng.random() + np.arange(Nloc)) / Nloc
    cumsum = np.cumsum(w)
    idx = np.zeros(Nloc, dtype=np.int64)
    i = j = 0
    while i < Nloc:
        if positions[i] < cumsum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx


# -----------------------
# SV constants for z_t approximation
# -----------------------
def make_z_constants(beta):
    eps = 1e-6
    mu_eta = -1.27036
    var_eta = 4.93480
    c_z = np.log(beta ** 2) + mu_eta
    Rz = var_eta
    return eps, c_z, Rz


def stationary_prior(alpha, sigma):
    P0 = sigma ** 2 / (1.0 - alpha ** 2)
    m0 = 0.0
    return m0, P0


# -----------------------
# Filters (pure functions)
# -----------------------
def run_ekf_z(y_obs, alpha, sigma, beta):
    """
    EKF on z_t = log(y^2 + eps) with Gaussian log-chi2 approx.
    This is scalar KF (but we keep EKF name to match assignment).
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    T = len(y_obs)

    eps, c_z, Rz = make_z_constants(beta)
    m0, P0 = stationary_prior(alpha, sigma)

    m = np.zeros(T)
    P = np.zeros(T)

    m_prev, P_prev = m0, P0
    for t in range(T):
        # predict
        m_pred = alpha * m_prev
        P_pred = alpha ** 2 * P_prev + sigma ** 2

        # pseudo measurement
        z = np.log(y_obs[t] ** 2 + eps)

        # linear measurement z = x + c + e
        S = P_pred + Rz
        K = P_pred / S

        m_post = m_pred + K * (z - (m_pred + c_z))
        # standard scalar covariance update
        P_post = (1.0 - K) * P_pred

        m[t], P[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post

    return m, P


def run_ukf_z(y_obs, alpha, sigma, beta,
             alpha_u=1e-3, beta_u=2.0, kappa=0.0):
    """
    UKF on transformed observation z_t.
    For z_t, measurement map is linear, so UKF should match EKF closely.
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    T = len(y_obs)

    eps, c_z, Rz = make_z_constants(beta)
    m0, P0 = stationary_prior(alpha, sigma)

    m = np.zeros(T)
    P = np.zeros(T)

    lam = alpha_u ** 2 * (1 + kappa) - 1
    Wm0 = lam / (1 + lam)
    Wc0 = Wm0 + (1 - alpha_u ** 2 + beta_u)
    Wi = 1 / (2 * (1 + lam))

    m_prev, P_prev = m0, P0
    for t in range(T):
        # predict
        m_pred = alpha * m_prev
        P_pred = alpha ** 2 * P_prev + sigma ** 2

        # sigma points
        sqrtP = np.sqrt((1 + lam) * P_pred)
        X = np.array([m_pred, m_pred + sqrtP, m_pred - sqrtP])

        # measurement on z: Z = X + c_z
        Z = X + c_z

        z_pred = Wm0 * Z[0] + Wi * (Z[1] + Z[2])
        Pzz = (
            Wc0 * (Z[0] - z_pred) ** 2
            + Wi * ((Z[1] - z_pred) ** 2 + (Z[2] - z_pred) ** 2)
            + Rz
        )
        Pxz = (
            Wc0 * (X[0] - m_pred) * (Z[0] - z_pred)
            + Wi * (
                (X[1] - m_pred) * (Z[1] - z_pred)
                + (X[2] - m_pred) * (Z[2] - z_pred)
            )
        )

        K = Pxz / Pzz
        z = np.log(y_obs[t] ** 2 + eps)

        m_post = m_pred + K * (z - z_pred)
        P_post = P_pred - K * Pzz * K

        m[t], P[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post

    return m, P


def run_naive_ukf_y(y_obs, alpha, sigma, beta,
                    alpha_u=1e-3, beta_u=2.0, kappa=0.0):
    """
    Naive UKF applied directly to raw y_t under WRONG additive model.
    Expected to fail (big RMSE, low coverage).
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    T = len(y_obs)

    m0, P0 = stationary_prior(alpha, sigma)

    m = np.zeros(T)
    P = np.zeros(T)

    lam = alpha_u ** 2 * (1 + kappa) - 1
    Wm0 = lam / (1 + lam)
    Wc0 = Wm0 + (1 - alpha_u ** 2 + beta_u)
    Wi = 1 / (2 * (1 + lam))

    R_naive = beta ** 2  # WRONG constant variance

    m_prev, P_prev = m0, P0
    for t in range(T):
        m_pred = alpha * m_prev
        P_pred = alpha ** 2 * P_prev + sigma ** 2

        sqrtP = np.sqrt((1 + lam) * P_pred)
        X = np.array([m_pred, m_pred + sqrtP, m_pred - sqrtP])

        # WRONG mean mapping for y
        Y = beta * np.exp(X / 2.0)

        y_pred = Wm0 * Y[0] + Wi * (Y[1] + Y[2])
        Pyy = (
            Wc0 * (Y[0] - y_pred) ** 2
            + Wi * ((Y[1] - y_pred) ** 2 + (Y[2] - y_pred) ** 2)
            + R_naive
        )
        Pxy = (
            Wc0 * (X[0] - m_pred) * (Y[0] - y_pred)
            + Wi * (
                (X[1] - m_pred) * (Y[1] - y_pred)
                + (X[2] - m_pred) * (Y[2] - y_pred)
            )
        )

        K = Pxy / Pyy
        m_post = m_pred + K * (y_obs[t] - y_pred)
        P_post = P_pred - K * Pyy * K

        m[t], P[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post

    return m, P


def run_pf(y_obs, alpha, sigma, beta, N=5000, ess_frac=0.5, seed=0):
    """
    Bootstrap PF on SV likelihood.
    Returns (m, P, ess_hist).
    Deterministic via local rng.
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    T = len(y_obs)
    rng = np.random.default_rng(seed)

    ess_threshold = ess_frac * N
    _, P0 = stationary_prior(alpha, sigma)

    m = np.zeros(T)
    P = np.zeros(T)
    ess_hist = np.zeros(T)

    particles = np.sqrt(P0) * rng.standard_normal(N)
    weights = np.ones(N, dtype=np.float64) / N

    for t in range(T):
        particles = alpha * particles + sigma * rng.standard_normal(N)

        var = beta ** 2 * np.exp(particles)
        logw = -0.5 * (np.log(2 * np.pi * var) + (y_obs[t] ** 2) / var)
        logw -= np.max(logw)
        w = np.exp(logw)
        w /= np.sum(w)
        weights = w

        m[t] = np.sum(weights * particles)
        P[t] = np.sum(weights * (particles - m[t]) ** 2)

        ess = effective_sample_size(weights)
        ess_hist[t] = ess

        if ess < ess_threshold:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights.fill(1.0 / N)

    return m, P, ess_hist


# -----------------------
# Benchmark wrapper
# -----------------------
def bench(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    rt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 ** 2)
    return out, rt, peak_mb


def run_all_benchmarks(x_true, y_obs, alpha, sigma, beta, seed=0, N_pf=5000):
    """
    Runs & benchmarks all filters on the same data.
    Returns list of rows:
      (name, mhat, Phat, runtime_sec, peak_mb)
    """
    (ekf_m, ekf_P), rt_ekf, mb_ekf = bench(lambda: run_ekf_z(y_obs, alpha, sigma, beta))
    (ukf_m, ukf_P), rt_ukf, mb_ukf = bench(lambda: run_ukf_z(y_obs, alpha, sigma, beta))
    (nu_m, nu_P),   rt_nu,  mb_nu  = bench(lambda: run_naive_ukf_y(y_obs, alpha, sigma, beta))
    (pf_m, pf_P, ess_hist), rt_pf, mb_pf = bench(lambda: run_pf(y_obs, alpha, sigma, beta, N=N_pf, seed=seed))

    rows = [
        ("EKF(z)", ekf_m, ekf_P, rt_ekf, mb_ekf),
        ("UKF(z)", ukf_m, ukf_P, rt_ukf, mb_ukf),
        ("Naive UKF(y)", nu_m, nu_P, rt_nu, mb_nu),
        (f"PF (N={N_pf})", pf_m, pf_P, rt_pf, mb_pf),
    ]
    return rows, ess_hist


def main():
    data = np.load("data_sv.npz")
    x_true = data["x_true"].astype(np.float64)
    y_obs  = data["y_obs"].astype(np.float64)
    alpha  = float(data["alpha"])
    sigma  = float(data["sigma"])
    beta   = float(data["beta"])
    seed   = int(data["seed"])

    rows, _ = run_all_benchmarks(x_true, y_obs, alpha, sigma, beta, seed=seed, N_pf=5000)

    print(f"{'Method':<14} {'RMSE':>8} {'MAE':>8} {'AvgVar':>10} "
          f"{'Cov@2σ':>9} {'Runtime(s)':>12} {'PeakCPU(MB)':>12}")
    print("-" * 85)
    for name, mhat, Phat, rt, mb in rows:
        print(f"{name:<14} {rmse(mhat, x_true):8.4f} {mae(mhat, x_true):8.4f} "
              f"{float(np.mean(Phat)):10.4f} {coverage(mhat, Phat, x_true):9.3f} "
              f"{rt:12.4f} {mb:12.2f}")


if __name__ == "__main__":
    main()

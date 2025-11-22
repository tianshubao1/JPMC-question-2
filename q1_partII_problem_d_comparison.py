#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2 Problem 4 (rerun) — Fair comparison EKF(z), UKF(z), naive UKF(y), PF on SV model
Runnable script (no argparse).

Measures:
  RMSE, MAE, AvgVar, Cov@2σ, Runtime(s), PeakCPU(MB)

This script *re-executes* all filters on the same data for fair benchmarking.
"""

import time
import tracemalloc
import numpy as np

# -----------------------
# Utilities
# -----------------------
def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
def mae(a, b):  return float(np.mean(np.abs(a-b)))
def coverage(m, P, truth, k=2.0):
    std = np.sqrt(P)
    lo, hi = m - k*std, m + k*std
    return float(np.mean((truth >= lo) & (truth <= hi)))

def systematic_resample(w):
    Nloc = len(w)
    positions = (np.random.rand() + np.arange(Nloc)) / Nloc
    cumsum = np.cumsum(w)
    idx = np.zeros(Nloc, dtype=np.int64)
    i, j = 0, 0
    while i < Nloc:
        if positions[i] < cumsum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx

def effective_sample_size(w):
    return 1.0 / np.sum(w**2)

# -----------------------
# Load SV data
# -----------------------
data = np.load("data_sv.npz")
x_true = data["x_true"].astype(np.float64)
y_obs  = data["y_obs"].astype(np.float64)
T      = int(data["T"])
alpha  = float(data["alpha"])
sigma  = float(data["sigma"])
beta   = float(data["beta"])
seed   = int(data["seed"])
np.random.seed(seed)

# SV constants for z_t = log(y^2 + eps) approx
eps = 1e-6
mu_eta = -1.27036
var_eta = 4.93480
c_z = np.log(beta**2) + mu_eta
Rz = var_eta

# -----------------------
# EKF(z) == KF on pseudo-measurement
# -----------------------
def run_ekf_z():
    m = np.zeros(T)
    P = np.zeros(T)
    m_prev = 0.0
    P_prev = sigma**2 / (1 - alpha**2)

    for t in range(T):
        # predict
        m_pred = alpha * m_prev
        P_pred = alpha**2 * P_prev + sigma**2

        # pseudo measurement
        z = np.log(y_obs[t]**2 + eps)

        # linear measurement: z = x + c + e
        H = 1.0
        S = H*P_pred*H + Rz
        K = P_pred*H / S

        m_post = m_pred + K * (z - (m_pred + c_z))
        P_post = (1 - K*H) * P_pred

        m[t], P[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post
    return m, P

# -----------------------
# UKF(z)
# -----------------------
def run_ukf_z():
    m = np.zeros(T)
    P = np.zeros(T)

    m_prev = 0.0
    P_prev = sigma**2 / (1 - alpha**2)

    # UT params (scalar)
    kappa = 0.0
    alpha_u = 1e-3
    beta_u = 2.0
    lam = alpha_u**2 * (1 + kappa) - 1
    Wm0 = lam / (1 + lam)
    Wc0 = Wm0 + (1 - alpha_u**2 + beta_u)
    Wi  = 1 / (2*(1+lam))

    for t in range(T):
        # predict (linear Gaussian)
        m_pred = alpha * m_prev
        P_pred = alpha**2 * P_prev + sigma**2

        # sigma points for x
        sqrtP = np.sqrt((1+lam)*P_pred)
        X = np.array([m_pred, m_pred + sqrtP, m_pred - sqrtP])

        # measurement mapping on z_t (still linear)
        Z = X + c_z

        z_pred = Wm0*Z[0] + Wi*(Z[1] + Z[2])
        Pzz = Wc0*(Z[0]-z_pred)**2 + Wi*((Z[1]-z_pred)**2 + (Z[2]-z_pred)**2) + Rz
        Pxz = Wc0*(X[0]-m_pred)*(Z[0]-z_pred) + Wi*((X[1]-m_pred)*(Z[1]-z_pred) + (X[2]-m_pred)*(Z[2]-z_pred))

        K = Pxz / Pzz
        z = np.log(y_obs[t]**2 + eps)

        m_post = m_pred + K*(z - z_pred)
        P_post = P_pred - K*Pzz*K

        m[t], P[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post
    return m, P

# -----------------------
# Naive UKF(y) (wrong additive model)
# -----------------------
def run_naive_ukf_y():
    m = np.zeros(T)
    P = np.zeros(T)

    m_prev = 0.0
    P_prev = sigma**2 / (1 - alpha**2)

    kappa = 0.0
    alpha_u = 1e-3
    beta_u = 2.0
    lam = alpha_u**2 * (1 + kappa) - 1
    Wm0 = lam / (1 + lam)
    Wc0 = Wm0 + (1 - alpha_u**2 + beta_u)
    Wi  = 1 / (2*(1+lam))

    R_naive = beta**2  # WRONG constant variance

    for t in range(T):
        m_pred = alpha * m_prev
        P_pred = alpha**2 * P_prev + sigma**2

        sqrtP = np.sqrt((1+lam)*P_pred)
        X = np.array([m_pred, m_pred + sqrtP, m_pred - sqrtP])

        # WRONG mean mapping for y: assumes y = beta*exp(x/2) + noise
        Y = beta*np.exp(X/2.0)

        y_pred = Wm0*Y[0] + Wi*(Y[1] + Y[2])
        Pyy = Wc0*(Y[0]-y_pred)**2 + Wi*((Y[1]-y_pred)**2 + (Y[2]-y_pred)**2) + R_naive
        Pxy = Wc0*(X[0]-m_pred)*(Y[0]-y_pred) + Wi*((X[1]-m_pred)*(Y[1]-y_pred) + (X[2]-m_pred)*(Y[2]-y_pred))

        K = Pxy / Pyy
        m_post = m_pred + K*(y_obs[t] - y_pred)
        P_post = P_pred - K*Pyy*K

        m[t], P[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post
    return m, P

# -----------------------
# PF bootstrap
# -----------------------
def run_pf(N=5000, ess_frac=0.5):
    ess_threshold = ess_frac*N
    m = np.zeros(T)
    P = np.zeros(T)
    ess_hist = np.zeros(T)

    particles = np.sqrt(sigma**2/(1-alpha**2))*np.random.randn(N)
    weights = np.ones(N)/N

    for t in range(T):
        particles = alpha*particles + sigma*np.random.randn(N)

        var = beta**2*np.exp(particles)
        logw = -0.5*(np.log(2*np.pi*var) + (y_obs[t]**2)/var)
        logw -= np.max(logw)
        w = np.exp(logw)
        w /= np.sum(w)
        weights = w

        m[t] = np.sum(weights*particles)
        P[t] = np.sum(weights*(particles-m[t])**2)

        ess = effective_sample_size(weights)
        ess_hist[t] = ess
        if ess < ess_threshold:
            idx = systematic_resample(weights)
            particles = particles[idx]
            weights.fill(1.0/N)

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
    peak_mb = peak/(1024**2)
    return out, rt, peak_mb

# run + bench
(ekf_m, ekf_P), rt_ekf, mb_ekf = bench(run_ekf_z)
(ukf_m, ukf_P), rt_ukf, mb_ukf = bench(run_ukf_z)
(nu_m, nu_P),   rt_nu,  mb_nu  = bench(run_naive_ukf_y)
(pf_m, pf_P, ess_hist), rt_pf, mb_pf = bench(lambda: run_pf(N=5000))

rows = [
    ("EKF(z)", ekf_m, ekf_P, rt_ekf, mb_ekf),
    ("UKF(z)", ukf_m, ukf_P, rt_ukf, mb_ukf),
    ("Naive UKF(y)", nu_m, nu_P, rt_nu, mb_nu),
    ("PF (N=5000)", pf_m, pf_P, rt_pf, mb_pf)
]

print(f"{'Method':<14} {'RMSE':>8} {'MAE':>8} {'AvgVar':>10} {'Cov@2σ':>9} {'Runtime(s)':>12} {'PeakCPU(MB)':>12}")
print("-"*85)
for name, mhat, Phat, rt, mb in rows:
    print(f"{name:<14} {rmse(mhat,x_true):8.4f} {mae(mhat,x_true):8.4f} "
          f"{float(np.mean(Phat)):10.4f} {coverage(mhat,Phat,x_true):9.3f} "
          f"{rt:12.4f} {mb:12.2f}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2 Problem 3 — Standard Particle Filter (Bootstrap/SIR) for SV model (v2)
Fixes the weight-degeneracy histogram:
  - keep weights in float64 (avoid underflow to 0)
  - capture weights directly from the main PF run (no re-run randomness)
  - plot log10(weights) for interpretability

Outputs (PNG):
  - pf_results.npz
  - fig_pf_vs_true.png
  - fig_pf_ess.png
  - fig_pf_logweights_hist.png
  - fig_pf_compare_ekf_ukf.png   (if ekf_ukf_results.npz found)
"""

import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. Load data
# ------------------------------------------------------------
try:
    data = np.load("data_sv.npz")
    x_true = data["x_true"].astype(np.float64)
    y_obs  = data["y_obs"].astype(np.float64)
    T      = int(data["T"])
    alpha  = float(data["alpha"])
    sigma  = float(data["sigma"])
    beta   = float(data["beta"])
    seed   = int(data["seed"])
    print("Loaded data_sv.npz")
except FileNotFoundError:
    raise FileNotFoundError("data_sv.npz not found. Run Problem 1 first.")

# Optional EKF/UKF for overlay
have_ekfukf = False
try:
    res = np.load("ekf_ukf_results.npz")
    m_ekf = res["m_ekf"].astype(np.float64)
    m_ukf = res["m_ukf"].astype(np.float64)
    have_ekfukf = True
    print("Loaded ekf_ukf_results.npz for comparison")
except FileNotFoundError:
    print("ekf_ukf_results.npz not found; PF will run without EKF/UKF overlays.")

np.random.seed(seed)

# ------------------------------------------------------------
# 1. Particle filter settings
# ------------------------------------------------------------
N = 5000
ess_threshold = 0.5 * N

P0 = (sigma**2) / (1.0 - alpha**2)
m0 = 0.0

# pick calm/spike times from observations
spike_t = int(np.argmax(y_obs**2))
calm_t  = int(np.argmin(y_obs**2))

# ------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------
def transition(x_prev):
    return alpha * x_prev + sigma * np.random.randn(*x_prev.shape)

def log_likelihood_y_given_x(y_t, x_t):
    var = (beta**2) * np.exp(x_t)
    return -0.5 * (np.log(2*np.pi*var) + (y_t**2) / var)

def effective_sample_size(w):
    return 1.0 / np.sum(w**2)

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

# ------------------------------------------------------------
# 3. Run PF
# ------------------------------------------------------------
tracemalloc.start()
start = time.perf_counter()

particles = (m0 + np.sqrt(P0) * np.random.randn(N))
weights   = np.ones(N, dtype=np.float64) / N

pf_mean = np.zeros(T, dtype=np.float64)
pf_var  = np.zeros(T, dtype=np.float64)
ess_hist = np.zeros(T, dtype=np.float64)
resampled = np.zeros(T, dtype=bool)

# store weights for hist plot
w_spike = None
w_calm = None

for t in range(T):
    particles = transition(particles)

    logw = log_likelihood_y_given_x(y_obs[t], particles)
    logw -= np.max(logw)                 # stabilize
    w = np.exp(logw)                     # float64 stays nonzero
    w /= np.sum(w)                       # normalize
    weights = w

    pf_mean[t] = np.sum(weights * particles)
    pf_var[t]  = np.sum(weights * (particles - pf_mean[t])**2)

    ess = effective_sample_size(weights)
    ess_hist[t] = ess

    # snapshot weights BEFORE any resample at the chosen times
    if t == calm_t:
        w_calm = weights.copy()
    if t == spike_t:
        w_spike = weights.copy()

    if ess < ess_threshold:
        idx = systematic_resample(weights)
        particles = particles[idx]
        weights.fill(1.0 / N)
        resampled[t] = True

runtime = time.perf_counter() - start
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_mb = peak / (1024**2)

rmse_pf = float(np.sqrt(np.mean((pf_mean - x_true)**2)))
mae_pf  = float(np.mean(np.abs(pf_mean - x_true)))
avgVar_pf = float(np.mean(pf_var))

print(f"PF done. N={N}, RMSE={rmse_pf:.4f}, MAE={mae_pf:.4f}, avgVar={avgVar_pf:.4f}")
print(f"Runtime: {runtime:.3f} s  ({runtime*1e3/T:.3f} ms/step)")
print(f"Peak CPU memory during PF: {peak_mb:.2f} MB")
print(f"calm_t={calm_t}, spike_t={spike_t}")

# ------------------------------------------------------------
# 4. Save results
# ------------------------------------------------------------
np.savez(
    "pf_results.npz",
    pf_mean=pf_mean, pf_var=pf_var,
    ess_hist=ess_hist, resampled=resampled,
    N=N, ess_threshold=ess_threshold,
    calm_t=calm_t, spike_t=spike_t,
    rmse_pf=rmse_pf, mae_pf=mae_pf, avgVar_pf=avgVar_pf,
    runtime_sec=runtime, peak_cpu_mb=peak_mb
)

# ------------------------------------------------------------
# 5. Figures (PNG)
# ------------------------------------------------------------
tgrid = np.arange(T)

# (a) PF mean vs true
plt.figure(figsize=(11,4))
plt.plot(tgrid, x_true, label="true $x_t$", lw=1.0)
plt.plot(tgrid, pf_mean, label="PF mean", lw=1.0)
plt.fill_between(
    tgrid,
    pf_mean - 2*np.sqrt(pf_var),
    pf_mean + 2*np.sqrt(pf_var),
    alpha=0.2,
    label="PF $\\pm 2\\sigma$ band"
)
plt.title("Particle Filter on SV model")
plt.xlabel("time"); plt.ylabel("$x_t$")
plt.legend()
plt.tight_layout()
plt.savefig("fig_pf_vs_true.png", dpi=300)
plt.show()

# (b) ESS over time
plt.figure(figsize=(11,3.5))
plt.plot(tgrid, ess_hist, lw=1.0)
plt.axhline(ess_threshold, linestyle="--", label="resample threshold")
plt.title("Effective Sample Size (ESS) over time")
plt.xlabel("time"); plt.ylabel("ESS")
plt.legend()
plt.tight_layout()
plt.savefig("fig_pf_ess.png", dpi=300)
plt.show()

# (c) Log-weight histograms (degeneracy)
if w_calm is None or w_spike is None:
    print("Warning: did not capture calm/spike weights (unexpected).")
else:
    logw_calm = np.log10(w_calm + 1e-300)
    logw_spike = np.log10(w_spike + 1e-300)

    plt.figure(figsize=(11,3.5))
    plt.hist(logw_calm, bins=60, alpha=0.6, label=f"calm t={calm_t}")
    plt.hist(logw_spike, bins=60, alpha=0.6, label=f"spike t={spike_t}")
    plt.title("Particle log10-weight distributions\n(degeneracy during spikes)")
    plt.xlabel(r"$\log_{10}(w_t^{(i)})$"); plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_pf_logweights_hist.png", dpi=300)
    plt.show()

# (d) Compare PF vs EKF/UKF if available
if have_ekfukf:
    plt.figure(figsize=(11,4))
    plt.plot(tgrid, x_true, label="true $x_t$", lw=1.0)
    plt.plot(tgrid, pf_mean, label="PF mean", lw=1.0)
    plt.plot(tgrid, m_ekf, label="EKF(z)", lw=0.9)
    plt.plot(tgrid, m_ukf, label="UKF(z)", lw=0.9, linestyle="--")
    plt.title("PF vs EKF/UKF on SV model")
    plt.xlabel("time"); plt.ylabel("$x_t$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_pf_compare_ekf_ukf.png", dpi=300)
    plt.show()


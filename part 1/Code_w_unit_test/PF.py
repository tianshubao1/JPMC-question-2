#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonlinear/Non-Gaussian SSM with EKF/UKF and Particle Filter
Problem c
Bootstrap / SIR Particle Filter for SV model
Outputs:
  - pf_results.npz
  - fig_pf_vs_true.png
  - fig_pf_ess.png
  - fig_pf_logweights_hist.png
  - fig_pf_compare_ekf_ukf.png
"""

import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def transition(x_prev, alpha, sigma, rng):
    """SV transition prior: x_t = alpha x_{t-1} + sigma v_t."""
    return alpha * x_prev + sigma * rng.standard_normal(size=x_prev.shape)


def log_likelihood_y_given_x(y_t, x_t, beta):
    """
    True SV likelihood:
        p(y_t|x_t) = N(0, beta^2 exp(x_t))
    Returns log-likelihood for each particle.
    """
    var = (beta**2) * np.exp(x_t)
    return -0.5 * (np.log(2*np.pi*var) + (y_t**2) / var)


def effective_sample_size(w):
    """ESS = 1 / sum_i w_i^2."""
    return 1.0 / np.sum(w**2)


def systematic_resample(w, rng):
    """Systematic resampling. Returns indices."""
    Nloc = len(w)
    positions = (rng.random() + np.arange(Nloc)) / Nloc
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


def stationary_prior(alpha, sigma):
    """Stationary AR(1) prior (m0, P0)."""
    P0 = (sigma**2) / (1.0 - alpha**2)
    m0 = 0.0
    return m0, P0


# ------------------------------------------------------------
# Core PF runner (testable)
# ------------------------------------------------------------
def run_pf(y_obs, alpha, sigma, beta, N=5000, ess_frac=0.5, seed=0):
    """
    Bootstrap PF on SV model (true likelihood).

    Args:
        y_obs: (T,) observations
        alpha, sigma, beta: SV params
        N: number of particles
        ess_frac: resample threshold as fraction of N
        seed: RNG seed for determinism

    Returns dict with:
        pf_mean, pf_var, ess_hist, resampled,
        w_calm, w_spike, calm_t, spike_t,
        rmse_pf, mae_pf, avgVar_pf,
        runtime_sec, peak_cpu_mb
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    T = len(y_obs)
    rng = np.random.default_rng(seed)

    ess_threshold = ess_frac * N
    m0, P0 = stationary_prior(alpha, sigma)

    # define calm/spike times from data
    spike_t = int(np.argmax(y_obs**2))
    calm_t  = int(np.argmin(y_obs**2))

    tracemalloc.start()
    start = time.perf_counter()

    particles = (m0 + np.sqrt(P0) * rng.standard_normal(N)).astype(np.float64)
    weights   = np.ones(N, dtype=np.float64) / N

    pf_mean   = np.zeros(T, dtype=np.float64)
    pf_var    = np.zeros(T, dtype=np.float64)
    ess_hist  = np.zeros(T, dtype=np.float64)
    resampled = np.zeros(T, dtype=bool)

    w_spike = None
    w_calm  = None

    for t in range(T):
        # propagate
        particles = transition(particles, alpha, sigma, rng)

        # weight update in log-space
        logw = log_likelihood_y_given_x(y_obs[t], particles, beta)
        logw -= np.max(logw)       # stabilize
        w = np.exp(logw)
        w /= np.sum(w)
        weights = w

        # moments
        pf_mean[t] = np.sum(weights * particles)
        pf_var[t]  = np.sum(weights * (particles - pf_mean[t])**2)

        ess = effective_sample_size(weights)
        ess_hist[t] = ess

        # snapshot weights BEFORE any resample
        if t == calm_t:
            w_calm = weights.copy()
        if t == spike_t:
            w_spike = weights.copy()

        # resample if ESS low
        if ess < ess_threshold:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights.fill(1.0 / N)
            resampled[t] = True

    runtime = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024**2)

    # if x_true is not supplied we can’t compute RMSE here
    out = dict(
        pf_mean=pf_mean, pf_var=pf_var,
        ess_hist=ess_hist, resampled=resampled,
        N=N, ess_threshold=ess_threshold,
        calm_t=calm_t, spike_t=spike_t,
        w_calm=w_calm, w_spike=w_spike,
        runtime_sec=runtime, peak_cpu_mb=peak_mb
    )
    return out


def compute_metrics(pf_mean, pf_var, x_true):
    """Compute RMSE/MAE/avg posterior var."""
    x_true = np.asarray(x_true, dtype=np.float64)
    rmse_pf = float(np.sqrt(np.mean((pf_mean - x_true)**2)))
    mae_pf  = float(np.mean(np.abs(pf_mean - x_true)))
    avgVar_pf = float(np.mean(pf_var))
    return rmse_pf, mae_pf, avgVar_pf


# ------------------------------------------------------------
# main() for running as script
# ------------------------------------------------------------
def main():
    # ---- load data ----
    data = np.load("data_sv.npz")
    x_true = data["x_true"].astype(np.float64)
    y_obs  = data["y_obs"].astype(np.float64)
    T      = int(data["T"])
    alpha  = float(data["alpha"])
    sigma  = float(data["sigma"])
    beta   = float(data["beta"])
    seed   = int(data["seed"])
    print("Loaded data_sv.npz")

    # optional EKF/UKF overlay
    have_ekfukf = False
    try:
        res = np.load("ekf_ukf_results.npz")
        m_ekf = res["m_ekf"].astype(np.float64)
        m_ukf = res["m_ukf"].astype(np.float64)
        have_ekfukf = True
        print("Loaded ekf_ukf_results.npz for comparison")
    except FileNotFoundError:
        print("ekf_ukf_results.npz not found; PF runs without overlays.")
        m_ekf = m_ukf = None

    # ---- PF run ----
    out = run_pf(y_obs, alpha, sigma, beta, N=5000, ess_frac=0.5, seed=seed)
    pf_mean = out["pf_mean"]
    pf_var  = out["pf_var"]
    ess_hist = out["ess_hist"]
    resampled = out["resampled"]

    rmse_pf, mae_pf, avgVar_pf = compute_metrics(pf_mean, pf_var, x_true)
    out.update(dict(rmse_pf=rmse_pf, mae_pf=mae_pf, avgVar_pf=avgVar_pf))

    print(f"PF done. N={out['N']}, RMSE={rmse_pf:.4f}, MAE={mae_pf:.4f}, avgVar={avgVar_pf:.4f}")
    print(f"Runtime: {out['runtime_sec']:.3f} s  ({out['runtime_sec']*1e3/T:.3f} ms/step)")
    print(f"Peak CPU memory during PF: {out['peak_cpu_mb']:.2f} MB")
    print(f"calm_t={out['calm_t']}, spike_t={out['spike_t']}")

    # ---- save results ----
    np.savez("pf_results.npz", **out)

    # ---- figures ----
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
    plt.legend(); plt.tight_layout()
    plt.savefig("fig_pf_vs_true.png", dpi=300)
    plt.show()

    # (b) ESS over time
    plt.figure(figsize=(11,3.5))
    plt.plot(tgrid, ess_hist, lw=1.0)
    plt.axhline(out["ess_threshold"], linestyle="--", label="resample threshold")
    plt.title("Effective Sample Size (ESS) over time")
    plt.xlabel("time"); plt.ylabel("ESS")
    plt.legend(); plt.tight_layout()
    plt.savefig("fig_pf_ess.png", dpi=300)
    plt.show()

    # (c) log-weight histograms
    w_calm = out["w_calm"]; w_spike = out["w_spike"]
    if w_calm is not None and w_spike is not None:
        logw_calm = np.log10(w_calm + 1e-300)
        logw_spike = np.log10(w_spike + 1e-300)

        plt.figure(figsize=(11,3.5))
        plt.hist(logw_calm, bins=60, alpha=0.6, label=f"calm t={out['calm_t']}")
        plt.hist(logw_spike, bins=60, alpha=0.6, label=f"spike t={out['spike_t']}")
        plt.title("Particle log10-weight distributions\n(degeneracy during spikes)")
        plt.xlabel(r"$\log_{10}(w_t^{(i)})$"); plt.ylabel("count")
        plt.legend(); plt.tight_layout()
        plt.savefig("fig_pf_logweights_hist.png", dpi=300)
        plt.show()

    # (d) compare vs EKF/UKF
    if have_ekfukf:
        plt.figure(figsize=(11,4))
        plt.plot(tgrid, x_true, label="true $x_t$", lw=1.0)
        plt.plot(tgrid, pf_mean, label="PF mean", lw=1.0)
        plt.plot(tgrid, m_ekf, label="EKF(z)", lw=0.9)
        plt.plot(tgrid, m_ukf, label="UKF(z)", lw=0.9, linestyle="--")
        plt.title("PF vs EKF/UKF on SV model")
        plt.xlabel("time"); plt.ylabel("$x_t$")
        plt.legend(); plt.tight_layout()
        plt.savefig("fig_pf_compare_ekf_ukf.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 2c — Compare EDH, LEDH, Kernel PFF on SV SSM (from Problem 1 PF)

SV SSM:
    x_t = alpha * x_{t-1} + sigma * v_t
    y_t = beta * exp(0.5*x_t) * w_t
    v_t, w_t ~ N(0,1)

We compare:
  - Baseline PF
  - EDH flow PF (global Daum–Huang style linearization)
  - LEDH flow PF (local linearization per particle)
  - Kernel PFF PF (Hu & van Leeuwen 2021 kernel-embedded flow)

Diagnostics:
  - RMSE of filtered mean vs truth
  - ESS over time
  - Flow magnitude during pseudo-time (avg |dx/dλ|)
  - Jacobian magnitude during pseudo-time (|∂x_new/∂x_old|)

NOTE:
SV likelihood is non-additive. We therefore use the likelihood score/Hessian
inside EDH/LEDH flows, which is the proper SSH replacement for SV.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict

FlowType = Literal["baseline", "edh", "ledh", "kernel"]


# ============================================================
#  SV Model
# ============================================================

@dataclass
class SVParams:
    T: int = 500
    alpha: float = 0.98
    sigma: float = 0.15
    beta: float = 0.65
    seed: int = 0


def simulate_sv_np(params: SVParams) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate SV model in NumPy."""
    rng = np.random.default_rng(params.seed)
    T, alpha, sigma, beta = params.T, params.alpha, params.sigma, params.beta

    x = np.zeros(T)
    y = np.zeros(T)

    var0 = sigma**2 / (1.0 - alpha**2)
    x_prev = rng.normal(0.0, np.sqrt(var0))

    for t in range(T):
        x_t = alpha * x_prev + sigma * rng.normal()
        y_t = beta * np.exp(0.5 * x_t) * rng.normal()
        x[t], y[t] = x_t, y_t
        x_prev = x_t

    return x, y


def sv_loglik(x: np.ndarray, y: float, beta: float) -> np.ndarray:
    """log p(y|x) for SV (vectorized over x)."""
    b2 = beta**2
    expx = np.exp(x)
    return -0.5 * (
        np.log(2.0 * np.pi)
        + np.log(b2 * expx)
        + (y**2) / (b2 * expx)
    )


def sv_score_and_hessian(x: np.ndarray, y: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient and Hessian of log p(y|x):
        score = d/dx loglik
        hess  = d^2/dx^2 loglik
    """
    b2 = beta**2
    expx = np.exp(x)
    ratio = (y**2) / (b2 * expx)

    score = 0.5 * (ratio - 1.0)
    hess  = -0.5 * ratio
    return score, hess


# ============================================================
#  Flow updates (EDH / LEDH / Kernel)
# ============================================================

def edh_flow_update(
    X0: np.ndarray,
    y: float,
    beta: float,
    flow_steps: int = 8,
    flow_dt: float = 0.015,
    max_flow: float = 2.0
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    EDH: global flow using score/Hessian at ensemble mean.
    Returns updated particles and diagnostics.
    """
    X = X0.copy()
    N = X.size

    flow_mag = np.zeros(flow_steps)
    jac_mag  = np.zeros(flow_steps)

    for k in range(flow_steps):
        mu = X.mean()
        P  = X.var(ddof=1) + 1e-6
        invP = 1.0 / P

        score_mu, hess_mu = sv_score_and_hessian(mu, y, beta)
        score_mu = float(score_mu)
        hess_mu  = float(hess_mu)

        # per-particle grad log posterior (prior+lik), but lik part global
        grad_post = -(X - mu) * invP + score_mu     # (N,)
        hess_post = -invP + hess_mu                # scalar (global)

        flow = P * grad_post                       # (N,)
        flow = np.clip(flow, -max_flow, max_flow)

        X = X + flow_dt * flow

        flow_mag[k] = np.mean(np.abs(flow))

        # Jacobian for 1D Euler step:
        # x_new = x + dt * P * (-(x-mu)/P + score_mu)
        # derivative wrt x approx: 1 + dt * P * hess_post
        J = 1.0 + flow_dt * P * hess_post
        jac_mag[k] = abs(J)

    return X, {"flow_mag": flow_mag, "jac_mag": jac_mag}


def ledh_flow_update(
    X0: np.ndarray,
    y: float,
    beta: float,
    flow_steps: int = 8,
    flow_dt: float = 0.015,
    max_flow: float = 2.0
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    LEDH: local flow using per-particle score/Hessian.
    Returns updated particles and diagnostics.
    """
    X = X0.copy()
    N = X.size

    flow_mag = np.zeros(flow_steps)
    jac_mag  = np.zeros(flow_steps)

    for k in range(flow_steps):
        mu = X.mean()
        P  = X.var(ddof=1) + 1e-6
        invP = 1.0 / P

        score_i, hess_i = sv_score_and_hessian(X, y, beta)

        grad_post = -(X - mu) * invP + score_i
        hess_post = -invP + hess_i

        flow = P * grad_post
        flow = np.clip(flow, -max_flow, max_flow)
        X = X + flow_dt * flow

        flow_mag[k] = np.mean(np.abs(flow))

        J_i = 1.0 + flow_dt * P * hess_post
        jac_mag[k] = np.mean(np.abs(J_i))

    return X, {"flow_mag": flow_mag, "jac_mag": jac_mag}


def kernel_pff_update(
    X0: np.ndarray,
    y: float,
    beta: float,
    alpha_k: float = 1.0,
    flow_steps: int = 8,
    flow_dt: float = 0.015,
    max_flow: float = 2.0
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Kernel PFF (1D scalar kernel):
      - compute posterior score at all particles
      - smooth via Gaussian kernel weights
    """
    X = X0.copy()
    N = X.size

    flow_mag = np.zeros(flow_steps)
    jac_mag  = np.zeros(flow_steps)

    for k in range(flow_steps):
        mu = X.mean()
        P  = X.var(ddof=1) + 1e-6
        invP = 1.0 / P

        score_i, hess_i = sv_score_and_hessian(X, y, beta)
        grad_post_i = -(X - mu) * invP + score_i
        hess_post_i = -invP + hess_i

        flows = np.zeros_like(X)
        Js     = np.zeros_like(X)

        for i in range(N):
            d2 = (X - X[i])**2 / (P + 1e-12)
            K  = np.exp(-0.5 * d2 / alpha_k)
            w  = K / (K.sum() + 1e-12)

            g_smooth = (w * grad_post_i).sum()
            h_smooth = (w * hess_post_i).sum()

            flow_i = P * g_smooth
            flow_i = np.clip(flow_i, -max_flow, max_flow)

            flows[i] = flow_i
            Js[i]    = 1.0 + flow_dt * P * h_smooth

        X = X + flow_dt * flows

        flow_mag[k] = np.mean(np.abs(flows))
        jac_mag[k]  = np.mean(np.abs(Js))

    return X, {"flow_mag": flow_mag, "jac_mag": jac_mag}


# ============================================================
#  Particle Filter wrapper
# ============================================================

def systematic_resample(rng: np.random.Generator, w: np.ndarray) -> np.ndarray:
    N = len(w)
    u0 = rng.random() / N
    cs = np.cumsum(w)
    idx = np.zeros(N, dtype=int)
    j = 0
    for i in range(N):
        u = u0 + i / N
        while u > cs[j]:
            j += 1
        idx[i] = j
    return idx


def run_sv_pf(
    params: SVParams,
    N: int = 500,
    method: FlowType = "baseline",
    obs_gap: int = 1,
    flow_steps: int = 8,
    flow_dt: float = 0.015,
    alpha_k: float = 1.0,
    max_flow: float = 2.0,
    seed: int = 0
) -> Dict[str, np.ndarray]:
    """
    PF with optional flow proposal.
    - If obs_gap>1, only assimilate every obs_gap steps.
    Returns dict with rmse, ess, diag_flow_mag, diag_jac_mag.
    """
    x_true, y_obs = simulate_sv_np(params)
    rng = np.random.default_rng(seed)

    T = params.T
    alpha, sigma, beta = params.alpha, params.sigma, params.beta

    # init from stationary prior
    var0 = sigma**2 / (1 - alpha**2)
    X = rng.normal(0.0, np.sqrt(var0), size=N)

    rmse = np.zeros(T)
    ess  = np.zeros(T)

    # store diagnostics for a few assimilation steps
    diag_flow_mag = []
    diag_jac_mag  = []

    for t in range(T):
        # predict
        X = alpha * X + sigma * rng.normal(size=N)

        assimilate = (t % obs_gap == 0)

        if assimilate and method != "baseline":
            if method == "edh":
                X, diag = edh_flow_update(X, y_obs[t], beta,
                                          flow_steps=flow_steps, flow_dt=flow_dt,
                                          max_flow=max_flow)
            elif method == "ledh":
                X, diag = ledh_flow_update(X, y_obs[t], beta,
                                           flow_steps=flow_steps, flow_dt=flow_dt,
                                           max_flow=max_flow)
            elif method == "kernel":
                X, diag = kernel_pff_update(X, y_obs[t], beta,
                                            alpha_k=alpha_k,
                                            flow_steps=flow_steps, flow_dt=flow_dt,
                                            max_flow=max_flow)
            diag_flow_mag.append(diag["flow_mag"])
            diag_jac_mag.append(diag["jac_mag"])

        # weights with true SV likelihood
        logw = sv_loglik(X, y_obs[t], beta)
        logw -= logw.max()
        w = np.exp(logw) + 1e-300
        w /= w.sum()

        ess[t] = 1.0 / np.sum(w**2)

        # resample
        idx = systematic_resample(rng, w)
        X = X[idx]

        rmse[t] = abs(X.mean() - x_true[t])

    diag_flow_mag = np.array(diag_flow_mag) if diag_flow_mag else np.zeros((0, flow_steps))
    diag_jac_mag  = np.array(diag_jac_mag)  if diag_jac_mag  else np.zeros((0, flow_steps))

    return {
        "rmse": rmse,
        "ess": ess,
        "x_true": x_true,
        "y_obs": y_obs,
        "flow_mag": diag_flow_mag,
        "jac_mag": diag_jac_mag,
    }


# ============================================================
#  Experiments + Plots
# ============================================================

def compare_methods_on_sv(
    params: SVParams,
    N: int = 500,
    obs_gap: int = 1,
    seed: int = 0,
    save_dir: str = "fig2c",
    tag: str = "case1"
):
    os.makedirs(save_dir, exist_ok=True)

    methods: FlowType = ["baseline", "edh", "ledh", "kernel"]
    out = {}

    for m in methods:
        out[m] = run_sv_pf(
            params=params, N=N, method=m, obs_gap=obs_gap,
            flow_steps=8, flow_dt=0.015, alpha_k=1.0, max_flow=2.0,
            seed=seed
        )

    t = np.arange(params.T)

    # --- RMSE plot ---
    plt.figure(figsize=(9, 4.8))
    for m in methods:
        plt.plot(t, out[m]["rmse"], label=m.upper())
    plt.yscale("log")
    plt.xlabel("time")
    plt.ylabel("RMSE |mean(x)-x_true|")
    plt.title(f"SV SSM RMSE comparison (obs_gap={obs_gap}, beta={params.beta})")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    rmse_path = os.path.join(save_dir, f"{tag}_rmse.png")
    plt.savefig(rmse_path, dpi=200)
    plt.show()

    # --- ESS plot ---
    plt.figure(figsize=(9, 4.2))
    for m in methods:
        plt.plot(t, out[m]["ess"], label=m.upper())
    plt.xlabel("time")
    plt.ylabel("ESS")
    plt.title(f"Effective Sample Size (obs_gap={obs_gap}, beta={params.beta})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ess_path = os.path.join(save_dir, f"{tag}_ess.png")
    plt.savefig(ess_path, dpi=200)
    plt.show()

    # --- Flow magnitude diagnostics ---
    plt.figure(figsize=(9, 4.2))
    for m in ["edh", "ledh", "kernel"]:
        fm = out[m]["flow_mag"]
        if fm.size > 0:
            plt.plot(fm.mean(axis=0), label=f"{m.upper()} avg |flow|")
    plt.xlabel("pseudo-time step")
    plt.ylabel("avg |dx/dλ|")
    plt.title("Flow magnitude (avg over assimilations)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    flow_path = os.path.join(save_dir, f"{tag}_flowmag.png")
    plt.savefig(flow_path, dpi=200)
    plt.show()

    # --- Jacobian magnitude diagnostics ---
    plt.figure(figsize=(9, 4.2))
    for m in ["edh", "ledh", "kernel"]:
        jm = out[m]["jac_mag"]
        if jm.size > 0:
            plt.plot(jm.mean(axis=0), label=f"{m.upper()} avg |Jac|")
    plt.xlabel("pseudo-time step")
    plt.ylabel("avg |Jacobian|")
    plt.title("Jacobian magnitude (avg over assimilations)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    jac_path = os.path.join(save_dir, f"{tag}_jacmag.png")
    plt.savefig(jac_path, dpi=200)
    plt.show()

    print(f"[saved] {rmse_path}")
    print(f"[saved] {ess_path}")
    print(f"[saved] {flow_path}")
    print(f"[saved] {jac_path}")

    return out



# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    SAVE_DIR = "fig2c"

    # (Case 1) Standard assimilation
    params1 = SVParams(T=300, alpha=0.98, sigma=0.15, beta=0.65, seed=0)
    compare_methods_on_sv(params1, N=600, obs_gap=1, seed=0,
                          save_dir=SAVE_DIR, tag="case1_std")

    # (Case 2) Sparse observations
    compare_methods_on_sv(params1, N=600, obs_gap=5, seed=0,
                          save_dir=SAVE_DIR, tag="case2_sparse")

    # (Case 3) More informative observations
    params3 = SVParams(T=300, alpha=0.98, sigma=0.15, beta=1.0, seed=0)
    compare_methods_on_sv(params3, N=600, obs_gap=1, seed=0,
                          save_dir=SAVE_DIR, tag="case3_strongobs")


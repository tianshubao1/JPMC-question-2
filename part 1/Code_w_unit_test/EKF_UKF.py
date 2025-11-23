#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Nonlinear/Non-Gaussian SSM with EKF/UKF and Particle Filter
- Problem b
- EKF and UKF for the Stochastic Volatility (SV) SSM
- Loads data_sv.npz
- EKF on transformed observations z_t = log(y_t^2 + eps)
- UKF on same z_t (scaled sigma points)
- Naive UKF on raw y_t (additive-Gaussian assumption) as failure demo
- Saves metrics + plots (only in main())

"""

import time
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# constants for z transform
# ---------------------------
MU_ETA  = -1.270362845
VAR_ETA = 4.934802200


def make_z_observation(y_obs, beta, eps=1e-6):
    """
    Compute transformed pseudo-observation z_t and its linear-Gaussian params:
        z_t = log(y_t^2 + eps) ≈ x_t + c + e_t, e_t~N(0,Rz)

    Returns:
        z_obs, c, Rz
    """
    z_obs = np.log(y_obs**2 + eps).astype(np.float32)
    c  = np.log(beta**2) + MU_ETA
    Rz = VAR_ETA
    return z_obs, c, Rz


def stationary_prior(alpha, sigma):
    """Return stationary AR(1) prior (m0, P0)."""
    P0 = (sigma**2) / (1.0 - alpha**2)
    m0 = 0.0
    return m0, P0


# ------------------------------------------------------------
# EKF on z_t (scalar KF)
# ------------------------------------------------------------
def run_ekf_z(z_obs, alpha, sigma, c, Rz, m0, P0):
    T = len(z_obs)
    m_f = np.zeros(T, np.float32)
    P_f = np.zeros(T, np.float32)

    m_prev, P_prev = m0, P0
    for t in range(T):
        m_pred = alpha * m_prev
        P_pred = alpha**2 * P_prev + sigma**2

        r = z_obs[t] - (m_pred + c)
        S = P_pred + Rz
        K = P_pred / S

        m_post = m_pred + K * r
        P_post = (1 - K)**2 * P_pred + (K**2) * Rz  # Joseph scalar

        m_f[t], P_f[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post

    return m_f, P_f


# ------------------------------------------------------------
# UKF utilities
# ------------------------------------------------------------
def sigma_points_1d(m, P, alpha_ut=1e-3, beta_ut=2.0, kappa_ut=0.0):
    """
    1D scaled unscented transform sigma points and weights.
    Returns:
        X: (3,) sigma points
        Wm, Wc: (3,) weights for mean/cov
    """
    n = 1
    lam = alpha_ut**2 * (n + kappa_ut) - n
    gamma = np.sqrt(n + lam)

    X0 = m
    X1 = m + gamma * np.sqrt(P)
    X2 = m - gamma * np.sqrt(P)
    X = np.array([X0, X1, X2], dtype=np.float32)

    Wm = np.full(2*n + 1, 1.0/(2*(n+lam)), dtype=np.float32)
    Wc = Wm.copy()
    Wm[0] = lam/(n+lam)
    Wc[0] = lam/(n+lam) + (1 - alpha_ut**2 + beta_ut)

    return X, Wm, Wc


def run_ukf_z(z_obs, alpha, sigma, c, Rz, m0, P0,
             alpha_ut=1e-3, beta_ut=2.0, kappa_ut=0.0):
    T = len(z_obs)
    m_f = np.zeros(T, np.float32)
    P_f = np.zeros(T, np.float32)

    m_prev, P_prev = m0, P0
    for t in range(T):
        m_pred = alpha * m_prev
        P_pred = alpha**2 * P_prev + sigma**2

        X, Wm, Wc = sigma_points_1d(m_pred, P_pred, alpha_ut, beta_ut, kappa_ut)
        Z = X + c  # h_z(x)=x+c

        z_hat = np.sum(Wm * Z)
        P_zz = np.sum(Wc * (Z - z_hat)**2) + Rz
        P_xz = np.sum(Wc * (X - m_pred) * (Z - z_hat))

        K = P_xz / P_zz
        r = z_obs[t] - z_hat

        m_post = m_pred + K * r
        P_post = P_pred - K**2 * P_zz

        m_f[t], P_f[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post

    return m_f, P_f


# ------------------------------------------------------------
# Naive UKF on raw y_t (failure case)
# ------------------------------------------------------------
def run_naive_ukf_y(y_obs, alpha, sigma, m0, P0,
                    Ry=1.0, alpha_ut=1e-3, beta_ut=2.0, kappa_ut=0.0):
    T = len(y_obs)
    m_f = np.zeros(T, np.float32)
    P_f = np.zeros(T, np.float32)
    K_hist = np.zeros(T, np.float32)

    m_prev, P_prev = m0, P0
    for t in range(T):
        m_pred = alpha * m_prev
        P_pred = alpha**2 * P_prev + sigma**2

        X, Wm, Wc = sigma_points_1d(m_pred, P_pred, alpha_ut, beta_ut, kappa_ut)

        # conditional mean model h_y(x)=0  (wrong on purpose)
        Z = np.zeros_like(X)

        z_hat = np.sum(Wm * Z)
        P_zz = np.sum(Wc * (Z - z_hat)**2) + Ry
        P_xz = np.sum(Wc * (X - m_pred) * (Z - z_hat))

        K = P_xz / P_zz
        r = y_obs[t] - z_hat

        m_post = m_pred + K * r
        P_post = P_pred - K**2 * P_zz

        m_f[t], P_f[t] = m_post, P_post
        K_hist[t] = K
        m_prev, P_prev = m_post, P_post

    return m_f, P_f, K_hist


def compute_metrics(m_est, P_est, x_true):
    rmse = float(np.sqrt(np.mean((m_est - x_true)**2)))
    mae  = float(np.mean(np.abs(m_est - x_true)))
    avgP = float(np.mean(P_est))
    return rmse, mae, avgP


# ------------------------------------------------------------
# main() only for running + plotting + saving
# ------------------------------------------------------------
def main():
    # ---- load data ----
    data = np.load("data_sv.npz")
    x_true = data["x_true"].astype(np.float32)
    y_obs  = data["y_obs"].astype(np.float32)
    T      = int(data["T"])
    alpha  = float(data["alpha"])
    sigma  = float(data["sigma"])
    beta   = float(data["beta"])
    seed   = int(data["seed"])

    np.random.seed(seed)

    # ---- build z ----
    eps = 1e-6
    z_obs, c, Rz = make_z_observation(y_obs, beta, eps=eps)
    m0, P0 = stationary_prior(alpha, sigma)

    # ---- EKF ----
    t0 = time.perf_counter()
    m_ekf, P_ekf = run_ekf_z(z_obs, alpha, sigma, c, Rz, m0, P0)
    rt_ekf = time.perf_counter() - t0

    # ---- UKF ----
    t0 = time.perf_counter()
    m_ukf, P_ukf = run_ukf_z(z_obs, alpha, sigma, c, Rz, m0, P0)
    rt_ukf = time.perf_counter() - t0

    # ---- naive UKF ----
    Ry_naive = np.var(y_obs)
    m_nu, P_nu, K_nu = run_naive_ukf_y(y_obs, alpha, sigma, m0, P0, Ry=Ry_naive)

    # ---- metrics ----
    rmse_ekf, mae_ekf, avgP_ekf = compute_metrics(m_ekf, P_ekf, x_true)
    rmse_ukf, mae_ukf, avgP_ukf = compute_metrics(m_ukf, P_ukf, x_true)
    rmse_nu,  mae_nu,  avgP_nu  = compute_metrics(m_nu,  P_nu,  x_true)

    print(f"EKF(z): RMSE={rmse_ekf:.4f}, MAE={mae_ekf:.4f}, avgP={avgP_ekf:.4f}")
    print(f"UKF(z): RMSE={rmse_ukf:.4f}, MAE={mae_ukf:.4f}, avgP={avgP_ukf:.4f}")
    print(f"Naive UKF(y): RMSE={rmse_nu:.4f}, MAE={mae_nu:.4f}, avgP={avgP_nu:.4f}")
    print(f"Runtime EKF(z): {rt_ekf*1e3:.2f} ms/step")
    print(f"Runtime UKF(z): {rt_ukf*1e3:.2f} ms/step")

    # ---- save ----
    np.savez(
        "ekf_ukf_results.npz",
        m_ekf=m_ekf, P_ekf=P_ekf, runtime_ekf=rt_ekf,
        m_ukf=m_ukf, P_ukf=P_ukf, runtime_ukf=rt_ukf,
        m_ukf_naive=m_nu, P_ukf_naive=P_nu, K_naive=K_nu,
        z_obs=z_obs, eps=eps, c=c, Rz=Rz,
        rmse_ekf=rmse_ekf, mae_ekf=mae_ekf, avgP_ekf=avgP_ekf,
        rmse_ukf=rmse_ukf, mae_ukf=mae_ukf, avgP_ukf=avgP_ukf,
        rmse_nu=rmse_nu,  mae_nu=mae_nu,  avgP_nu=avgP_nu,
        Ry_naive=Ry_naive
    )

    # ---- plots ----
    tgrid = np.arange(T)

    plt.figure(figsize=(11, 4))
    plt.plot(tgrid, x_true, label="true $x_t$", lw=1.0)
    plt.plot(tgrid, m_ekf, label="EKF mean", lw=1.0)
    plt.plot(tgrid, m_ukf, label="UKF mean", lw=1.0, linestyle="--")
    plt.fill_between(
        tgrid,
        m_ukf - 2*np.sqrt(P_ukf),
        m_ukf + 2*np.sqrt(P_ukf),
        alpha=0.15,
        label="UKF $\\pm 2\\sigma$ band"
    )
    plt.title("EKF vs UKF on SV model using transformed observation $z_t$")
    plt.xlabel("time")
    plt.ylabel("$x_t$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_ekf_ukf_vs_true.png")
    plt.show()

    plt.figure(figsize=(11, 4))
    plt.plot(tgrid, x_true, label="true $x_t$", lw=1.0)
    plt.plot(tgrid, m_nu, label="Naive UKF on $y_t$", lw=1.0)
    plt.title("Naive UKF on raw $y_t$ (additive-Gaussian assumption) — little/no update")
    plt.xlabel("time")
    plt.ylabel("$x_t$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_naive_ukf_fail.png")
    plt.show()


if __name__ == "__main__":
    main()

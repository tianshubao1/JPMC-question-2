#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lgssm_kalman_tf.py

- Example-2 2D tracking LGSSM
- Synthetic data
- Kalman filter (Joseph + standard covariance update)
- Save figures as PNG
- Print scalars for LaTeX
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass


# =========================
# Config
# =========================
T = 200
dt = 1.0
q  = 0.1
r  = 0.5
seed = 0
outdir = "outputs"
use_float64 = True
# =========================


@dataclass
class LGSSMParams:
    A: tf.Tensor
    C: tf.Tensor
    Q: tf.Tensor
    R: tf.Tensor
    m0: tf.Tensor
    P0: tf.Tensor
    B: tf.Tensor
    D: tf.Tensor


def make_tracking_lgssm(dt=1.0, q=0.1, r=0.5):
    A = np.array([
        [1, dt, 0,  0],
        [0,  1, 0,  0],
        [0,  0, 1, dt],
        [0,  0, 0,  1],
    ], dtype=np.float64)

    C = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)

    B = np.eye(4, dtype=np.float64)
    D = np.eye(2, dtype=np.float64)

    Q = (q**2) * np.eye(4, dtype=np.float64)
    R = (r**2) * np.eye(2, dtype=np.float64)

    m0 = np.array([0., 1., 0., 0.5], dtype=np.float64)
    P0 = np.eye(4, dtype=np.float64)

    return LGSSMParams(
        A=tf.constant(A),
        C=tf.constant(C),
        Q=tf.constant(Q),
        R=tf.constant(R),
        m0=tf.constant(m0),
        P0=tf.constant(P0),
        B=tf.constant(B),
        D=tf.constant(D),
    )


def simulate_lgssm(params: LGSSMParams, T=100, seed=0):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    A, C, Q, R, m0, P0 = params.A, params.C, params.Q, params.R, params.m0, params.P0
    dx = A.shape[0]
    dy = C.shape[0]

    chol_Q  = tf.linalg.cholesky(Q)
    chol_R  = tf.linalg.cholesky(R)
    chol_P0 = tf.linalg.cholesky(P0)

    x_arr = tf.TensorArray(A.dtype, size=T)
    y_arr = tf.TensorArray(A.dtype, size=T)

    x_prev = m0[:, None] + chol_P0 @ tf.random.normal((dx, 1), dtype=A.dtype)

    for t in tf.range(T):
        v = chol_Q @ tf.random.normal((dx, 1), dtype=A.dtype)
        x_t = A @ x_prev + v

        w = chol_R @ tf.random.normal((dy, 1), dtype=A.dtype)
        y_t = C @ x_t + w

        x_arr = x_arr.write(t, tf.squeeze(x_t, -1))
        y_arr = y_arr.write(t, tf.squeeze(y_t, -1))
        x_prev = x_t

    return x_arr.stack(), y_arr.stack()


def kalman_filter(y, params: LGSSMParams, use_joseph=True):
    A, C, Q, R, m0, P0, B, D = (
        params.A, params.C, params.Q, params.R,
        params.m0, params.P0, params.B, params.D
    )
    T = y.shape[0]
    dx = A.shape[0]
    I  = tf.eye(dx, dtype=A.dtype)

    m_arr = tf.TensorArray(A.dtype, size=T)
    P_arr = tf.TensorArray(A.dtype, size=T)
    S_arr = tf.TensorArray(A.dtype, size=T)

    m_prev = m0[:, None]
    P_prev = P0

    for t in tf.range(T):
        m_pred = A @ m_prev
        P_pred = A @ P_prev @ tf.transpose(A) + B @ Q @ tf.transpose(B)

        y_t = y[t][:, None]
        r   = y_t - C @ m_pred
        S   = C @ P_pred @ tf.transpose(C) + D @ R @ tf.transpose(D)

        K_T = tf.linalg.solve(S, C @ P_pred, adjoint=True)
        K   = tf.transpose(K_T)

        m_filt = m_pred + K @ r

        if use_joseph:
            KC = K @ C
            P_filt = (I - KC) @ P_pred @ tf.transpose(I - KC) \
                     + K @ (D @ R @ tf.transpose(D)) @ tf.transpose(K)
        else:
            # Standard covariance update (non-Joseph)
            P_filt = (I - K @ C) @ P_pred

        P_filt = 0.5 * (P_filt + tf.transpose(P_filt))

        m_arr = m_arr.write(t, tf.squeeze(m_filt, -1))
        P_arr = P_arr.write(t, P_filt)
        S_arr = S_arr.write(t, S)

        m_prev, P_prev = m_filt, P_filt

    return m_arr.stack(), P_arr.stack(), S_arr.stack()


def cov_diagnostics(P_seq):
    P_np = P_seq.numpy()
    T, dx, _ = P_np.shape
    min_eigs = np.zeros(T)
    sym_err  = np.zeros(T)
    for t in range(T):
        P = P_np[t]
        sym_err[t] = np.linalg.norm(P - P.T, ord='fro')
        min_eigs[t] = np.min(np.linalg.eigvalsh(P))
    return min_eigs, sym_err


def cond_number(A):
    s = np.linalg.svd(A, compute_uv=False)
    return s[0] / s[-1]


def compute_condition_numbers(P_seq, S_seq):
    P_np = P_seq.numpy()
    S_np = S_seq.numpy()
    T = P_np.shape[0]
    cond_P = np.zeros(T)
    cond_S = np.zeros(T)
    for t in range(T):
        cond_P[t] = cond_number(P_np[t])
        cond_S[t] = cond_number(S_np[t])
    return cond_P, cond_S



def plot_synthetic(x_true, y_obs, outdir):
    x_np = x_true.numpy()
    y_np = y_obs.numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # ----- latent states -----
    axes[0].plot(x_np[:, 0], label=r"$x_{\mathrm{pos}}$")
    axes[0].plot(x_np[:, 1], label=r"$x_{\mathrm{vel}}$")
    axes[0].plot(x_np[:, 2], label=r"$y_{\mathrm{pos}}$")
    axes[0].plot(x_np[:, 3], label=r"$y_{\mathrm{vel}}$")
    axes[0].set_title(r"Latent states $x_t$")
    axes[0].set_ylabel("state value")
    axes[0].legend(loc="upper left")

    # ----- observations -----
    axes[1].plot(y_np[:, 0], label=r"$x_{\mathrm{obs}}$")
    axes[1].plot(y_np[:, 1], label=r"$y_{\mathrm{obs}}$")
    axes[1].set_title(r"Observations $y_t$")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("obs value")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    path = os.path.join(outdir, "fig_synthetic_trajectories.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path



def plot_filtered_vs_true(x_true, m_filt, outdir):
    x_np = x_true.numpy()
    m_np = m_filt.numpy()
    dx = x_np.shape[1]

    fig, axes = plt.subplots(dx, 1, figsize=(10, 2.2*dx), sharex=True)
    if dx == 1:
        axes = [axes]
    for i in range(dx):
        axes[i].plot(x_np[:, i], label="true")
        axes[i].plot(m_np[:, i], label="filtered mean")
        axes[i].set_ylabel(f"x[{i}]")
        axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("time")

    plt.tight_layout()
    path = os.path.join(outdir, "fig_filtered_vs_true.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_joseph_stability(min_eig_j, sym_j, min_eig_std, sym_std, outdir):
    T = len(min_eig_j)
    t = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, min_eig_j, label="Joseph")
    axes[0].plot(t, min_eig_std, label="Standard")
    axes[0].axhline(0, linestyle="--")
    axes[0].set_title("Minimum eigenvalue of filtered covariance")
    axes[0].set_ylabel("min eig")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, sym_j, label="Joseph")
    axes[1].plot(t, sym_std, label="Standard")
    axes[1].set_title(r"Symmetry error $\|P_t - P_t^\top\|_F$")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("sym err")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(outdir, "fig_joseph_stability.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_condition_numbers(condP, condS, outdir):
    T = len(condP)
    t = np.arange(T)

    plt.figure(figsize=(10, 4))
    plt.plot(t, condP, label=r"$\kappa(P_{t|t-1})$")
    plt.plot(t, condS, label=r"$\kappa(S_t)$")

    plt.yscale("log")
    plt.xlabel("time")
    plt.ylabel("condition number (log)")
    plt.title(r"Condition numbers of $P_{t|t-1}$ and $S_t$")
    plt.legend(loc="upper right")  # make legend explicit
    plt.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(outdir, "fig_condition_numbers.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def main():
    os.makedirs(outdir, exist_ok=True)

    if use_float64:
        tf.keras.backend.set_floatx("float64")

    params = make_tracking_lgssm(dt=dt, q=q, r=r)

    x_true, y_obs = simulate_lgssm(params, T=T, seed=seed)

    m_j, P_j, S_j = kalman_filter(y_obs, params, use_joseph=True)
    m_std, P_std, S_std = kalman_filter(y_obs, params, use_joseph=False)

    mse = np.mean((m_j.numpy() - x_true.numpy())**2)
    l1_err = np.mean(np.abs(m_j.numpy() - x_true.numpy()))
    pos_rmse = np.sqrt(np.mean((m_j.numpy()[:, [0,2]] - x_true.numpy()[:, [0,2]])**2))
    vel_rmse = np.sqrt(np.mean((m_j.numpy()[:, [1,3]] - x_true.numpy()[:, [1,3]])**2))
    trace_P = np.mean([np.trace(P) for P in P_j.numpy()])

    min_eig_j, sym_j = cov_diagnostics(P_j)
    min_eig_std, sym_std = cov_diagnostics(P_std)

    condP_j, condS_j = compute_condition_numbers(P_j, S_j)

    f1 = plot_synthetic(x_true, y_obs, outdir)
    f2 = plot_filtered_vs_true(x_true, m_j, outdir)
    f3 = plot_joseph_stability(min_eig_j, sym_j, min_eig_std, sym_std, outdir)
    f4 = plot_condition_numbers(condP_j, condS_j, outdir)

    print("\n=== Scalars for LaTeX tables ===")
    print(f"MSE(filtered mean vs true): {mse:.6f}")
    print(f"Mean L1 filtering error:    {l1_err:.6f}")
    print(f"Position RMSE:              {pos_rmse:.6f}")
    print(f"Velocity RMSE:              {vel_rmse:.6f}")
    print(f"Average trace(P_t):         {trace_P:.6f}")
    print(f"Avg cond(P_t):              {np.mean(condP_j):.6f}")
    print(f"Avg cond(S_t):              {np.mean(condS_j):.6f}")


if __name__ == "__main__":
    main()

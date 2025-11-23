#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2 Problem 2b — Kernel-Embedded Particle Flow Filter (Hu & van Leeuwen 2021 )

Includes:
  (Fig 2) 2D collapse demo: scalar vs diagonal matrix-valued kernel
  (Fig 3) High-dim collapse projection: scalar vs matrix kernel

"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal, Sequence

KernelType = Literal["scalar", "matrix"]


# ============================================================
#  Utilities
# ============================================================

def shrink_cov(sample_cov: np.ndarray, shrink: float = 1e-2, ridge: float = 1e-6) -> np.ndarray:
    """
    Make covariance well-conditioned:
      - symmetrize
      - shrink toward scaled identity
      - add ridge
    """
    cov = 0.5 * (sample_cov + sample_cov.T)
    d = cov.shape[0]
    tr = np.trace(cov) / d
    cov = (1.0 - shrink) * cov + shrink * tr * np.eye(d)
    cov = cov + ridge * np.eye(d)
    return cov


# ============================================================
#  RKHS Kernel Particle Flow Core
# ============================================================

@dataclass
class GaussianPrior:
    mu: np.ndarray          # (d,)
    cov: np.ndarray         # (d,d)
    std_floor: float = 1e-3

    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])

    @property
    def std(self) -> np.ndarray:
        std = np.sqrt(np.diag(self.cov))
        return np.maximum(std, self.std_floor)


class KernelEmbeddedPFF:
    """
    Kernel-embedded particle flow filter in RKHS.
    Supports:
      - scalar kernel K(x,z)=k(||x-z||^2)
      - diagonal matrix-valued kernel K(x,z)=diag(k_a((x_a-z_a)^2/sigma_a^2))
    """

    def __init__(self, kernel_type: KernelType = "scalar", alpha: float = 0.1, jitter: float = 1e-9):
        self.kernel_type = kernel_type
        self.alpha = float(alpha)
        self.jitter = float(jitter)

    # ---------- base gaussian kernel k(d^2) ----------
    def k(self, d2: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * d2 / self.alpha)

    # ---------- scalar kernel values K(x, Z) ----------
    def scalar_kernel_vals(self, x: np.ndarray, Z: np.ndarray, cov: np.ndarray) -> np.ndarray:
        A = np.linalg.inv(cov)
        diffs = Z - x[None, :]                          # (N,d)
        d2 = np.einsum("ni,ij,nj->n", diffs, A, diffs)  # (N,)
        return self.k(d2)

    # ---------- diagonal matrix-valued kernel values ----------
    def matrix_kernel_vals(self, x: np.ndarray, Z: np.ndarray, std: np.ndarray) -> np.ndarray:
        diffs = Z - x[None, :]                          # (N,d)
        d2 = (diffs**2) / (std[None, :]**2 + self.jitter)  # (N,d)
        return self.k(d2.T)                             # (d,N)

    # ---------- gradient of log posterior ----------
    def grad_log_post(
        self,
        x: np.ndarray,
        prior: GaussianPrior,
        y: float | np.ndarray,
        obs_idx: int | Sequence[int],
        R: float | np.ndarray
    ) -> np.ndarray:
        d = prior.dim
        inv_cov = np.linalg.inv(prior.cov)

        # prior gradient
        grad_prior = -inv_cov @ (x - prior.mu)

        # likelihood gradient for partial linear obs (select components)
        obs_idx = np.atleast_1d(obs_idx)
        y = np.atleast_1d(y)

        if np.isscalar(R):
            Rm1 = np.eye(len(obs_idx)) / float(R)
        else:
            Rm1 = np.linalg.inv(np.asarray(R))

        innovation = y - x[obs_idx]

        H = np.zeros((len(obs_idx), d))
        for i, idx in enumerate(obs_idx):
            H[i, idx] = 1.0

        grad_like = H.T @ (Rm1 @ innovation)
        return grad_prior + grad_like

    # ---------- divergence term ----------
    def divergence(
        self,
        x: np.ndarray,
        Z: np.ndarray,
        prior: GaussianPrior,
        kernel_vals: np.ndarray
    ) -> np.ndarray:
        d = prior.dim
        N = Z.shape[0]

        if self.kernel_type == "scalar":
            A = np.linalg.inv(prior.cov)
            diffs = x[None, :] - Z                      # (N,d)
            Akdiff = diffs @ A.T                        # (N,d)
            div = -(kernel_vals[:, None] * Akdiff).sum(axis=0) / (self.alpha * N)
            return div

        # matrix kernel: kernel_vals is (d,N)
        std = prior.std
        diffs = x[None, :] - Z                          # (N,d)
        denom = (std**2 * self.alpha + self.jitter)     # (d,)
        div = -np.einsum("an,na->a", kernel_vals, diffs) / denom
        div /= N
        return div

    # ---------- one flow vector f(x) ----------
    def flow_vector(
        self,
        x: np.ndarray,
        Z: np.ndarray,
        prior: GaussianPrior,
        y: float | np.ndarray,
        obs_idx: int | Sequence[int],
        R: float | np.ndarray
    ) -> np.ndarray:
        d = prior.dim
        N = Z.shape[0]

        grads = np.stack(
            [self.grad_log_post(z, prior, y, obs_idx, R) for z in Z],
            axis=1
        )  # (d,N)

        if self.kernel_type == "scalar":
            K = self.scalar_kernel_vals(x, Z, prior.cov)  # (N,)
            w = K / (K.sum() + self.jitter)
            g = grads @ w
            div = self.divergence(x, Z, prior, K)

        else:
            K = self.matrix_kernel_vals(x, Z, prior.std)  # (d,N)
            g = np.zeros(d)
            for a in range(d):
                wa = K[a] / (K[a].sum() + self.jitter)
                g[a] = grads[a] @ wa
            div = self.divergence(x, Z, prior, K)

        flow = prior.cov @ (g - div)

        # optional flow clipping for stability (if user sets prior.max_flow_norm)
        max_norm = getattr(prior, "max_flow_norm", None)
        if max_norm is not None:
            nrm = np.linalg.norm(flow)
            if nrm > max_norm:
                flow = flow * (max_norm / (nrm + 1e-12))

        return flow

    # ---------- iterate pseudo-time flow on whole particle set ----------
    def transport_particles(
        self,
        X0: np.ndarray,
        prior: GaussianPrior,
        y: float | np.ndarray,
        obs_idx: int | Sequence[int],
        R: float | np.ndarray,
        n_steps: int = 100,
        step_size: float = 0.05
    ) -> np.ndarray:
        X = X0.copy()
        N = X.shape[0]
        for _ in range(n_steps):
            X_new = X.copy()
            for i in range(N):
                X_new[i] = X[i] + step_size * self.flow_vector(
                    X[i], X, prior, y, obs_idx, R
                )
            X = X_new
        return X


# ============================================================
#  Figure 2 (Hu21): 2D collapse demo
# ============================================================

def plot_fig2_collapse(alpha: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = 2
    n = 30

    cov_A = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
    cov_B = np.array([[1.0, 0.0],
                      [0.0, 0.05]])

    mu = np.zeros(d)
    obs_idx = 1
    R = 0.2**2
    y = 1.0

    cases = [
        (cov_A, "scalar", "(a) scalar kernel, equal rates"),
        (cov_A, "matrix", "(b) matrix kernel, equal rates"),
        (cov_B, "scalar", "(c) scalar kernel, unequal rates"),
        (cov_B, "matrix", "(d) matrix kernel, unequal rates"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.subplots_adjust(hspace=0.25, wspace=0.2)

    for k, (cov, ktype, label) in enumerate(cases):
        r, c = divmod(k, 2)
        ax = axes[r, c]

        X0 = rng.multivariate_normal(mu, cov, n)
        prior = GaussianPrior(mu=mu, cov=cov)
        pff = KernelEmbeddedPFF(kernel_type=ktype, alpha=alpha)

        X = pff.transport_particles(
            X0, prior, y=y, obs_idx=obs_idx, R=R,
            n_steps=100, step_size=0.05
        )

        ax.scatter(X0[:, 0], X0[:, 1], c="gray", s=45, alpha=0.7, label="Prior")
        ax.scatter(X[:, 0],  X[:, 1],  c="red",  s=45, alpha=0.9, label="Posterior")
        ax.axhline(y=y, linestyle="--", linewidth=1.2)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(label, fontweight="bold", fontsize=12)
        if r == 1:
            ax.set_xlabel("x₁ (unobserved)")
        if c == 0:
            ax.set_ylabel("x₂ (observed)")

    plt.tight_layout()
    plt.show()


# ============================================================
#  Figure 3 (Hu21): high-dim collapse projection
# ============================================================

def plot_fig3_highdim(alpha: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = 10
    n = 40

    cov = np.eye(d)
    mu = np.zeros(d)

    X0 = rng.multivariate_normal(mu, cov, n)
    true_x = rng.standard_normal(d)

    obs_idx = 2
    R = 0.1**2
    y = true_x[obs_idx] + rng.normal(0, 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(wspace=0.2)

    for j, ktype in enumerate(["scalar", "matrix"]):
        ax = axes[j]
        prior = GaussianPrior(mu=mu, cov=cov)
        pff = KernelEmbeddedPFF(kernel_type=ktype, alpha=alpha)

        X = pff.transport_particles(
            X0, prior, y=y, obs_idx=obs_idx, R=R,
            n_steps=120, step_size=0.05
        )

        ax.scatter(X0[:, 0], X0[:, 1], c="gray", s=50, alpha=0.7, label="Prior" if j == 0 else None)
        ax.scatter(X[:, 0],  X[:, 1],  c="red",  s=50, alpha=0.9, label="Posterior" if j == 0 else None)
        ax.scatter(true_x[0], true_x[1], c="blue", marker="*", s=160, label="Truth" if j == 0 else None)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlabel("x₁")
        if j == 0:
            ax.set_ylabel("x₂")
        ax.set_title(f"({chr(ord('a') + j)}) {ktype} kernel", fontweight="bold")

        if j == 0:
            ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)
    plot_fig2_collapse(alpha=0.1, seed=0)
    plot_fig3_highdim(alpha=0.1, seed=0)

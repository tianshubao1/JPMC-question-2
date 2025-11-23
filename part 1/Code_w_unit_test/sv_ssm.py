#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Nonlinear/Non-Gaussian SSM with EKF/UKF and Particle Filter
- Problem a
- Nonlinear / Non-Gaussian SSM design with Stochastic Volatility
- Generates synthetic data and basic visualizations.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------
# 1. Parameters (defaults)
# -------------------------------
T = 500
alpha = 0.98
sigma = 0.15
beta  = 0.65
seed = 0

# -------------------------------
# 2. Synthetic data generation
# -------------------------------
def simulate_sv(T, alpha, sigma, beta, dtype=tf.float32, seed=None):
    """
    Simulate stochastic volatility model in TensorFlow.
    Returns:
        x_true: (T,) latent log-volatility
        y_obs:  (T,) observations
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    var0 = (sigma**2) / (1.0 - alpha**2)
    x0 = tf.random.normal((), mean=0.0, stddev=tf.sqrt(var0), dtype=dtype)

    x = tf.TensorArray(dtype, size=T)
    y = tf.TensorArray(dtype, size=T)

    x_prev = x0
    for t in tf.range(T):
        v_t = tf.random.normal((), dtype=dtype)
        x_t = alpha * x_prev + sigma * v_t

        w_t = tf.random.normal((), dtype=dtype)
        y_t = beta * tf.exp(0.5 * x_t) * w_t

        x = x.write(t, x_t)
        y = y.write(t, y_t)
        x_prev = x_t

    return x.stack(), y.stack()


def main():
    tf.random.set_seed(seed)
    np.random.seed(seed)

    x_true, y_obs = simulate_sv(T, alpha, sigma, beta)

    x_np = x_true.numpy()
    y_np = y_obs.numpy()

    print("Simulated SV model")
    print(f"T={T}, alpha={alpha}, sigma={sigma}, beta={beta}")
    print(f"x_true mean/std: {x_np.mean():.4f} / {x_np.std():.4f}")
    print(f"y_obs  mean/std: {y_np.mean():.4f} / {y_np.std():.4f}")

    np.savez(
        "data_sv.npz",
        x_true=x_np, y_obs=y_np,
        T=T, alpha=alpha, sigma=sigma, beta=beta, seed=seed
    )

    # plots
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(x_np, lw=1.0)
    axes[0].set_title("Latent log-volatility $x_t$")
    axes[0].set_ylabel("$x_t$")

    axes[1].plot(y_np, lw=0.8)
    axes[1].set_title("Observations $y_t$")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("$y_t$")

    plt.tight_layout()
    plt.savefig("fig_sv_latent_obs.png")
    plt.show()

    plt.figure(figsize=(11, 3.5))
    plt.plot(y_np**2, lw=0.8)
    plt.title("Squared observations $y_t^2$ (volatility proxy)")
    plt.xlabel("time")
    plt.ylabel("$y_t^2$")
    plt.tight_layout()
    plt.savefig("fig_sv_obs_sq.png")
    plt.show()


if __name__ == "__main__":
    main()

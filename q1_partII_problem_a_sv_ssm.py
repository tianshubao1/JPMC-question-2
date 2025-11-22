#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 — Nonlinear / Non-Gaussian SSM design (Stochastic Volatility model)
Generates synthetic data and basic visualizations.

Model (Doucet & Johansen 2009, Example 4):
    x1 ~ N(0, sigma^2 / (1 - alpha^2))
    x_t = alpha * x_{t-1} + sigma * v_t,     v_t ~ N(0, 1)
    y_t = beta * exp(x_t / 2) * w_t,         w_t ~ N(0, 1)
So p(y_t | x_t) = N(0, beta^2 * exp(x_t)), nonlinear in x_t.

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------
# 1. Parameters
# -------------------------------
T = 500                     # length of sequence
alpha = 0.98                # AR(1) coefficient for latent log-vol
sigma = 0.15                # process noise std
beta  = 0.65                # observation scale

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

# -------------------------------
# 2. Synthetic data generation
# -------------------------------
def simulate_sv(T, alpha, sigma, beta, dtype=tf.float32):
    """
    Simulate stochastic volatility model in TensorFlow.
    Returns:
        x_true: (T,) latent log-volatility
        y_obs:  (T,) observations
    """
    # stationary prior variance
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

x_true, y_obs = simulate_sv(T, alpha, sigma, beta)

# Convert to numpy for saving/plotting
x_np = x_true.numpy()
y_np = y_obs.numpy()

print("Simulated SV model")
print(f"T={T}, alpha={alpha}, sigma={sigma}, beta={beta}")
print(f"x_true mean/std: {x_np.mean():.4f} / {x_np.std():.4f}")
print(f"y_obs  mean/std: {y_np.mean():.4f} / {y_np.std():.4f}")

# -------------------------------
# 3. Save data for later problems
# -------------------------------
np.savez(
    "data_sv.npz",
    x_true=x_np,
    y_obs=y_np,
    T=T, alpha=alpha, sigma=sigma, beta=beta, seed=seed
)

# -------------------------------
# 4. Visualizations
# -------------------------------
# (a) Latent x_t and observation y_t
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

# (b) Squared observations
plt.figure(figsize=(11, 3.5))
plt.plot(y_np**2, lw=0.8)
plt.title("Squared observations $y_t^2$ (volatility proxy)")
plt.xlabel("time")
plt.ylabel("$y_t^2$")
plt.tight_layout()
plt.savefig("fig_sv_obs_sq.png")
plt.show()



"""
Bonus Question 1(b): HMC with Differentiable Particle Filter
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time

tfd = tfp.distributions
tfb = tfp.bijectors


# ==========================================
# 1. DIFFERENTIABLE SINKHORN RESAMPLING
# ==========================================
@tf.function
def sinkhorn_resample(particles, log_weights, epsilon=0.1, n_iters=10):
    """
    Differentiable resampling using Entropy-Regularized Optimal Transport.
    Returns: new_particles, log_det_jacobian (approx 0 for transport)
    """
    N = tf.shape(particles)[0]

    # Normalize source weights
    log_w = log_weights - tf.reduce_logsumexp(log_weights)
    a = tf.exp(log_w)
    b = tf.ones(N) / tf.cast(N, tf.float32)  # Target: Uniform

    # Cost matrix (Squared Euclidean)
    # C_ij = |x_i - x_j|^2. Since 1D: (x_i - x_j)^2
    # particles shape [N], need [N, 1] - [1, N]
    x_col = tf.expand_dims(particles, 1)
    x_row = tf.expand_dims(particles, 0)
    C = tf.square(x_col - x_row)

    # Sinkhorn Algorithm (Log-domain)
    # K = exp(-C/eps)
    K_log = -C / epsilon

    f = tf.zeros_like(a)
    g = tf.zeros_like(b)

    for _ in range(n_iters):
        # f = log(a) - logsumexp(K_log + g)
        term1 = K_log + tf.expand_dims(g, 0)  # [N, N]
        f = tf.math.log(a + 1e-9) - tf.reduce_logsumexp(term1, axis=1)

        # g = log(b) - logsumexp(K_log^T + f)
        term2 = K_log + tf.expand_dims(f, 1)
        g = tf.math.log(b + 1e-9) - tf.reduce_logsumexp(term2, axis=0)

    # Transport Matrix P = exp(f + g + K_log)
    P_log = tf.expand_dims(f, 1) + tf.expand_dims(g, 0) + K_log
    P = tf.exp(P_log)

    # Barycentric Projection: x_new = N * P^T @ x_old
    # Maps mass from old locations to uniform slots
    # shape: [N, N] @ [N, 1] -> [N, 1]
    new_particles = tf.cast(N, tf.float32) * tf.matmul(tf.transpose(P), tf.expand_dims(particles, 1))
    new_particles = tf.squeeze(new_particles)

    return new_particles


# ==========================================
# 2. DIFFERENTIABLE PF (DPF) LOG-LIKELIHOOD
# ==========================================
@tf.function
def dpf_log_likelihood(y_obs_seq, log_sigma_v, log_sigma_w, N=100):
    """
    Computes log p(y|theta) differentiably.
    Params are in log-space for unconstrained optimization/HMC.
    """
    sigma_v2 = tf.exp(log_sigma_v) ** 2
    sigma_w2 = tf.exp(log_sigma_w) ** 2
    sigma_v = tf.exp(log_sigma_v)
    sigma_w = tf.exp(log_sigma_w)

    T = tf.shape(y_obs_seq)[0]

    # Prior initialization
    particles = tf.random.normal((N,), 0.0, tf.sqrt(5.0))
    log_weights = tf.zeros(N)

    total_log_lik = 0.0

    for k in tf.range(1, T + 1):
        k_float = tf.cast(k, tf.float32)

        # 1. Prediction (Transition)
        noise = tf.random.normal((N,), 0.0, 1.0)
        # Model: 0.5x + 25x/(1+x^2) + 8cos(1.2k)
        x_pred = 0.5 * particles + \
                 25.0 * particles / (1.0 + tf.square(particles)) + \
                 8.0 * tf.cos(1.2 * k_float) + \
                 sigma_v * noise

        # 2. Update (Likelihood)
        # y = x^2/20
        y_curr = y_obs_seq[k - 1]
        mu_y = tf.square(x_pred) / 20.0

        # Gaussian Log Likelihood
        log_lik_step = -0.5 * tf.math.log(2 * np.pi * sigma_w2) - \
                       tf.square(y_curr - mu_y) / (2 * sigma_w2)

        log_weights += log_lik_step

        # 3. Accumulate Marginal Likelihood (LogSumExp)
        max_w = tf.reduce_max(log_weights)
        w_unnorm = tf.exp(log_weights - max_w)
        sum_w = tf.reduce_sum(w_unnorm)
        step_log_lik = max_w + tf.math.log(sum_w) - tf.math.log(tf.cast(N, tf.float32))

        total_log_lik += step_log_lik

        # 4. Resample (Sinkhorn / Soft)
        # Normalize weights for resampling
        norm_log_w = log_weights - (max_w + tf.math.log(sum_w))

        # Apply Sinkhorn Resampling (Differentiable!)
        # This mixes particles based on weights, preserving gradients w.r.t weights
        particles = sinkhorn_resample(x_pred, norm_log_w, epsilon=0.5, n_iters=10)

        # Reset weights
        log_weights = tf.zeros(N)

    return total_log_lik


# ==========================================
# 3. HMC SETUP
# ==========================================
def run_hmc(y_data, n_results=500, n_burnin=200):
    # Target Log Prob: Log Likelihood + Log Prior
    # Priors: sigma ~ LogNormal(0, 1) => log_sigma ~ Normal(0, 1)

    @tf.function
    def target_log_prob_fn(log_sigma_v, log_sigma_w):
        # Priors
        prior_v = -0.5 * tf.square(log_sigma_v)
        prior_w = -0.5 * tf.square(log_sigma_w)

        # Likelihood (DPF)
        # Note: Using fewer particles for HMC gradient speed usually
        lik = dpf_log_likelihood(y_data, log_sigma_v, log_sigma_w, N=50)

        return lik + prior_v + prior_w

    # Kernel: No-U-Turn Sampler (NUTS)
    kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.01
    )

    # Adaptive Step Size
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(n_burnin * 0.8)
    )

    # Initial State
    init_state = [tf.constant(0.0), tf.constant(0.0)]  # log(1.0)

    print("Running HMC with Differentiable PF Gradients...")
    start_time = time.time()

    # Run Chain
    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=n_results,
        num_burnin_steps=n_burnin,
        current_state=init_state,
        kernel=kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
    )

    dt = time.time() - start_time
    acceptance_rate = tf.reduce_mean(tf.cast(kernel_results, tf.float32))

    return states, acceptance_rate, dt


# ==========================================
# 4. MAIN & COMPARISON
# ==========================================
if __name__ == "__main__":
    # 1. Generate Synthetic Data (Same config as Part A)
    np.random.seed(42)
    T = 50
    # ... (Re-implement simple generation in numpy for consistency)
    x = np.zeros(T);
    y = np.zeros(T)
    curr = np.random.normal(0, np.sqrt(5))
    for k in range(1, T + 1):
        curr = 0.5 * curr + 25 * curr / (1 + curr ** 2) + 8 * np.cos(1.2 * k) + np.random.normal(0, 1)
        y[k - 1] = curr ** 2 / 20 + np.random.normal(0, 1)

    y_tf = tf.constant(y, dtype=tf.float32)

    # 2. Run HMC
    states, acc, runtime = run_hmc(y_tf, n_results=500, n_burnin=200)

    # Convert back to parameter space
    sigma_v2_chain = np.exp(states[0].numpy()) ** 2
    sigma_w2_chain = np.exp(states[1].numpy()) ** 2

    print(f"\nHMC Done. Runtime: {runtime:.2f}s")
    print(f"Acceptance Rate: {acc:.2f}")
    print(f"Posterior Means: sigma_v2={sigma_v2_chain.mean():.3f}, sigma_w2={sigma_w2_chain.mean():.3f}")

    # 3. Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(sigma_v2_chain);
    ax[0].set_title(r'HMC $\sigma_v^2$')
    ax[1].plot(sigma_w2_chain);
    ax[1].set_title(r'HMC $\sigma_w^2$')
    plt.tight_layout()
    plt.savefig("bonus_hmc_trace.png")
    plt.show()
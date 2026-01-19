"""
Bonus Question 1(a): PMMH with Invertible PF-PF (Li 2017)
"""

import numpy as np
import matplotlib.pyplot as plt
import time


# ==========================================
# 1. MODEL DYNAMICS (Andrieu et al. 2010)
# ==========================================
def transition_fn(x_prev, k):
    """x_k = 0.5*x + 25*x/(1+x^2) + 8*cos(1.2*k)"""
    return (0.5 * x_prev +
            25.0 * x_prev / (1.0 + x_prev ** 2) +
            8.0 * np.cos(1.2 * k))


def observation_fn(x):
    """y_k = x^2 / 20"""
    return x ** 2 / 20.0


def dhdx(x):
    """Derivative of h(x) w.r.t x: h'(x) = x / 10"""
    return x / 10.0


# ==========================================
# 2. INVERTIBLE LEDH FLOW (Li 2017)
# ==========================================
def ledh_flow_step(x, y_obs, sigma_w2, P_pred, n_steps=10):
    """
    Apply LEDH flow to migrate particles from Prior to Posterior.
    Returns: x_new, log_det_jacobian
    """
    curr_x = x.copy()
    log_detJ = 0.0
    epsilon = 1.0 / n_steps

    # Pre-calculate R inverse
    R_inv = 1.0 / sigma_w2

    for _ in range(n_steps):
        # Linearization H = dh/dx
        H = dhdx(curr_x)

        # S = H P H^T + R
        S = H * P_pred * H + sigma_w2

        # Flow Parameters (Li 17 LEDH simplification for 1D)
        # A = -0.5 P H^T R^-1 H (Wait, standard flow derivation usually:)
        # drift = A x + b

        # Using the exact form provided in your snippet which works for 1D:
        # A = -0.5 * P * H^2 / S
        # b = (P * H / S) * (y - h(x) + H*x)

        A = -0.5 * (P_pred * H ** 2) / S
        innovation = y_obs - observation_fn(curr_x) + H * curr_x
        b = (P_pred * H / S) * innovation

        # Update
        curr_x = curr_x + epsilon * (A * curr_x + b)

        # Accumulate Jacobian Determinant
        # Jacobian of (x + eps(Ax+b)) is (1 + eps*A) roughly
        det_step = 1.0 + epsilon * A
        log_detJ += np.log(np.abs(det_step))

    return curr_x, log_detJ


# ==========================================
# 3. PF-PF LIKELIHOOD ESTIMATOR
# ==========================================
def pf_pf_likelihood(y_seq, theta_v, theta_w, N=100):
    """
    Returns log p(y_{1:T} | theta) using PF-PF.
    theta_v: sigma_v^2
    theta_w: sigma_w^2
    """
    T = len(y_seq)

    # Init particles (Prior N(0, 5))
    x = np.random.normal(0, np.sqrt(5), N)
    log_w = np.zeros(N)  # Uniform weights initially

    total_log_likelihood = 0.0

    for k in range(1, T + 1):
        # 1. Proposal (Prediction)
        # x_pred ~ p(x_k | x_{k-1})
        means = transition_fn(x, k)
        x_pred = means + np.random.normal(0, np.sqrt(theta_v), N)

        # 2. Flow (Migration)
        P_empirical = np.var(x_pred) + 1e-6
        x_flow = np.zeros(N)
        log_detJ = np.zeros(N)

        # Apply flow to each particle (Vectorizable, but loop for clarity/safety)
        # Note: In production code, vectorizing ledh_flow_step is much faster
        for i in range(N):
            x_flow[i], log_detJ[i] = ledh_flow_step(
                x_pred[i], y_seq[k - 1], theta_w, P_empirical
            )

        # 3. Weighting (Li 17 Invertible Update)
        # log w = log p(y|x) + log p(x|x_prev) - log q(x|x_prev) + log_detJ
        # q(x) is defined by the flow mapping x = T(x_pred).
        # By change of variables: q(x_flow) = p(x_pred) / |J|
        # So weight reduces to: log p(y|x_flow) + log p(x_flow|prev) - (log p(x_pred|prev) - log|J|)
        # Which simplifies to likelihood * ratio of transitions * J

        # Likelihood p(y|x_flow)
        log_lik = -0.5 * np.log(2 * np.pi * theta_w) - \
                  (y_seq[k - 1] - observation_fn(x_flow)) ** 2 / (2 * theta_w)

        # Transition p(x_flow | x_old)
        mean_k = transition_fn(x, k)  # Re-eval mean at old particles
        log_trans_flow = -0.5 * np.log(2 * np.pi * theta_v) - \
                         (x_flow - mean_k) ** 2 / (2 * theta_v)

        # Proposal p(x_pred | x_old) (Original predicted state)
        log_trans_pred = -0.5 * np.log(2 * np.pi * theta_v) - \
                         (x_pred - mean_k) ** 2 / (2 * theta_v)

        # Incremental weight
        # Note: log_w carries over previous weights
        log_inc = log_lik + log_trans_flow - log_trans_pred - log_detJ
        log_w = log_w + log_inc

        # 4. Normalize & Accumulate Likelihood
        max_log = np.max(log_w)
        w_unnorm = np.exp(log_w - max_log)
        sum_w = np.sum(w_unnorm)
        norm_w = w_unnorm / sum_w

        # p(y_k | y_{1:k-1}) approx sum(w_unnorm)/N
        # (Standard PF likelihood accumulation)
        total_log_likelihood += max_log + np.log(sum_w) - np.log(N)

        # 5. Resample (Systematic)
        indices = systematic_resample(norm_w)
        x = x_flow[indices]
        log_w = np.zeros(N)  # Reset weights

    return total_log_likelihood


def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


# ==========================================
# 4. PMMH ALGORITHM
# ==========================================
def run_pmmh(y_obs, n_iter=1000, N_part=100):
    # Initial Params (sigma_v^2, sigma_w^2)
    current_params = np.array([1.0, 1.0])  # Start at truth

    # Initial Likelihood
    current_ll = pf_pf_likelihood(y_obs, current_params[0], current_params[1], N_part)

    chain = []
    accept_count = 0

    start_time = time.time()

    for i in range(n_iter):
        # Propose new params (Log-Random Walk)
        prop_params = current_params * np.exp(np.random.normal(0, 0.2, 2))

        # Check priors (flat positive)
        if np.any(prop_params <= 0):
            chain.append(current_params)
            continue

        # Compute Likelihood
        prop_ll = pf_pf_likelihood(y_obs, prop_params[0], prop_params[1], N_part)

        # MH Ratio (Prior is flat, so just likelihood ratio + Jacobian of log-transform)
        # Jacobian adjustment for log-proposal: prop/curr (cancels out if symmetric in log space)
        # Standard MH ratio:
        ratio = prop_ll - current_ll

        if np.log(np.random.rand()) < ratio:
            current_params = prop_params
            current_ll = prop_ll
            accept_count += 1

        chain.append(current_params)

        if i % 100 == 0:
            print(f"Iter {i}/{n_iter} | Acc: {accept_count / (i + 1):.2f} | Params: {current_params}")

    dt = time.time() - start_time
    return np.array(chain), accept_count / n_iter, dt


# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Generate Data
    T = 50
    x_true = np.zeros(T)
    y_obs = np.zeros(T)
    x = np.random.normal(0, np.sqrt(5))
    true_params = [1.0, 1.0]  # sigma_v2, sigma_w2

    for k in range(1, T + 1):
        x = transition_fn(x, k) + np.random.normal(0, 1.0)
        y = x ** 2 / 20 + np.random.normal(0, 1.0)
        x_true[k - 1] = x
        y_obs[k - 1] = y

    print("Running PMMH with Li(17) Flow...")
    chain, acc, runtime = run_pmmh(y_obs, n_iter=1000, N_part=100)

    print(f"\nDone. Runtime: {runtime:.2f}s")
    print(f"Acceptance Rate: {acc:.2f}")
    print(f"Posterior Means: sigma_v2={chain[:, 0].mean():.3f}, sigma_w2={chain[:, 1].mean():.3f}")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(chain[:, 0]);
    ax[0].axhline(1.0, c='r');
    ax[0].set_title(r'$\sigma_v^2$')
    ax[1].plot(chain[:, 1]);
    ax[1].axhline(1.0, c='r');
    ax[1].set_title(r'$\sigma_w^2$')
    plt.tight_layout()
    plt.savefig("bonus_pmmh_trace.png")
    plt.show()
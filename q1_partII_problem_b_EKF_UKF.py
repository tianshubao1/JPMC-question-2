
"""
Q2 Problem 2 — EKF and UKF for the Stochastic Volatility (SV) SSM
Runnable script (no argparse). Loads data from Problem 1 (data_sv.npz), runs:
  (A) EKF via log-squared observation approximation (standard in SV)
  (B) UKF on the same transformed observation z_t (scaled sigma points)
  (C) Naive UKF directly on y_t (additive-Gaussian assumption) to illustrate failure

SV model:
    x_t = alpha x_{t-1} + sigma v_t,   v_t ~ N(0,1)
    y_t = beta exp(x_t/2) w_t,         w_t ~ N(0,1)
p(y_t | x_t) = N(0, beta^2 exp(x_t)).

Because y_t is multiplicative and zero-mean conditional on x_t, a moment-matching UKF
applied directly to y_t yields near-zero cross-covariance and thus near-zero gain.
We show this as a "sigma-point failure" case.

For EKF/UKF we therefore use the standard transformed pseudo-measurement:
    z_t = log(y_t^2 + eps)
z_t = log(beta^2) + x_t + log(w_t^2),
log(w_t^2) is log-chi-square(1). Approximate it Gaussian:
    log(w_t^2) ≈ N(mu_eta, var_eta),
mu_eta = psi(1/2) + log 2 ≈ -1.270362845
var_eta = trigamma(1/2) = pi^2/2 ≈ 4.934802200

So:
    z_t ≈ h_z(x_t) + e_t,  h_z(x)=x+c,  e_t~N(0,Rz),
c = log(beta^2) + mu_eta,  Rz=var_eta.

Outputs:
  - ekf_ukf_results.npz
  - fig_ekf_ukf_vs_true.pdf
  - fig_naive_ukf_fail.pdf
"""

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. Load dataset from Problem 1
# ------------------------------------------------------------
try:
    data = np.load("data_sv.npz")
    x_true = data["x_true"].astype(np.float32)
    y_obs  = data["y_obs"].astype(np.float32)
    T      = int(data["T"])
    alpha  = float(data["alpha"])
    sigma  = float(data["sigma"])
    beta   = float(data["beta"])
    seed   = int(data["seed"])
    print("Loaded data_sv.npz")
except FileNotFoundError:
    raise FileNotFoundError(
        "data_sv.npz not found. Run q2_problem1_sv_ssm.py first."
    )

tf.random.set_seed(seed)
np.random.seed(seed)

# ------------------------------------------------------------
# 1. Transformed pseudo-observation z_t
# ------------------------------------------------------------
eps = 1e-6
z_obs = np.log(y_obs**2 + eps).astype(np.float32)

mu_eta  = -1.270362845
var_eta = 4.934802200

c  = np.log(beta**2) + mu_eta
Rz = var_eta

# stationary prior
P0 = (sigma**2) / (1.0 - alpha**2)
m0 = 0.0

# ------------------------------------------------------------
# 2A. EKF (reduces to KF on z_t approximation)
# ------------------------------------------------------------
def run_ekf_z(z_obs, T, alpha, sigma, c, Rz, m0, P0):
    m_f = np.zeros(T, np.float32)
    P_f = np.zeros(T, np.float32)

    m_prev, P_prev = m0, P0
    for t in range(T):
        # predict
        m_pred = alpha * m_prev
        P_pred = alpha**2 * P_prev + sigma**2

        # update with z = x + c + e
        r = z_obs[t] - (m_pred + c)
        S = P_pred + Rz
        K = P_pred / S

        m_post = m_pred + K * r
        P_post = (1 - K)**2 * P_pred + (K**2) * Rz   # Joseph scalar

        m_f[t], P_f[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post

    return m_f, P_f

t0 = time.perf_counter()
m_ekf, P_ekf = run_ekf_z(z_obs, T, alpha, sigma, c, Rz, m0, P0)
rt_ekf = time.perf_counter() - t0

# ------------------------------------------------------------
# 2B. Scaled-UKF on z_t
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

def run_ukf_z(z_obs, T, alpha, sigma, c, Rz, m0, P0,
             alpha_ut=1e-3, beta_ut=2.0, kappa_ut=0.0):
    m_f = np.zeros(T, np.float32)
    P_f = np.zeros(T, np.float32)

    m_prev, P_prev = m0, P0
    for t in range(T):
        # ---- predict (linear, so exact) ----
        m_pred = alpha * m_prev
        P_pred = alpha**2 * P_prev + sigma**2

        # sigma points for x ~ N(m_pred, P_pred)
        X, Wm, Wc = sigma_points_1d(m_pred, P_pred, alpha_ut, beta_ut, kappa_ut)

        # propagate through measurement h_z(x)=x+c  (still do UT for formality)
        Z = X + c

        z_hat = np.sum(Wm * Z)
        P_zz = np.sum(Wc * (Z - z_hat)**2) + Rz
        P_xz = np.sum(Wc * (X - m_pred) * (Z - z_hat))

        # gain & update
        K = P_xz / P_zz
        r = z_obs[t] - z_hat

        m_post = m_pred + K * r
        P_post = P_pred - K**2 * P_zz   # equivalent to Joseph here

        m_f[t], P_f[t] = m_post, P_post
        m_prev, P_prev = m_post, P_post

    return m_f, P_f

t0 = time.perf_counter()
m_ukf, P_ukf = run_ukf_z(z_obs, T, alpha, sigma, c, Rz, m0, P0)
rt_ukf = time.perf_counter() - t0

# ------------------------------------------------------------
# 2C. Naive UKF directly on y_t (additive-Gaussian assumption) — expected failure
# ------------------------------------------------------------
def run_naive_ukf_y(y_obs, T, alpha, sigma, beta, m0, P0,
                    Ry=1.0, alpha_ut=1e-3, beta_ut=2.0, kappa_ut=0.0):
    """
    Treat y_t ≈ h_y(x_t) + e_t, e_t ~ N(0, Ry), with h_y(x)=0?
    If we (incorrectly) set h_y(x)=0 (conditional mean), UT gives zero cross-cov.
    This is included to illustrate sigma-point/moment failure on multiplicative noise.
    """
    m_f = np.zeros(T, np.float32)
    P_f = np.zeros(T, np.float32)
    K_hist = np.zeros(T, np.float32)

    m_prev, P_prev = m0, P0
    for t in range(T):
        m_pred = alpha * m_prev
        P_pred = alpha**2 * P_prev + sigma**2

        X, Wm, Wc = sigma_points_1d(m_pred, P_pred, alpha_ut, beta_ut, kappa_ut)

        # naive additive model with conditional mean 0
        Z = np.zeros_like(X)  # h_y(x)=0 for all sigma points

        z_hat = np.sum(Wm * Z)  # ~0
        P_zz = np.sum(Wc * (Z - z_hat)**2) + Ry  # ~Ry
        P_xz = np.sum(Wc * (X - m_pred) * (Z - z_hat))  # ~0

        K = P_xz / P_zz  # ~0
        r = y_obs[t] - z_hat

        m_post = m_pred + K * r  # ~m_pred (no update)
        P_post = P_pred - K**2 * P_zz

        m_f[t], P_f[t] = m_post, P_post
        K_hist[t] = K
        m_prev, P_prev = m_post, P_post

    return m_f, P_f, K_hist

Ry_naive = np.var(y_obs)  # arbitrary additive noise scale
m_ukf_naive, P_ukf_naive, K_naive = run_naive_ukf_y(
    y_obs, T, alpha, sigma, beta, m0, P0, Ry=Ry_naive
)

# ------------------------------------------------------------
# 3. Metrics
# ------------------------------------------------------------
def metrics(m_est, P_est, name):
    rmse = float(np.sqrt(np.mean((m_est - x_true)**2)))
    mae  = float(np.mean(np.abs(m_est - x_true)))
    avgP = float(np.mean(P_est))
    print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, avgP={avgP:.4f}")
    return rmse, mae, avgP

rmse_ekf, mae_ekf, avgP_ekf = metrics(m_ekf, P_ekf, "EKF(z)")
rmse_ukf, mae_ukf, avgP_ukf = metrics(m_ukf, P_ukf, "UKF(z)")
rmse_nu,  mae_nu,  avgP_nu  = metrics(m_ukf_naive, P_ukf_naive, "Naive UKF(y)")

print(f"Runtime EKF(z): {rt_ekf*1e3:.2f} ms")
print(f"Runtime UKF(z): {rt_ukf*1e3:.2f} ms")
print("(Naive UKF uses same cost as UKF z; omitted.)")

# ------------------------------------------------------------
# 4. Save results
# ------------------------------------------------------------
np.savez(
    "ekf_ukf_results.npz",
    m_ekf=m_ekf, P_ekf=P_ekf, runtime_ekf=rt_ekf,
    m_ukf=m_ukf, P_ukf=P_ukf, runtime_ukf=rt_ukf,
    m_ukf_naive=m_ukf_naive, P_ukf_naive=P_ukf_naive, K_naive=K_naive,
    z_obs=z_obs, eps=eps, mu_eta=mu_eta, var_eta=var_eta, c=c, Rz=Rz,
    rmse_ekf=rmse_ekf, mae_ekf=mae_ekf, avgP_ekf=avgP_ekf,
    rmse_ukf=rmse_ukf, mae_ukf=mae_ukf, avgP_ukf=avgP_ukf,
    rmse_nu=rmse_nu, mae_nu=mae_nu, avgP_nu=avgP_nu,
    Ry_naive=Ry_naive
)

# ------------------------------------------------------------
# 5. Plots
# ------------------------------------------------------------
tgrid = np.arange(T)

# EKF vs UKF (z)
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

# Naive UKF failure
plt.figure(figsize=(11, 4))
plt.plot(tgrid, x_true, label="true $x_t$", lw=1.0)
plt.plot(tgrid, m_ukf_naive, label="Naive UKF on $y_t$", lw=1.0)
plt.title("Naive UKF on raw $y_t$ (additive-Gaussian assumption) — little/no update")
plt.xlabel("time")
plt.ylabel("$x_t$")
plt.legend()
plt.tight_layout()
plt.savefig("fig_naive_ukf_fail.png")
plt.show()
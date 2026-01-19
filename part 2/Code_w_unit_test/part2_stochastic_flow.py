"""
File: part2_stochastic_flow.py
Replicates main results of Dai & Daum (2021) "Stiffness Mitigation".
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. UTILITIES
# ==========================================
def sym(A):
    return 0.5 * (A + A.T)


def safe_inv_2x2(A, eps=1e-12):
    return np.linalg.inv(A + eps * np.eye(2))


def trace(A):
    return float(np.trace(A))


def cov2d(X):
    """Unbiased sample covariance for Nx2 array"""
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


# ==========================================
# 2. PROBLEM SETUP (Section 4)
# ==========================================
SENSORS = np.array([[3.5, 0.0], [-3.5, 0.0]])
X_TRUE = np.array([4.0, 4.0])

# Prior N(m0, P0)
M0 = np.array([3.0, 5.0])
P0 = np.array([[1000.0, 0.0], [0.0, 2.0]])
P0_INV = safe_inv_2x2(P0)

# Measurement noise
R = np.array([[0.04, 0.0], [0.0, 0.04]])
R_INV = safe_inv_2x2(R)

Z_OBS = np.array([0.4754, 1.1868])

# Stochastic Flow Parameters
MU = 0.2
Q = np.array([[4.0, 0.0], [0.0, 0.4]])
Q_CHOL = np.linalg.cholesky(Q)

# Hessian of Prior (Constant)
HESS_LOG_P0 = -P0_INV


# ==========================================
# 3. MEASUREMENT MODEL
# ==========================================
def h_and_jac_and_hess(x):
    """Computes h(x), Jacobian, and Hessian list."""
    xx, yy = float(x[0]), float(x[1])
    h_val = np.zeros(2)
    J = np.zeros((2, 2))
    H_list = []

    for i, (xi, yi) in enumerate(SENSORS):
        dx, dy = xx - xi, yy - yi
        r2 = dx * dx + dy * dy
        r4 = r2 * r2

        h_val[i] = np.arctan2(dy, dx)
        J[i] = [-dy / r2, dx / r2]

        Hxx = 2.0 * dx * dy / r4
        Hxy = -(dx * dx - dy * dy) / r4
        Hyx = (dy * dy - dx * dx) / r4
        Hyy = -2.0 * dx * dy / r4
        H_list.append(np.array([[Hxx, Hxy], [Hyx, Hyy]]))

    return h_val, J, H_list


def grad_hess_logL(x):
    h_val, J, H_list = h_and_jac_and_hess(x)
    e = Z_OBS - h_val
    r = R_INV @ e
    g = J.T @ r
    H = -J.T @ R_INV @ J
    for i in range(2):
        H += r[i] * H_list[i]
    return g, sym(H)


# --- Reference Hessian for BVP ---
def compute_linearized_hessian(x_ref):
    _, J, _ = h_and_jac_and_hess(x_ref)
    return -sym(J.T @ R_INV @ J)


# Global Constant Matrix used by stiffness_ratio and BVP
HESS_L0_REF = compute_linearized_hessian(M0)


# ==========================================
# 4. OPTIMAL HOMOTOPY (BVP)
# ==========================================
def beta_dd(beta):
    # Eq 28: M = - (H_p0 + beta * H_L)
    M = -(HESS_LOG_P0 + beta * HESS_L0_REF)
    Minv = safe_inv_2x2(M, eps=1e-10)
    M2inv = Minv @ Minv

    # beta'' = -mu * [tr(H_L)tr(M^-1) + tr(M)tr(M^-2 H_L)]
    term = trace(HESS_L0_REF) * trace(Minv) + trace(M) * trace(M2inv @ HESS_L0_REF)
    return -MU * term


def integrate_beta_rk4(beta0, v0, grid):
    beta, v = float(beta0), float(v0)
    betas = np.zeros(len(grid))
    vs = np.zeros(len(grid))
    betas[0], vs[0] = beta, v

    for k in range(len(grid) - 1):
        dt = float(grid[k + 1] - grid[k])

        def f(state):
            b, vv = state
            return np.array([vv, beta_dd(b)])

        y0 = np.array([beta, v])
        k1 = f(y0)
        k2 = f(y0 + 0.5 * dt * k1)
        k3 = f(y0 + 0.5 * dt * k2)
        k4 = f(y0 + dt * k3)
        y1 = y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        beta, v = y1[0], y1[1]
        betas[k + 1], vs[k + 1] = beta, v
    return betas, vs


def solve_beta_star_shooting(n_grid=801, max_expand=30):
    grid = np.linspace(0.0, 1.0, n_grid)

    def terminal_residual(v0):
        betas, _ = integrate_beta_rk4(0.0, v0, grid)
        return betas[-1] - 1.0

    # Bracket
    vL, vU = -10.0, 10.0
    fL, fU = terminal_residual(vL), terminal_residual(vU)

    expand = 0
    while fL * fU > 0 and expand < max_expand:
        vL *= 2.0;
        vU *= 2.0
        fL, fU = terminal_residual(vL), terminal_residual(vU)
        expand += 1

    if fL * fU > 0:
        raise RuntimeError("Failed to bracket v0.")

    # Bisection
    for _ in range(50):
        vM = 0.5 * (vL + vU)
        fM = terminal_residual(vM)
        if fL * fM <= 0:
            vU, fU = vM, fM
        else:
            vL, fL = vM, fM

    v0_star = 0.5 * (vL + vU)
    beta_star, beta_dot_star = integrate_beta_rk4(0.0, v0_star, grid)
    return grid, beta_star, beta_dot_star


# ==========================================
# 5. METRICS & SDE
# ==========================================
def stiffness_ratio(beta, beta_dot):
    # S = Hess_log_p0 + beta * Hess_logL_ref
    S = HESS_LOG_P0 + beta * HESS_L0_REF
    Sinv = safe_inv_2x2(S, eps=1e-10)

    # F = 0.5 Q S - 0.5 beta_dot S^-1 Hess_L
    F = 0.5 * (Q @ S) - 0.5 * beta_dot * (Sinv @ HESS_L0_REF)

    ev = np.linalg.eigvals(F)
    re = np.abs(np.real(ev))
    re = re[re > 1e-14]
    if re.size == 0: return np.inf
    return float(re.max() / re.min())


def drift_f(x, beta, beta_dot):
    g0 = -P0_INV @ (x - M0)
    gh, Hh = grad_hess_logL(x)  # Exact nonlinear hessian

    S = HESS_LOG_P0 + beta * Hh
    Sinv = safe_inv_2x2(S, eps=1e-10)

    gp = g0 + beta * gh

    # Drift
    term1 = Sinv @ Hh @ Sinv
    K1 = 0.5 * Q + 0.5 * beta_dot * term1
    K2 = -beta_dot * Sinv

    return K1 @ gp + K2 @ gh


def propagate_particles(X0, beta_vals, betadot_vals, dlam, xi):
    X = X0.copy()
    sq = np.sqrt(dlam)
    K = len(beta_vals)
    N = X.shape[0]

    for k in range(K):
        b, bd = float(beta_vals[k]), float(betadot_vals[k])
        F = np.zeros_like(X)
        for i in range(N):
            F[i] = drift_f(X[i], b, bd)
        X = X + F * dlam + (xi[k] @ Q_CHOL.T) * sq
    return X


# ==========================================
# 6. MAIN & VISUALIZATION
# ==========================================
def main():
    print("Running Part 2(1)(a) Replication...")
    os.makedirs("figures", exist_ok=True)

    # 1. BVP
    grid, beta_star, beta_dot_star = solve_beta_star_shooting(n_grid=801)

    # 2. Stiffness Curves
    lam_plot = np.linspace(0, 1, 401)
    # Stiffness Baseline (beta=lambda, beta_dot=1)
    R_base = np.array([stiffness_ratio(l, 1.0) for l in lam_plot])

    # Stiffness Optimal
    beta_star_plot = np.interp(lam_plot, grid, beta_star)
    beta_dot_star_plot = np.interp(lam_plot, grid, beta_dot_star)
    R_opt = np.array([stiffness_ratio(b, bd) for b, bd in zip(beta_star_plot, beta_dot_star_plot)])

    # 3. MC Simulation
    K = 200;
    N = 50;
    MC = 20
    lam_mid = (np.arange(K) + 0.5) / K
    dlam = 1.0 / K

    # Baseline Schedule
    beta_base = lam_mid
    betadot_base = np.ones(K)

    # Optimal Schedule (Interpolated)
    beta_opt = np.interp(lam_mid, grid, beta_star)
    betadot_opt = np.interp(lam_mid, grid, beta_dot_star)

    rows = []
    seed0 = 12345
    for mc in range(1, MC + 1):
        rng = np.random.default_rng(seed0 + mc)
        X0 = rng.multivariate_normal(mean=M0, cov=P0, size=N)
        xi = rng.standard_normal(size=(K, N, 2))  # CRN

        Xb = propagate_particles(X0, beta_base, betadot_base, dlam, xi)
        Xo = propagate_particles(X0, beta_opt, betadot_opt, dlam, xi)

        mb, mo = Xb.mean(0), Xo.mean(0)
        mse_b = float(np.sum((mb - X_TRUE) ** 2))
        mse_o = float(np.sum((mo - X_TRUE) ** 2))
        trPb = trace(cov2d(Xb))
        trPo = trace(cov2d(Xo))
        rows.append([mc, mse_b, mse_o, trPb, trPo])

    rows = np.array(rows)
    avg = rows[:, 1:].mean(axis=0)
    print("\nTable 1 Results (Avg):")
    print(f"MSE Base: {avg[0]:.4f}, MSE Opt: {avg[1]:.4f}")
    print(f"TrP Base: {avg[2]:.4f}, TrP Opt: {avg[3]:.4f}")

    # 4. PLOTS (Figure 2 Style)
    fig = plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(grid, beta_star, 'b-', label=r'$\beta^*(\lambda)$')
    ax1.plot(grid, grid, 'k--', label=r'$\beta=\lambda$')
    ax1.set_xlabel(r'$\lambda$');
    ax1.set_ylabel(r'$\beta$')
    ax1.legend();
    ax1.grid(True)
    ax1.set_title("Optimal Schedule")

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(grid, beta_star - grid, 'r-')
    ax2.set_xlabel(r'$\lambda$');
    ax2.set_ylabel(r'$\beta^*(\lambda) - \lambda$')
    ax2.grid(True);
    ax2.set_title("Schedule Deviation")

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(grid, beta_dot_star, 'g-')
    ax3.set_xlabel(r'$\lambda$');
    ax3.set_ylabel(r'$d\beta^*/d\lambda$')
    ax3.grid(True);
    ax3.set_title("Optimal Slope")

    ax4 = plt.subplot(2, 2, 4)
    ax4.semilogy(lam_plot, R_base, 'k--', label='Baseline')
    ax4.semilogy(lam_plot, R_opt, 'b-', label='Optimal')
    ax4.set_xlabel(r'$\lambda$');
    ax4.set_ylabel('Stiffness Ratio (log)')
    ax4.legend();
    ax4.grid(True)
    ax4.set_title("Stiffness Ratio")

    plt.tight_layout()
    plt.savefig("fig_replicated_fig2.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
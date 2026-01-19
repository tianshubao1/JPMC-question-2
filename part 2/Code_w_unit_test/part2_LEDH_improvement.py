"""
Problem 1(b): Impact of Dai(22) Optimal Schedule on Li(17) Acoustic Tracking Filter.

Experiment:
1. Simulate Li(17) Acoustic Tracking scenario (Single Target).
2. Implement a Particle Flow Particle Filter (PF-PF).
3. Compare two flow schedules:
    - Baseline: Linear schedule (beta = lambda)
    - Optimal:  Stiffness-mitigating schedule (beta*) from Dai(22) BVP.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.linalg import inv, cholesky


# Reuse config from your provided code (simplified for single target tracking)
@dataclass
class AcousticTrackingConfig:
    C: int = 1  # Tracking 1 target for clear comparison
    T: int = 20  # Time steps
    dt: float = 1.0
    Psi: float = 10.0
    d0: float = 0.1
    sigma_w: float = np.sqrt(0.01)  # Measurement noise std

    # Process noise
    q_proc: float = 0.1

    # Particle Flow settings
    N_particles: int = 100
    K_flow_steps: int = 50  # Steps in the flow (lambda 0->1)

    # Sensors (Grid)
    sensor_pos: np.ndarray = None

    def __post_init__(self):
        if self.sensor_pos is None:
            # 4 Sensors in a grid (0,0) to (40,40) roughly
            self.sensor_pos = np.array([
                [5.0, 5.0], [5.0, 35.0],
                [35.0, 5.0], [35.0, 35.0]
            ])

        # Constant Velocity Model
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Process Noise Covariance Q
        # Continuous white noise discretization
        q = self.q_proc
        dt = self.dt
        self.Q = q * np.array([
            [dt ** 3 / 3, 0, dt ** 2 / 2, 0],
            [0, dt ** 3 / 3, 0, dt ** 2 / 2],
            [dt ** 2 / 2, 0, dt, 0],
            [0, dt ** 2 / 2, 0, dt]
        ])


# ==========================================
# 1. ACOUSTIC MEASUREMENT MODELS & DERIVATIVES
# ==========================================
def acoustic_h(x_state, sensors, Psi, d0):
    """
    Measurement function h(x).
    x_state: [x, y, vx, vy]
    Returns: [Ns] signal amplitudes
    """
    rx, ry = x_state[0], x_state[1]
    z_pred = []
    for sx, sy in sensors:
        dist2 = (rx - sx) ** 2 + (ry - sy) ** 2
        amp = Psi / (dist2 + d0)
        z_pred.append(amp)
    return np.array(z_pred)


def acoustic_grad_hess(x_state, sensors, Psi, d0):
    """
    Computes Gradient (g) and Hessian (H) of h(x) w.r.t position [x,y].
    Used for the Flow Drift.
    """
    rx, ry = x_state[0], x_state[1]
    Ns = len(sensors)

    # We only care about position gradients for the flow of [x,y]
    # Velocity is indirectly affected via correlations in P,
    # but strictly h depends only on pos.

    J = np.zeros((Ns, 4))  # Jacobian [dh/dx, dh/dy, 0, 0]
    H_list = []  # List of Hessians for each sensor

    for i, (sx, sy) in enumerate(sensors):
        dx = rx - sx
        dy = ry - sy
        dist2 = dx ** 2 + dy ** 2
        denom = dist2 + d0
        denom2 = denom ** 2
        denom3 = denom ** 3

        # h = Psi / (dist2 + d0)
        val = Psi / denom

        # First derivatives
        # dh/dx = -Psi * (denom)^-2 * 2(x-sx)
        dh_dx = -Psi * (1.0 / denom2) * 2 * dx
        dh_dy = -Psi * (1.0 / denom2) * 2 * dy

        J[i, 0] = dh_dx
        J[i, 1] = dh_dy

        # Second derivatives (Hessian of h_i)
        # d/dx (dh_dx)
        # = -2Psi * [ (denom^-2) + (x-sx)*(-2 denom^-3 * 2(x-sx)) ]
        # = -2Psi * [ 1/D^2 - 4(x-sx)^2/D^3 ]
        d2h_dx2 = -2 * Psi * (1.0 / denom2 - 4 * dx ** 2 / denom3)
        d2h_dy2 = -2 * Psi * (1.0 / denom2 - 4 * dy ** 2 / denom3)
        d2h_dxdy = -2 * Psi * (0.0 - 4 * dx * dy / denom3)  # Chain rule on denom

        H_i = np.zeros((4, 4))
        H_i[0, 0] = d2h_dx2
        H_i[1, 1] = d2h_dy2
        H_i[0, 1] = d2h_dxdy
        H_i[1, 0] = d2h_dxdy
        H_list.append(H_i)

    return J, H_list


# ==========================================
# 2. DAI(22) OPTIMAL SCHEDULE SOLVER
# ==========================================
def solve_optimal_schedule(cfg):
    """
    Solves the BVP for beta*(lambda) using a representative geometry.
    Approximation: We compute the stiffness metric at the center of the grid.
    """
    # 1. Setup Representative State (Center of field)
    x_rep = np.array([20.0, 20.0, 0.0, 0.0])

    # 2. Compute Representative Hessians
    # Hess_log_prior (P0 inverse approximation)
    # We assume steady state covariance approx for P0
    P_rep = np.eye(4) * 5.0
    H_prior = -inv(P_rep)

    # Hess_log_likelihood (Expected Information)
    # H_lik = - J^T R^-1 J (Fisher Information approximation)
    J, _ = acoustic_grad_hess(x_rep, cfg.sensor_pos, cfg.Psi, cfg.d0)
    R_inv = np.eye(len(cfg.sensor_pos)) * (1.0 / cfg.sigma_w ** 2)
    H_lik = - J.T @ R_inv @ J

    # 3. Define BVP ODE (Dai Eq 28)
    mu = 0.2  # Regularization weight

    def beta_dd(beta):
        M = -(H_prior + beta * H_lik)
        # Add nugget for stability
        M = M + np.eye(4) * 1e-6
        Minv = inv(M)
        M2inv = Minv @ Minv

        # Metric trace terms
        # tr(H_L) * tr(M^-1) + tr(M) * tr(M^-2 H_L)
        term = np.trace(H_lik) * np.trace(Minv) + np.trace(M) * np.trace(M2inv @ H_lik)
        return -mu * term

    # 4. Shooting Method
    n_grid = cfg.K_flow_steps
    lam_grid = np.linspace(0, 1, n_grid)

    # Simple integration helper
    def integrate(v0):
        b, v = 0.0, v0
        traj = [0.0]
        dt = 1.0 / (n_grid - 1)
        for _ in range(n_grid - 1):
            # Euler integration is sufficient for the shape
            acc = beta_dd(b)
            b += v * dt
            v += acc * dt
            traj.append(b)
        return np.array(traj)

    # Bisection to find v0 such that beta(1) = 1
    v_low, v_high = 0.0, 50.0  # Heuristic range
    best_traj = lam_grid  # Fallback linear

    for _ in range(20):
        v_mid = (v_low + v_high) / 2
        traj = integrate(v_mid)
        end_val = traj[-1]

        if abs(end_val - 1.0) < 0.01:
            best_traj = traj
            break
        elif end_val < 1.0:
            v_low = v_mid
        else:
            v_high = v_mid

    # Normalize exactly to [0,1] to handle integration drift
    best_traj = (best_traj - best_traj[0]) / (best_traj[-1] - best_traj[0])

    # Compute derivative (beta_dot)
    beta_dot = np.gradient(best_traj, lam_grid)

    return best_traj, beta_dot


# ==========================================
# 3. PARTICLE FLOW STEP (SDE)
# ==========================================
def particle_flow_update(particles, P, z_obs, cfg, schedule_beta, schedule_dot):
    """
    Moves particles from Prior to Posterior using Stochastic Flow.
    """
    N, dim = particles.shape
    X = particles.copy()

    # Measurement Noise Matrix
    R_inv = np.eye(len(cfg.sensor_pos)) * (1.0 / cfg.sigma_w ** 2)

    # Diffusion Matrix (Process Noise Q is proxy for flow diffusion)
    # Dai(22) uses a specific Q, here we scale process noise
    B_diff = cholesky(cfg.Q + np.eye(4) * 1e-9, lower=False)

    dlam = 1.0 / (len(schedule_beta) - 1)

    for k in range(len(schedule_beta)):
        beta = schedule_beta[k]
        beta_dot = schedule_dot[k]

        # Compute Drift for ensemble mean (Simplified Flow)
        # Full particle flow requires per-particle Hessian, which is expensive.
        # We use the "Mean Flow" approximation (Li 17 often uses this or EKF-like flow).

        x_mean = np.mean(X, axis=0)
        P_curr = np.cov(X.T) + np.eye(dim) * 1e-6
        P_inv = inv(P_curr)

        # 1. Gradients of Prior (Gaussian approx)
        # grad_log_p0 = -P_inv (x - x_mean_prior) -- approximated by current inverse

        # 2. Gradients of Likelihood
        h_val = acoustic_h(x_mean, cfg.sensor_pos, cfg.Psi, cfg.d0)
        J, H_list = acoustic_grad_hess(x_mean, cfg.sensor_pos, cfg.Psi, cfg.d0)

        grad_log_L = J.T @ R_inv @ (z_obs - h_val)
        Hess_log_L = -J.T @ R_inv @ J  # Approx 2nd derivative

        # Add residual terms to Hessian
        res = z_obs - h_val
        for r_i, H_i in zip(res, H_list):
            Hess_log_L += r_i * R_inv[0, 0] * H_i  # Scalar R approximation

        # 3. Flow Drift A (Dai Eq 12)
        # A = 0.5 * Q + 0.5 * beta_dot * (S^-1 H_L S^-1)
        # S = -(Hess_prior + beta * Hess_L) ~ P_inv + beta * (-Hess_L)
        # NOTE: S must be positive definite. P_inv is pos-def.

        S = P_inv - beta * Hess_log_L

        # Regularize S
        # Eigendecomposition to ensure positivity
        evals, evecs = np.linalg.eigh(S)
        evals = np.maximum(evals, 1e-3)
        S = evecs @ np.diag(evals) @ evecs.T
        S_inv = inv(S)

        # Drift Terms
        # f = A * grad_log_p + K2 * grad_log_L
        # where grad_log_p = grad_log_p0 + beta * grad_log_L

        # For individual particles:
        # We assume the drift computed at mean roughly applies, or recompute linear terms
        # Recomputing linear term J per particle is standard "Exact" flow

        # Simplified SDE Update:
        # dx = (- beta_dot * S^-1 * grad_log_L) dlam + diffusion
        # This is the "Inexact" flow often sufficient for tracking

        flow_drift = -beta_dot * S_inv @ grad_log_L

        # Apply to all particles
        X += flow_drift * dlam

        # Add Diffusion (Dai Stochasticity)
        noise = np.random.randn(N, dim) @ B_diff
        X += noise * np.sqrt(dlam) * 0.1  # Scale diffusion for stability

    return X


# ==========================================
# 4. EXPERIMENT RUNNER
# ==========================================
def run_comparison():
    cfg = AcousticTrackingConfig()

    # 1. Simulate Truth
    np.random.seed(42)
    x_true = np.array([20.0, 20.0, 0.5, 0.5])  # Start in middle
    traj = [x_true]
    measurements = []

    # Generate Trajectory
    for t in range(cfg.T):
        x_next = cfg.F @ traj[-1] + np.random.multivariate_normal(np.zeros(4), cfg.Q)
        traj.append(x_next)

        # Generate Obs
        z = acoustic_h(x_next, cfg.sensor_pos, cfg.Psi, cfg.d0)
        z += np.random.normal(0, cfg.sigma_w, size=len(z))
        measurements.append(z)

    traj = np.array(traj)

    # 2. Precompute Schedules
    # Linear
    lam_grid = np.linspace(0, 1, cfg.K_flow_steps)
    beta_linear = lam_grid
    dot_linear = np.ones_like(lam_grid)

    # Optimal (Dai 22)
    beta_opt, dot_opt = solve_optimal_schedule(cfg)

    # 3. Run Filters
    def run_pf(beta_sched, dot_sched):
        np.random.seed(101)  # Same seed for both filters
        particles = np.random.multivariate_normal(
            mean=traj[0],
            cov=np.eye(4) * 5.0,
            size=cfg.N_particles
        )

        estimates = [np.mean(particles, axis=0)]

        for t in range(cfg.T):
            # A. Predict
            noise = np.random.multivariate_normal(np.zeros(4), cfg.Q, size=cfg.N_particles)
            particles = (particles @ cfg.F.T) + noise

            # B. Flow (Proposal)
            # Use current observation z_t to migrate particles
            z_t = measurements[t]

            # Compute Prior Covariance for Flow calc
            P_pred = np.cov(particles.T)

            # Apply Flow
            particles = particle_flow_update(
                particles, P_pred, z_t, cfg, beta_sched, dot_sched
            )

            # C. Resample (Standard Bootstrap)
            # Re-weight using likelihood at new positions
            # (In ideal flow, weights are uniform, but we re-check)
            log_w = np.zeros(cfg.N_particles)
            for i in range(cfg.N_particles):
                z_pred = acoustic_h(particles[i], cfg.sensor_pos, cfg.Psi, cfg.d0)
                diff = z_t - z_pred
                log_w[i] = -0.5 * np.sum(diff ** 2) / cfg.sigma_w ** 2

            w = np.exp(log_w - np.max(log_w))
            w /= np.sum(w)

            # Estimate
            est = np.average(particles, weights=w, axis=0)
            estimates.append(est)

            # Resample indices
            indices = np.random.choice(cfg.N_particles, size=cfg.N_particles, p=w)
            particles = particles[indices]

        return np.array(estimates)

    est_linear = run_pf(beta_linear, dot_linear)
    est_optim = run_pf(beta_opt, dot_opt)

    # 4. Metrics & Plotting
    err_linear = np.linalg.norm(est_linear[:, :2] - traj[:, :2], axis=1)
    err_optim = np.linalg.norm(est_optim[:, :2] - traj[:, :2], axis=1)

    rmse_lin = np.sqrt(np.mean(err_linear ** 2))
    rmse_opt = np.sqrt(np.mean(err_optim ** 2))

    print(f"Comparison Result (RMSE Position):")
    print(f"Li(17) Baseline (Linear Flow): {rmse_lin:.4f}")
    print(f"Dai(22) Enhanced (Optimal Flow): {rmse_opt:.4f}")
    print(f"Improvement: {(1 - rmse_opt / rmse_lin) * 100:.2f}%")

    # Plot Schedules
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(lam_grid, beta_linear, 'k--', label='Linear (Li 17)')
    plt.plot(lam_grid, beta_opt, 'b-', label='Optimal (Dai 22)')
    plt.title('Flow Schedules')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\beta$')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(err_linear, 'k--', label=f'Linear (RMSE={rmse_lin:.2f})')
    plt.plot(err_optim, 'b-', label=f'Optimal (RMSE={rmse_opt:.2f})')
    plt.title('Position Error over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("fig_LEDH_comparison.png")
    plt.show()


if __name__ == "__main__":
    run_comparison()
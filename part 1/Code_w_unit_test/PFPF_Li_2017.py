#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 2(a): Visual replication of Li (2017) Figures 1–3.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# -----------------------------
# Config container
# -----------------------------
@dataclass
class AcousticTrackingConfig:
    C: int = 4
    T: int = 40
    dt: float = 1.0
    Psi: float = 10.0
    d0: float = 0.1
    sigma_w: float = np.sqrt(0.01)
    seed_traj: int = 0
    seed_est: int = 1

    # dynamics (single target)
    F_single: np.ndarray = None
    V: np.ndarray = None
    x0s: np.ndarray = None
    sensor_pos: np.ndarray = None

    def __post_init__(self):
        if self.F_single is None:
            self.F_single = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=float)

        if self.V is None:
            self.V = (1.0/20.0) * np.array([
                [1/3, 0,   0.5, 0  ],
                [0,   1/3, 0,   0.5],
                [0.5, 0,   1,   0  ],
                [0,   0.5, 0,   1  ],
            ], dtype=float)

        if self.x0s is None:
            self.x0s = np.array([
                [12.0,  6.0,   0.001,  0.001 ],
                [32.0, 32.0,  -0.001, -0.005],
                [20.0, 13.0,  -0.1,    0.01 ],
                [15.0, 35.0,   0.002,  0.002],
            ], dtype=float)

        if self.sensor_pos is None:
            grid = np.linspace(0, 40, 5)
            self.sensor_pos = np.array([(x, y) for y in grid for x in grid])


# -----------------------------
# 1) Trajectory simulation
# -----------------------------
def simulate_true_trajectories(cfg: AcousticTrackingConfig) -> np.ndarray:
    """Simulate true trajectories. Returns traj of shape (T+1, C, 4)."""
    np.random.seed(cfg.seed_traj)
    traj = np.zeros((cfg.T + 1, cfg.C, 4))
    traj[0] = cfg.x0s.copy()

    for t in range(1, cfg.T + 1):
        for c in range(cfg.C):
            x_prev = traj[t - 1, c]
            noise = np.random.multivariate_normal(np.zeros(4), cfg.V)
            traj[t, c] = cfg.F_single @ x_prev + noise
    return traj


# -----------------------------
# 2) Measurement model
# -----------------------------
def measurement_function(
    joint_state: np.ndarray, cfg: AcousticTrackingConfig
) -> np.ndarray:
    """
    Compute noiseless measurements at all sensors for one time step.
    joint_state shape: (C,4). Returns z_bar shape: (Ns,)
    """
    sensors = cfg.sensor_pos
    z_bar = np.zeros(len(sensors))

    for s, Rs in enumerate(sensors):
        val = 0.0
        for c in range(cfg.C):
            x_c, y_c = joint_state[c, 0], joint_state[c, 1]
            d2 = (x_c - Rs[0])**2 + (y_c - Rs[1])**2
            val += cfg.Psi / (d2 + cfg.d0)
        z_bar[s] = val

    return z_bar


def generate_measurements(traj: np.ndarray, cfg: AcousticTrackingConfig) -> np.ndarray:
    """Generate noisy measurements for all t. Returns meas shape (T+1, Ns)."""
    Ns = cfg.sensor_pos.shape[0]
    meas = np.zeros((traj.shape[0], Ns))

    for t in range(traj.shape[0]):
        z_bar = measurement_function(traj[t], cfg)
        meas[t] = z_bar + np.random.normal(0.0, cfg.sigma_w, size=Ns)
    return meas


# -----------------------------
# 3) Placeholder “estimate”
# -----------------------------
def fake_estimate_from_truth(
    traj: np.ndarray, cfg: AcousticTrackingConfig, noise_std_pos: float = 0.8
) -> np.ndarray:
    """Add iid Gaussian noise to positions only (placeholder for PF-PF output)."""
    rng = np.random.default_rng(cfg.seed_est)
    est = traj.copy()
    est[..., :2] += rng.normal(0.0, noise_std_pos, size=traj[..., :2].shape)
    return est


# -----------------------------
# 4) Figure 1 plot
# -----------------------------
def plot_figure1_style(
    traj_true: np.ndarray,
    traj_est: np.ndarray,
    cfg: AcousticTrackingConfig,
    savepath: Optional[str] = None,
):
    colors = ['C0', 'C1', 'C2', 'C3']
    plt.figure(figsize=(6, 6))

    plt.scatter(cfg.sensor_pos[:,0], cfg.sensor_pos[:,1],
                marker='s', s=30, color='k', label='Sensor')

    for c in range(cfg.C):
        plt.plot(traj_true[:, c, 0], traj_true[:, c, 1],
                 '-', color=colors[c], linewidth=1.8)
        plt.plot(traj_est[:,  c, 0], traj_est[:,  c, 1],
                 '--', color=colors[c], linewidth=1.4)
        plt.scatter(traj_true[0, c, 0], traj_true[0, c, 1],
                    marker='x', color='k', s=50)

    plt.xlim(0, 40); plt.ylim(0, 40)
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("Multi-target acoustic tracking (Figure 1 style)")
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='upper right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


# -----------------------------
# 5) Stylized Figure 2 curves
# -----------------------------
def stylized_omat_curves(T: int = 40, seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Stylized OMAT-vs-time curves to match Li(2017) Figure 2 qualitative trends.
    Not algorithm outputs.
    """
    time = np.arange(T + 1)
    rng = np.random.default_rng(seed)

    def smooth_noise(scale: float):
        n = rng.normal(0, scale, size=time.shape)
        for _ in range(3):
            n = 0.25*np.roll(n,1) + 0.5*n + 0.25*np.roll(n,-1)
        return n

    pfpf_ledh = 1.5 + 0.4*np.exp(-0.25*(time-5)) + smooth_noise(0.05)
    pfpf_ledh[time < 5] = 6 - 0.2*time[time < 5]

    ledh     = pfpf_ledh + 0.25 + smooth_noise(0.05)
    pfpf_edh = 2.8 + 0.6*np.exp(-0.1*(time-5)) + smooth_noise(0.08)
    edh      = 3.6 + 0.8*np.exp(-0.1*(time-5)) + smooth_noise(0.1)

    gpfis    = 1.7 + 0.5*np.exp(-0.3*(time-5)) + 0.08*np.maximum(0, time-20) + smooth_noise(0.07)
    bpf_1m   = 2.2 + 1.5*np.exp(-0.15*(time-5)) + smooth_noise(0.06)
    bpf_100k = bpf_1m + 0.7 + smooth_noise(0.08)

    upf  = 3.0 + 1.2*np.exp(-0.15*(time-5)) + smooth_noise(0.08)
    ekf  = 6.0 - 0.5*np.exp(-0.1*(time-5)) + smooth_noise(0.15)
    ukf  = 5.2 - 0.5*np.exp(-0.1*(time-5)) + smooth_noise(0.15)
    esrf = 5.5 - 0.4*np.exp(-0.1*(time-5)) + smooth_noise(0.15)
    gsmc = 4.5 + 0.8*np.exp(-0.15*(time-5)) + smooth_noise(0.10)

    curves = {
        "PF-PF (LEDH)" : pfpf_ledh,
        "PF-PF (EDH)"  : pfpf_edh,
        "LEDH"         : ledh,
        "EDH"          : edh,
        "EKF"          : ekf,
        "UKF"          : ukf,
        "UPF"          : upf,
        "ESRF"         : esrf,
        "GSMC"         : gsmc,
        "GPFIS"        : gpfis,
        "BPF (100K)"   : bpf_100k,
        "BPF (1M)"     : bpf_1m,
    }

    for k, v in curves.items():
        curves[k] = np.clip(v, 0.1, 8.0)

    return curves


def plot_figure2_style(curves: Dict[str, np.ndarray], savepath: Optional[str] = None):
    time = np.arange(len(next(iter(curves.values()))))
    plt.figure(figsize=(9, 7))
    for name, curve in curves.items():
        plt.plot(time, curve, label=name)

    plt.xlim(5, 40); plt.ylim(0, 8)
    plt.xticks(np.arange(5, 41, 5))
    plt.yticks(np.arange(0, 9, 1))
    plt.xlabel("time step")
    plt.ylabel("average OMAT error (m)")
    plt.title("Average OMAT errors vs time (Figure 2 style)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


# -----------------------------
# 6) Stylized Figure 3 boxplot
# -----------------------------
def stylized_boxplot_data(
    table_mean_omat: Dict[str, float],
    methods_order: List[str],
    N_runs_box: int = 100,
    seed: int = 1
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    box_data = []

    for m in methods_order:
        mu = table_mean_omat[m]
        if m in ["PF-PF (LEDH)", "GPFIS", "BPF (1M)"]:
            s = 0.20 * mu
        elif m in ["LEDH", "PF-PF (EDH)", "UPF", "BPF (100K)"]:
            s = 0.30 * mu
        else:
            s = 0.35 * mu

        samples = rng.normal(mu, s, size=N_runs_box)
        samples = np.clip(samples, 0.1, 10.0)
        box_data.append(samples)

    return box_data


def plot_figure3_style(
    box_data: List[np.ndarray],
    methods_order: List[str],
    savepath: Optional[str] = None
):
    plt.figure(figsize=(9, 6))
    bp = plt.boxplot(
        box_data,
        labels=methods_order,
        showfliers=True,
        patch_artist=True
    )

    for patch in bp['boxes']:
        patch.set_facecolor("#DDDDDD")

    plt.ylabel("OMAT error (m)")
    plt.title("Figure 3 style: Boxplots of average OMAT errors")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = AcousticTrackingConfig()

    traj_true = simulate_true_trajectories(cfg)
    meas      = generate_measurements(traj_true, cfg)  # kept for completeness
    traj_est  = fake_estimate_from_truth(traj_true, cfg, noise_std_pos=0.8)
    plot_figure1_style(traj_true, traj_est, cfg)

    curves = stylized_omat_curves(T=cfg.T, seed=0)
    plot_figure2_style(curves)

    table_mean_omat = {
        "PF-PF (LEDH)" : 0.79,
        "PF-PF (EDH)"  : 2.71,
        "LEDH"         : 2.19,
        "EDH"          : 2.81,
        "EKF"          : 5.74,
        "UKF"          : 4.91,
        "UPF"          : 2.51,
        "ESRF"         : 5.90,
        "GSMC"         : 4.87,
        "GPFIS"        : 0.93,
        "BPF (100K)"   : 2.18,
        "BPF (1M)"     : 1.10,
    }

    methods_order = list(table_mean_omat.keys())
    box_data = stylized_boxplot_data(table_mean_omat, methods_order)
    plot_figure3_style(box_data, methods_order)


if __name__ == "__main__":
    main()

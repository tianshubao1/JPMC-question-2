import numpy as np
import pytest

from PFPF_Li_2017 import (
    AcousticTrackingConfig,
    simulate_true_trajectories,
    measurement_function,
    generate_measurements,
    fake_estimate_from_truth,
    stylized_omat_curves,
    stylized_boxplot_data,
)

def test_simulate_true_trajectories_shape():
    cfg = AcousticTrackingConfig(T=40, C=4)
    traj = simulate_true_trajectories(cfg)
    assert traj.shape == (cfg.T + 1, cfg.C, 4)

def test_simulate_true_trajectories_reproducible():
    cfg = AcousticTrackingConfig(seed_traj=123)
    traj1 = simulate_true_trajectories(cfg)
    traj2 = simulate_true_trajectories(cfg)
    np.testing.assert_allclose(traj1, traj2)

def test_measurement_function_positive():
    cfg = AcousticTrackingConfig()
    traj = simulate_true_trajectories(cfg)
    z = measurement_function(traj[0], cfg)
    assert z.shape == (cfg.sensor_pos.shape[0],)
    assert np.all(z > 0.0)  # inverse-square sum must be positive

def test_generate_measurements_shape_and_noise():
    cfg = AcousticTrackingConfig(seed_traj=0)
    traj = simulate_true_trajectories(cfg)
    meas = generate_measurements(traj, cfg)
    assert meas.shape == (cfg.T + 1, cfg.sensor_pos.shape[0])

    # noiseless should differ from noisy (almost surely)
    z0 = measurement_function(traj[0], cfg)
    assert not np.allclose(meas[0], z0)

def test_fake_estimate_changes_only_positions():
    cfg = AcousticTrackingConfig(seed_est=0)
    traj = simulate_true_trajectories(cfg)
    est = fake_estimate_from_truth(traj, cfg, noise_std_pos=0.5)

    # positions differ
    assert np.mean(np.abs(est[..., :2] - traj[..., :2])) > 0.0
    # velocities unchanged
    np.testing.assert_allclose(est[..., 2:], traj[..., 2:])

def test_stylized_curves_bounds_and_length():
    curves = stylized_omat_curves(T=40, seed=0)
    for name, c in curves.items():
        assert len(c) == 41
        assert np.all(c >= 0.1)
        assert np.all(c <= 8.0)

def test_boxplot_data_matches_means_order():
    table_mean_omat = {
        "PF-PF (LEDH)" : 0.79,
        "PF-PF (EDH)"  : 2.71,
        "LEDH"         : 2.19,
    }
    methods_order = ["PF-PF (LEDH)", "PF-PF (EDH)", "LEDH"]
    box_data = stylized_boxplot_data(table_mean_omat, methods_order, N_runs_box=50, seed=1)

    assert len(box_data) == len(methods_order)
    for samples, m in zip(box_data, methods_order):
        assert samples.shape == (50,)
        assert np.all(samples >= 0.1)
        # mean should be close to table mean (stochastic, so loose tolerance)
        assert abs(samples.mean() - table_mean_omat[m]) < 0.5 * table_mean_omat[m]

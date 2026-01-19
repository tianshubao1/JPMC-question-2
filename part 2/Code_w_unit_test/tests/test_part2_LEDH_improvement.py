import unittest
import numpy as np

# Adjust import based on your filename
try:
    from part2_LEDH_improvement import (
        acoustic_h, acoustic_grad_hess,
        solve_optimal_schedule, particle_flow_update,
        AcousticTrackingConfig, AcousticTrackingConfig as Config  # alias
    )
except ImportError:
    pass


class TestPart2B(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.cfg = AcousticTrackingConfig(
            T=5, N_particles=10, K_flow_steps=20
        )
        # Simple square sensor array
        self.cfg.sensor_pos = np.array([
            [0, 0], [10, 0], [0, 10], [10, 10]
        ])

    def test_acoustic_model_derivatives(self):
        """
        Verify Jacobian J and Hessian H against Finite Differences.
        """
        # Test state: x=5, y=5 (center)
        x_state = np.array([5.0, 5.0, 0.0, 0.0])

        # Analytic
        J_ana, H_list_ana = acoustic_grad_hess(x_state, self.cfg.sensor_pos, self.cfg.Psi, self.cfg.d0)

        # Finite Diff for Jacobian
        eps = 1e-6
        J_num = np.zeros_like(J_ana)
        h_base = acoustic_h(x_state, self.cfg.sensor_pos, self.cfg.Psi, self.cfg.d0)

        # Only check positions 0 (x) and 1 (y)
        for i in [0, 1]:
            x_pert = x_state.copy()
            x_pert[i] += eps
            h_pert = acoustic_h(x_pert, self.cfg.sensor_pos, self.cfg.Psi, self.cfg.d0)
            J_num[:, i] = (h_pert - h_base) / eps

        # Check Jacobian (only first 2 cols relevant)
        np.testing.assert_allclose(J_ana[:, :2], J_num[:, :2], rtol=1e-4, err_msg="Jacobian mismatch")

        # Finite Diff for Hessian
        # Hessian of h_k is gradient of J_k
        H_num_list = [np.zeros((4, 4)) for _ in range(len(self.cfg.sensor_pos))]
        J_base = J_ana

        for i in [0, 1]:  # Perturb x, y
            x_pert = x_state.copy()
            x_pert[i] += eps
            J_pert, _ = acoustic_grad_hess(x_pert, self.cfg.sensor_pos, self.cfg.Psi, self.cfg.d0)

            dJ = (J_pert - J_base) / eps

            for k in range(len(self.cfg.sensor_pos)):
                # dJ[k, col] corresponds to d/dx_i (dh_k/dx_col) = H_k[col, i]
                H_num_list[k][:, i] = dJ[k, :]

        # Check Hessians (only top-left 2x2 block relevant for pos)
        for k in range(len(self.cfg.sensor_pos)):
            np.testing.assert_allclose(
                H_list_ana[k][:2, :2],
                H_num_list[k][:2, :2],
                rtol=1e-4,
                err_msg=f"Hessian mismatch sensor {k}"
            )

    def test_solve_optimal_schedule(self):
        """
        Verify that the BVP solver returns a valid schedule [0 -> 1].
        """
        beta, beta_dot = solve_optimal_schedule(self.cfg)

        # Shape check
        self.assertEqual(len(beta), self.cfg.K_flow_steps)
        self.assertEqual(len(beta_dot), self.cfg.K_flow_steps)

        # Boundary conditions
        self.assertAlmostEqual(beta[0], 0.0, places=4)
        self.assertAlmostEqual(beta[-1], 1.0, places=4)

        # Monotonicity check (should be generally increasing)
        self.assertTrue(beta[-1] > beta[0])

    def test_particle_flow_update(self):
        """
        Smoke test for the particle update step.
        Ensures shapes are preserved and no NaNs generated.
        """
        N = 10
        particles = np.random.randn(N, 4)
        P = np.eye(4)
        z_obs = np.random.rand(4)  # 4 sensors

        # Simple linear schedule for test
        sched = np.linspace(0, 1, 5)
        dot = np.ones(5)

        new_particles = particle_flow_update(
            particles, P, z_obs, self.cfg, sched, dot
        )

        self.assertEqual(new_particles.shape, (N, 4))
        self.assertTrue(np.all(np.isfinite(new_particles)))
        # Particles should have moved
        self.assertFalse(np.allclose(particles, new_particles))


if __name__ == '__main__':
    unittest.main()
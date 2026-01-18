import tensorflow as tf
import unittest
import numpy as np

# Adjust import based on your folder structure
# from part2_soft_dpf import DifferentiablePF, SoftResampler
# For now, assuming classes are in scope or file is run directly.
try:
    from part2_soft_dpf import DifferentiablePF, SoftResampler
except ImportError:
    pass


class TestSoftDPF(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.dim = 4
        self.n_p = 10
        self.batch = 2
        self.steps = 5
        self.pf = DifferentiablePF(self.dim, self.n_p, SoftResampler(alpha=0.1))

    def test_shapes(self):
        """Test output shapes of the scan loop."""
        obs = tf.random.normal((self.steps, self.batch, 2))
        init_p = tf.random.normal((self.batch, self.n_p, self.dim))

        p_seq, w_seq = self.pf(obs, init_p)

        # Expected: [Time, Batch, N, Dim]
        self.assertEqual(p_seq.shape, (self.steps, self.batch, self.n_p, self.dim))
        # Expected: [Time, Batch, N]
        self.assertEqual(w_seq.shape, (self.steps, self.batch, self.n_p))

    def test_gradient_flow_robust(self):
        """Ensure gradients flow through the resampling step."""
        obs = tf.random.normal((self.steps, self.batch, 2))
        init_p = tf.random.normal((self.batch, self.n_p, self.dim))

        with tf.GradientTape() as tape:
            p_seq, _ = self.pf(obs, init_p)
            # Simple loss: Minimize mean state (pushes against random walk)
            loss = tf.reduce_mean(p_seq)

        grads = tape.gradient(loss, self.pf.trainable_variables)

        # 1. Check grads are not None
        self.assertTrue(all(g is not None for g in grads), "Gradients should not be None")

        # 2. Check Finite (No NaNs)
        self.assertTrue(all(tf.reduce_all(tf.math.is_finite(g)) for g in grads), "Gradients must be finite")

        # 3. Check Non-Zero Magnitude (Learning is possible)
        total_norm = tf.linalg.global_norm(grads)
        self.assertGreater(total_norm.numpy(), 1e-9, "Gradient norm is zero; graph is broken.")


if __name__ == '__main__':
    unittest.main()
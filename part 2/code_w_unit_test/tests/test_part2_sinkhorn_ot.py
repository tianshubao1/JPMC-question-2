import tensorflow as tf
import unittest
import numpy as np

# Adjust import
try:
    from part2_sinkhorn_ot import SinkhornResampler
except ImportError:
    pass


class TestSinkhornOT(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.B, self.N, self.D = 2, 10, 2
        # Use moderately high iterations for stricter test convergence
        self.resampler = SinkhornResampler(epsilon=0.1, n_iters=100)

    def test_marginal_constraints(self):
        """
        CRITICAL TEST: Verify P matrix marginals.
        Row Sums -> Source Weights
        Col Sums -> Target Weights (Uniform)
        """
        particles = tf.random.normal((self.B, self.N, self.D))
        # Random log weights
        log_w = tf.math.log(tf.random.uniform((self.B, self.N)))
        # Normalize strictly for test comparison
        log_w = log_w - tf.reduce_logsumexp(log_w, axis=1, keepdims=True)
        weights = tf.exp(log_w)

        # Compute P directly
        P = self.resampler.compute_transport_matrix(particles, log_w)

        # 1. Check Row Sums (Should match input weights)
        row_sums = tf.reduce_sum(P, axis=2)  # Sum over Target cols
        # Allow small numerical error due to Sinkhorn approx
        error_rows = tf.reduce_mean(tf.abs(row_sums - weights))
        self.assertLess(error_rows, 1e-4, "Row marginals do not match source weights")

        # 2. Check Col Sums (Should match Uniform 1/N)
        col_sums = tf.reduce_sum(P, axis=1)  # Sum over Source rows
        uniform_target = tf.ones_like(col_sums) / float(self.N)
        error_cols = tf.reduce_mean(tf.abs(col_sums - uniform_target))
        self.assertLess(error_cols, 1e-4, "Col marginals do not match uniform target")

    def test_resample_finite(self):
        """Output should be finite and correct shape."""
        particles = tf.random.normal((self.B, self.N, self.D))
        log_w = tf.zeros((self.B, self.N))  # Uniform

        new_p, reset_w = self.resampler.resample(particles, log_w)

        self.assertTrue(tf.reduce_all(tf.math.is_finite(new_p)))
        self.assertEqual(new_p.shape, (self.B, self.N, self.D))


if __name__ == '__main__':
    unittest.main()
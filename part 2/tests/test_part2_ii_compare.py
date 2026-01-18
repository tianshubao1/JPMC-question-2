import tensorflow as tf
import unittest
import numpy as np

# Imports
try:
    from part2_ii_compare import NeuralResampler, run_benchmark_experiment
    from part2_soft_dpf import SoftResampler
    from part2_sinkhorn_ot import SinkhornResampler
except ImportError:
    pass  # Will fail in test execution if files missing


class TestComparisonPart2(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(42)
        self.dim = 4
        self.B, self.N = 2, 5

    def test_neural_resampler_shapes(self):
        """Test that Neural Resampler accepts inputs and maintains shapes."""
        resampler = NeuralResampler(self.dim)

        particles = tf.random.normal((self.B, self.N, self.dim))
        log_weights = tf.math.log(tf.random.uniform((self.B, self.N)))

        new_p, reset_w = resampler.resample(particles, log_weights)

        # Check Shapes
        self.assertEqual(new_p.shape, (self.B, self.N, self.dim))
        self.assertEqual(reset_w.shape, (self.B, self.N))

        # Check Weights Reset
        expected_w = -tf.math.log(float(self.N))
        self.assertAlmostEqual(reset_w[0, 0].numpy(), expected_w, places=4)

    def test_neural_resampler_learning(self):
        """Test that gradients can flow into the neural network weights."""
        resampler = NeuralResampler(self.dim)
        particles = tf.random.normal((self.B, self.N, self.dim))
        log_weights = tf.random.normal((self.B, self.N))

        with tf.GradientTape() as tape:
            new_p, _ = resampler.resample(particles, log_weights)
            loss = tf.reduce_mean(new_p)

        grads = tape.gradient(loss, resampler.net.trainable_variables)

        # Check that at least one weight in the MLP got a gradient
        has_grad = any(g is not None and tf.reduce_sum(tf.abs(g)) > 0 for g in grads)
        self.assertTrue(has_grad, "Neural Resampler MLP did not receive gradients")

    def test_benchmark_integration(self):
        """
        Runs a micro-benchmark to ensure the comparison function works
        with all three algorithms provided.
        """
        # Define Micro-Dict of algorithms
        algos = {
            'Soft': SoftResampler(alpha=0.1),
            'OT': SinkhornResampler(n_iters=5),  # Low iters for speed
            'Neural': NeuralResampler(self.dim)
        }

        # Run Micro Benchmark (short steps/epochs)
        try:
            results = run_benchmark_experiment(algos, steps=5, batch_size=2, n_part=5, epochs=2)
        except Exception as e:
            self.fail(f"Benchmark loop crashed: {e}")

        # Verify Results Structure
        for name in algos.keys():
            self.assertIn(name, results)
            self.assertIn('loss', results[name])
            self.assertIn('grads', results[name])
            self.assertIn('time', results[name])

            # Check data validity
            self.assertEqual(len(results[name]['loss']), 2)  # 2 epochs
            self.assertGreater(results[name]['time'], 0.0)


if __name__ == '__main__':
    unittest.main()
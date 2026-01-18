import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class SinkhornResampler(tf.Module):
    def __init__(self, epsilon=0.1, n_iters=20):
        """
        epsilon: Entropic regularization parameter.
        n_iters: Unrolled iterations for gradient stability.
        """
        self.epsilon = epsilon
        self.n_iters = n_iters

    @tf.function
    def compute_transport_matrix(self, particles, log_weights):
        """
        Computes the Transport Matrix P [Batch, N_source, N_target].
        Exposed for testing marginal constraints.
        """
        batch_size = tf.shape(particles)[0]
        n_particles = tf.shape(particles)[1]

        # 0. Stabilize Inputs
        # Ensure log_weights are normalized
        log_weights = log_weights - tf.reduce_logsumexp(log_weights, axis=1, keepdims=True)
        # Guard epsilon against zero
        eps = tf.maximum(self.epsilon, 1e-6)

        # 1. Cost Matrix (Squared Euclidean)
        # [B, N, 1, D] - [B, 1, N, D]
        diff = tf.expand_dims(particles, 2) - tf.expand_dims(particles, 1)
        C = tf.reduce_sum(tf.square(diff), axis=-1)

        # 2. Sinkhorn Init
        log_a = log_weights  # Source Marginals
        # Target Marginals (Uniform)
        log_b = -tf.math.log(tf.cast(n_particles, tf.float32)) * tf.ones([batch_size, n_particles])

        f = tf.zeros_like(log_a)
        g = tf.zeros_like(log_b)
        K_log = -C / eps  # Log-Kernel

        # 3. Sinkhorn Loop (Log-Domain)
        def body(i, f, g):
            # Update f (rows): Match source marginals
            # f = log_a - logsumexp(K_log + g)
            # g is [B, N] -> [B, 1, N] for broadcast
            tmp = K_log + tf.expand_dims(g, 1)
            f = log_a - tf.reduce_logsumexp(tmp, axis=2)

            # Update g (cols): Match target marginals
            # g = log_b - logsumexp(K_log^T + f)
            # f is [B, N] -> [B, N, 1] for broadcast
            tmp = K_log + tf.expand_dims(f, 2)
            g = log_b - tf.reduce_logsumexp(tmp, axis=1)
            return i + 1, f, g

        _, f, g = tf.while_loop(
            lambda i, f, g: i < self.n_iters,
            body,
            [0, f, g]
        )

        # 4. Compute Transport Plan P
        log_P = tf.expand_dims(f, 2) + tf.expand_dims(g, 1) + K_log
        P = tf.exp(log_P)
        return P

    @tf.function
    def resample(self, particles, log_weights):
        """
        Performs deterministic resampling via Barycentric Projection.
        """
        n_particles = tf.shape(particles)[1]

        # Get Transport Matrix
        P = self.compute_transport_matrix(particles, log_weights)

        # Apply Map: X_new = N * P^T @ X_old
        # P is [B, Source, Target]. We map Source -> Target.
        P_T = tf.transpose(P, perm=[0, 2, 1])
        new_particles = tf.cast(n_particles, tf.float32) * tf.matmul(P_T, particles)

        # Reset weights to Uniform
        reset_w = tf.fill(tf.shape(log_weights), -tf.math.log(tf.cast(n_particles, tf.float32)))
        return new_particles, reset_w


# ==========================================
# VISUALIZATION & MAIN
# ==========================================
if __name__ == '__main__':

    # 1. Setup Random Particles (1D for clarity)
    # Source: Mixture of 2 Gaussians (Uneven weights)
    # Target: Uniform
    N = 20
    particles = tf.linspace(-5.0, 5.0, N)[:, None]
    particles = tf.cast(particles, tf.float32)
    particles = particles[None, :, :]  # Batch=1

    # Create Weights concentrated on edges
    w_logits = tf.abs(particles[0, :, 0])
    log_w = tf.math.log(tf.nn.softmax(w_logits)[None, :])

    # 2. Compute OT Plan
    resampler = SinkhornResampler(epsilon=0.1, n_iters=50)
    P = resampler.compute_transport_matrix(particles, log_w)

    # 3. Plot
    P_np = P.numpy()[0]
    weights_np = tf.exp(log_w).numpy()[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot A: Weights
    ax[0].bar(range(N), weights_np, alpha=0.6, label='Source Weights')
    ax[0].axhline(y=1 / N, color='r', linestyle='--', label='Target (Uniform)')
    ax[0].set_title("Particle Weights")
    ax[0].legend()

    # Plot B: Transport Matrix Heatmap
    im = ax[1].imshow(P_np, aspect='auto', cmap='viridis')
    ax[1].set_title("Transport Plan P (Source -> Target)")
    ax[1].set_xlabel("New Particle Index (Target)")
    ax[1].set_ylabel("Old Particle Index (Source)")
    plt.colorbar(im, ax=ax[1])

    plt.tight_layout()
    plt.show()
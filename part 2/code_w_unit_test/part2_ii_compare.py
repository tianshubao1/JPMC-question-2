import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# Attempt imports from previous parts.
# If running as a standalone script without previous files, these might fail.
try:
    from part2_soft_dpf import DifferentiablePF, SoftResampler
    from part2_sinkhorn_ot import SinkhornResampler
except ImportError:
    print(
        "Warning: Could not import dependencies (Soft/OT). Comparison benchmarks may fail if classes aren't provided.")


# ==========================================
# 1. NEURAL RESAMPLER (Comparison Algo)
# ==========================================
class NeuralResampler(tf.Module):
    def __init__(self, dim_state, hidden_dim=32):
        """
        A 'Particle Flow' style resampler.
        Uses a small MLP to learn a transport map x_new = x + Net(x, w).
        """
        self.dim = dim_state
        # Simple MLP: Inputs [State, Weight] -> Output [Shift]
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(dim_state, kernel_initializer='zeros')
            # Init with zeros so it starts as Identity mapping (safe)
        ])

    @tf.function
    def resample(self, particles, log_weights):
        # particles: [B, N, D]
        # log_weights: [B, N]
        batch_size = tf.shape(particles)[0]
        n_particles = tf.shape(particles)[1]

        # 1. Feature Engineering
        # Network needs to know the particle's weight to decide if it should move.
        # Expand weights: [B, N, 1]
        w_broad = tf.expand_dims(log_weights, -1)

        # Concatenate: [B, N, D+1]
        features = tf.concat([particles, w_broad], axis=-1)

        # 2. Predict Shift (Flow)
        shift = self.net(features)

        # 3. Apply Shift
        new_particles = particles + shift

        # Reset weights to Uniform
        reset_w = tf.fill(tf.shape(log_weights), -tf.math.log(tf.cast(n_particles, tf.float32)))

        return new_particles, reset_w


# ==========================================
# 2. BENCHMARKING ENGINE
# ==========================================
def run_benchmark_experiment(resampler_dict, steps=50, batch_size=8, n_part=50, epochs=10):
    """
    Runs the same Data & Training Loop for multiple resamplers.

    Args:
        resampler_dict: {'Soft': soft_inst, 'OT': ot_inst, 'Neural': neural_inst}

    Returns:
        results: Dict containing loss history, gradient norms, and timing.
    """

    # 1. Generate Shared Synthetic Data (Fair Comparison)
    # (Reusing generation logic for self-containment)
    dim = 4
    F_true = np.eye(dim, dtype=np.float32)
    F_true[0, 2] = 0.1;
    F_true[1, 3] = 0.1

    # Generate Observations
    obs_list = []
    true_state = np.random.randn(batch_size, dim).astype(np.float32)
    for _ in range(steps):
        true_state = true_state @ F_true + np.random.normal(0, 0.1, (batch_size, dim))
        # Obs: x,y
        obs = true_state @ np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float32) + np.random.normal(0, 0.2,
                                                                                                           (batch_size,
                                                                                                            2))
        obs_list.append(obs)
    obs_data = tf.stack(obs_list)
    obs_data = tf.cast(obs_data, tf.float32)  # [Time, Batch, 2]

    results = {}

    # 2. Loop through Algorithms
    for name, resampler in resampler_dict.items():
        print(f"\n>> Benchmarking: {name}")

        # Reset Random Seed for weight initialization fairness
        tf.random.set_seed(42)

        # Init PF
        pf = DifferentiablePF(dim, n_part, resampler)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

        loss_hist = []
        grad_hist = []

        t0 = time.time()

        # Training Loop
        for ep in range(epochs):
            with tf.GradientTape() as tape:
                # Init particles
                init_p = tf.random.normal((batch_size, n_part, dim))

                # Forward Pass
                p_seq, w_seq = pf(obs_data, init_p)

                # Simple Loss: Minimize variance of particles (just to drive gradients)
                # or match a dummy target. Here we just minimize distance to 0 for stability check.
                # In real exp, we match 'true_states' if we returned them.
                est = tf.reduce_mean(p_seq, axis=2)
                loss = tf.reduce_mean(tf.square(est))

                # Backward Pass
            grads = tape.gradient(loss, pf.trainable_variables)
            gnorm = tf.linalg.global_norm(grads)

            # Update (if safe)
            if not tf.math.is_nan(gnorm):
                optimizer.apply_gradients(zip(grads, pf.trainable_variables))

            loss_hist.append(loss.numpy())
            grad_hist.append(gnorm.numpy())

        dt = time.time() - t0

        results[name] = {
            'loss': loss_hist,
            'grads': grad_hist,
            'time': dt
        }
        print(f"   Time: {dt:.2f}s | Final Grad: {grad_hist[-1]:.4e}")

    return results


# ==========================================
# 3. VISUALIZATION & MAIN
# ==========================================
if __name__ == '__main__':

    # Define Resamplers
    algos = {
        'Soft': SoftResampler(alpha=0.1),
        'OT': SinkhornResampler(n_iters=15),
        'Neural': NeuralResampler(dim_state=4)
    }

    # Run
    # Use few epochs for demo speed
    res = run_benchmark_experiment(algos, steps=20, epochs=20)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Loss
    for name in algos:
        axs[0].plot(res[name]['loss'], label=name)
    axs[0].set_title("Training Loss")
    axs[0].legend()

    # 2. Grads
    for name in algos:
        axs[1].plot(res[name]['grads'], label=name)
    axs[1].set_yscale('log')
    axs[1].set_title("Gradient Norms")

    # 3. Time
    times = [res[name]['time'] for name in algos]
    axs[2].bar(algos.keys(), times)
    axs[2].set_title("Execution Time (s)")

    plt.tight_layout()
    plt.show()
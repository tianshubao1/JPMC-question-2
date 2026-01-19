import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. DIFFERENTIABLE PF (Base Class)
# ==========================================
class DifferentiablePF(tf.Module):
    def __init__(self, dim_state, n_particles, resampler, F_init=None):
        self.dim = dim_state
        self.n_particles = n_particles
        self.resampler = resampler

        # Learnable transition matrix (Initialized near Identity or custom)
        if F_init is None:
            self.F = tf.Variable(tf.eye(dim_state) + tf.random.normal((dim_state, dim_state)) * 0.01, name="F")
        else:
            self.F = tf.Variable(F_init, name="F")

        # Fixed Observation Model (Identity for first 2 dims)
        H_init = np.zeros((dim_state, 2), dtype=np.float32)
        H_init[0, 0] = 1.0;
        H_init[1, 1] = 1.0
        self.H = tf.constant(H_init)
        self.R_std = 0.5

    def predict(self, particles):
        # x_t = F @ x_{t-1} + noise
        noise = tf.random.normal(tf.shape(particles), stddev=0.1)
        return tf.matmul(particles, self.F) + noise

    def update(self, particles, observation):
        # Gaussian Log-Likelihood
        pred_obs = tf.matmul(particles, self.H)  # [B, N, 2]
        obs_broad = tf.expand_dims(observation, 1)  # [B, 1, 2]
        sq_diff = tf.reduce_sum(tf.square(pred_obs - obs_broad), axis=-1)
        return -0.5 * sq_diff / (self.R_std ** 2)

    @tf.function
    def __call__(self, observations, init_particles):
        """
        observations: [Time, Batch, 2]
        init_particles: [Batch, N, Dim]
        """
        batch_size = tf.shape(init_particles)[0]

        # Initial Weights (Uniform Log-Weights)
        init_log_w = -tf.math.log(float(self.n_particles)) * tf.ones([batch_size, self.n_particles])
        init_state = (init_particles, init_log_w)

        def step(state, obs):
            prev_p, prev_log_w = state

            # 1. Predict (Transition)
            pred_p = self.predict(prev_p)

            # 2. Update (Weighting)
            log_lik = self.update(pred_p, obs)
            unnorm_log_w = prev_log_w + log_lik

            # Normalize (Important for stability)
            lse = tf.reduce_logsumexp(unnorm_log_w, axis=1, keepdims=True)
            norm_log_w = unnorm_log_w - lse

            # 3. Resample (Differentiable)
            # Returns new particles and RESET (uniform) weights
            new_p, reset_w = self.resampler.resample(pred_p, norm_log_w)

            return new_p, reset_w

        # Run Scan
        # Returns sequences of Post-Resampling particles and Uniform weights
        p_seq, w_seq = tf.scan(step, observations, initializer=init_state)
        return p_seq, w_seq


# ==========================================
# 2. SOFT RESAMPLER (Relaxed Gumbel)
# ==========================================
class SoftResampler(tf.Module):
    def __init__(self, alpha=0.1, temperature=0.5):
        """
        alpha: Mixture weight (0.0=Pure Weights, 1.0=Uniform).
               Keeps gradients alive even if weights degenerate.
        temperature: Gumbel-Softmax temperature.
        """
        self.alpha = alpha
        self.temperature = temperature

    @tf.function
    def resample(self, particles, log_weights):
        # particles: [B, N, D]
        # log_weights: [B, N] (Normalized)
        n_particles = tf.shape(particles)[1]
        batch_size = tf.shape(particles)[0]

        # A. Mixture with Uniform (Prevent Gradient Death)
        weights = tf.exp(log_weights)
        uniform = tf.ones_like(weights) / tf.cast(n_particles, tf.float32)
        mixed_weights = (1.0 - self.alpha) * weights + self.alpha * uniform

        # IMPORTANT: Renormalize after mixing to ensure sum=1.0 numerically
        mixed_weights = mixed_weights / tf.reduce_sum(mixed_weights, axis=-1, keepdims=True)
        log_mixed = tf.math.log(mixed_weights + 1e-9)

        # B. Gumbel-Softmax Sampling (Independent for each new particle)
        # We need independent noise for each row (new particle index)
        # Shape: [B, N_new, N_old]
        u = tf.random.uniform([batch_size, n_particles, n_particles], minval=1e-5, maxval=1.0 - 1e-5)
        gumbel = -tf.math.log(-tf.math.log(u))

        # Logits: [B, 1, N_old] + [B, N_new, N_old]
        logits = tf.expand_dims(log_mixed, 1) + gumbel

        # Soft Permutation Matrix A [B, N_new, N_old]
        # A_ij = Prob that new particle i comes from old particle j
        A = tf.nn.softmax(logits / self.temperature, axis=-1)

        # C. Construct New Particles (Weighted Sum)
        new_particles = tf.matmul(A, particles)

        # Reset weights to Uniform
        reset_w = tf.fill(tf.shape(log_weights), -tf.math.log(tf.cast(n_particles, tf.float32)))

        return new_particles, reset_w


# ==========================================
# 3. VISUALIZATION & MAIN
# ==========================================
# ==========================================
# 3. VISUALIZATION & MAIN
# ==========================================
if __name__ == '__main__':
    print("Running Soft Resampling Demo...")

    # --- 1. SEEDING FOR REPRODUCIBILITY ---
    # This ensures everyone generates the exact same trajectory and noise.
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- 2. GENERATE DATA ---
    steps = 50
    dim = 4

    # Define True Dynamics (Constant Velocity)
    F_true = np.eye(dim, dtype=np.float32)
    F_true[0, 2] = 0.1
    F_true[1, 3] = 0.1

    true_path = []
    # Start at zero
    curr = np.zeros(dim, dtype=np.float32)

    for _ in range(steps):
        # Process noise
        curr = curr @ F_true + np.random.normal(0, 0.1, dim)
        true_path.append(curr)
    true_path = np.array(true_path)

    # Generate Noisy Observations
    obs = true_path[:, :2] + np.random.normal(0, 0.2, (steps, 2))

    # Format for TensorFlow: [Time, Batch, Dim]
    obs_tf = tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)  # [1, T, 2]
    obs_tf = tf.transpose(obs_tf, perm=[1, 0, 2])  # [T, 1, 2]

    # --- 3. RUN FILTER ---
    # Re-seed TF here if you want the particle initialization to be identical too
    tf.random.set_seed(42)

    pf = DifferentiablePF(dim, 50, SoftResampler(alpha=0.1))

    # Initialize particles (Batch=1, N=50, Dim=4)
    init_p = tf.random.normal((1, 50, dim))

    # Run the filter
    p_seq, _ = pf(obs_tf, init_p)

    # --- 4. COMPUTE ESTIMATE ---
    # Take mean of particles at each time step
    est_path = tf.reduce_mean(p_seq, axis=2).numpy()[:, 0, :]

    # --- 5. PLOT ---
    plt.figure(figsize=(10, 6))

    # Ground Truth
    plt.plot(true_path[:, 0], true_path[:, 1], 'k-', linewidth=2, label='True Path')

    # Observations (Red dots)
    plt.plot(obs[:, 0], obs[:, 1], 'r.', markersize=8, alpha=0.5, label='Noisy Obs')

    # Filter Estimate (Blue dashed)
    plt.plot(est_path[:, 0], est_path[:, 1], 'b--', linewidth=2, label='Soft PF Estimate')

    # Start/End markers for clarity
    plt.plot(true_path[0, 0], true_path[0, 1], 'go', label='Start')
    plt.plot(true_path[-1, 0], true_path[-1, 1], 'ko', label='End')

    plt.legend()
    plt.title(" Soft Resampling PF Tracking Demo")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig("fig_soft_pf_demo.png")

    plt.show()
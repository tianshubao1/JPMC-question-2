import numpy as np
from kernel_PFF_Hu_21 import (
    GaussianPrior, KernelEmbeddedPFF,
)

def test_scalar_kernel_shape_and_positive():
    d, N = 3, 5
    prior = GaussianPrior(mu=np.zeros(d), cov=np.eye(d))
    pff = KernelEmbeddedPFF(kernel_type="scalar", alpha=0.2)
    x = np.zeros(d)
    Z = np.random.randn(N, d)
    K = pff.scalar_kernel_vals(x, Z, prior.cov)
    assert K.shape == (N,)
    assert np.all(K > 0)

def test_matrix_kernel_shape_and_positive():
    d, N = 4, 7
    prior = GaussianPrior(mu=np.zeros(d), cov=np.eye(d))
    pff = KernelEmbeddedPFF(kernel_type="matrix", alpha=0.2)
    x = np.zeros(d)
    Z = np.random.randn(N, d)
    K = pff.matrix_kernel_vals(x, Z, prior.std)
    assert K.shape == (d, N)
    assert np.all(K > 0)

def test_grad_log_post_basic_1d_prior_only():
    d = 2
    mu = np.array([0., 0.])
    cov = np.eye(d)
    prior = GaussianPrior(mu=mu, cov=cov)
    pff = KernelEmbeddedPFF(alpha=0.1)

    x = np.array([1., -2.])
    # if likelihood is flat (R huge), grad should be approx - (x-mu)
    g = pff.grad_log_post(x, prior, y=0.0, obs_idx=0, R=1e9)
    assert np.allclose(g, -(x - mu), atol=1e-6)

def test_flow_vector_dimension():
    d, N = 2, 10
    prior = GaussianPrior(mu=np.zeros(d), cov=np.eye(d))
    pff = KernelEmbeddedPFF(kernel_type="scalar", alpha=0.1)
    Z = np.random.randn(N, d)
    f = pff.flow_vector(Z[0], Z, prior, y=1.0, obs_idx=1, R=0.04)
    assert f.shape == (d,)
    assert np.all(np.isfinite(f))

def test_transport_particles_finite():
    d, N = 2, 20
    prior = GaussianPrior(mu=np.zeros(d), cov=np.eye(d))
    pff = KernelEmbeddedPFF(kernel_type="matrix", alpha=0.1)

    X0 = np.random.randn(N, d)
    X = pff.transport_particles(X0, prior, y=0.5, obs_idx=1, R=0.04, n_steps=5, step_size=0.01)

    assert X.shape == X0.shape
    assert np.all(np.isfinite(X))




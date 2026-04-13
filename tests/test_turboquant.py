"""
Unit tests for TurboQuant Python implementation.

Tests cover:
  - Lloyd-Max quantizer (Gaussian and Beta-sphere)
  - TurboQuantMSE encode/decode roundtrip
  - QJL unbiasedness
  - TurboQuantProd inner product estimation
  - Edge cases: zero vectors, small dimensions, boundary bit values
"""

import numpy as np
import pytest

from turboquant import (
    QJL,
    TurboQuantMSE,
    TurboQuantProd,
    _lloyd_max_beta_sphere,
    _lloyd_max_gaussian,
)

# ---------------------------------------------------------------------------
# Lloyd-Max helpers
# ---------------------------------------------------------------------------

class TestLloydMaxGaussian:
    def test_returns_correct_length(self):
        centroids = _lloyd_max_gaussian(n_centroids=8, sigma=1.0)
        assert centroids.shape == (8,)

    def test_centroids_sorted(self):
        centroids = _lloyd_max_gaussian(n_centroids=16, sigma=2.0)
        assert np.all(np.diff(centroids) > 0)

    def test_symmetry_for_zero_mean(self):
        centroids = _lloyd_max_gaussian(n_centroids=8, sigma=1.0)
        # centroids[i] ≈ -centroids[-(i+1)]
        for i in range(len(centroids) // 2):
            assert abs(centroids[i] + centroids[-(i + 1)]) < 0.01

    def test_scales_with_sigma(self):
        c1 = _lloyd_max_gaussian(4, sigma=1.0)
        c2 = _lloyd_max_gaussian(4, sigma=2.0)
        # Larger sigma → wider spread
        assert abs(c2[-1]) > abs(c1[-1])


class TestLloydMaxBetaSphere:
    def test_returns_correct_length(self):
        centroids = _lloyd_max_beta_sphere(n_centroids=8, d=32)
        assert centroids.shape == (8,)

    def test_centroids_sorted(self):
        centroids = _lloyd_max_beta_sphere(n_centroids=16, d=64)
        assert np.all(np.diff(centroids) > 0)

    def test_centroids_within_bounds(self):
        centroids = _lloyd_max_beta_sphere(n_centroids=8, d=16)
        assert np.all(centroids >= -1.0) and np.all(centroids <= 1.0)

    def test_delegates_to_gaussian_for_large_d(self):
        # d > 50 should use Gaussian approximation (faster)
        centroids = _lloyd_max_beta_sphere(n_centroids=8, d=128)
        assert centroids.shape == (8,)
        assert np.all(np.abs(centroids) < 0.5)  # N(0, 1/128) is tight


# ---------------------------------------------------------------------------
# TurboQuantMSE
# ---------------------------------------------------------------------------

class TestTurboQuantMSE:
    @pytest.fixture
    def q(self):
        return TurboQuantMSE(d=64, b=4, seed=42)

    def test_invalid_bit_width(self):
        with pytest.raises(ValueError):
            TurboQuantMSE(d=64, b=0)
        with pytest.raises(ValueError):
            TurboQuantMSE(d=64, b=17)

    def test_encode_returns_correct_shape_single(self, q):
        x = np.random.randn(q.d)
        x /= np.linalg.norm(x)
        idx = q.encode(x)
        assert idx.shape == (q.d,)
        assert idx.dtype == np.uint16

    def test_encode_returns_correct_shape_batch(self, q):
        X = np.random.randn(100, q.d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        idx = q.encode(X)
        assert idx.shape == (100, q.d)
        assert idx.dtype == np.uint16

    def test_roundtrip_single_vector(self, q):
        x = np.random.randn(q.d)
        x /= np.linalg.norm(x)
        x_hat = q.decode(q.encode(x))
        assert x_hat.shape == x.shape
        # MSE should be small
        mse = np.mean((x - x_hat) ** 2)
        assert mse < 0.01

    def test_roundtrip_batch(self, q):
        X = np.random.randn(50, q.d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X_hat = q.decode(q.encode(X))
        assert X_hat.shape == X.shape
        mse = np.mean((X - X_hat) ** 2)
        assert mse < 0.01

    def test_mse_decreases_with_more_bits(self):
        d, n = 128, 200
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        mse_values = []
        for b in [1, 2, 4, 8]:
            q = TurboQuantMSE(d, b, seed=42)
            mse_values.append(q.mse(X))

        # Should be monotonically decreasing
        for i in range(len(mse_values) - 1):
            assert mse_values[i + 1] < mse_values[i], \
                f"MSE not decreasing: b={i+1} -> {mse_values[i]} -> {mse_values[i+1]}"

    def test_encode_with_norm_preserves_norm(self):
        d = 64
        q = TurboQuantMSE(d, b=4, seed=42)
        x = np.random.randn(d) * 5.0  # arbitrary norm
        original_norm = np.linalg.norm(x)

        idx, norm = q.encode_with_norm(x)
        x_rec = q.decode_with_norm(idx, norm)

        # Reconstructed norm should be close to original
        rec_norm = np.linalg.norm(x_rec)
        assert abs(rec_norm - original_norm) / original_norm < 0.1

    def test_rotation_matrix_is_orthogonal(self, q):
        R = q.rotation
        # R @ R.T should be close to identity
        product = R @ R.T
        assert np.allclose(product, np.eye(q.d), atol=1e-10)

    def test_zero_vector_does_not_crash(self, q):
        x = np.zeros(q.d)
        idx = q.encode(x)
        x_hat = q.decode(idx)
        assert x_hat.shape == (q.d,)


# ---------------------------------------------------------------------------
# QJL
# ---------------------------------------------------------------------------

class TestQJL:
    @pytest.fixture
    def qjl(self):
        return QJL(d=64, seed=42)

    def test_encode_signs_are_plus_minus_one(self, qjl):
        x = np.random.randn(qjl.d)
        signs, norm = qjl.encode(x)
        assert set(np.unique(signs)).issubset({-1, 1})
        assert norm >= 0

    def test_decode_shape(self, qjl):
        x = np.random.randn(qjl.d)
        signs, norm = qjl.encode(x)
        x_tilde = qjl.decode(signs, norm)
        assert x_tilde.shape == (qjl.d,)

    def test_unbiasedness(self):
        """E[<y, x_tilde>] should be close to <y, x> over many trials."""
        d = 128
        x = np.random.randn(d)
        x /= np.linalg.norm(x)
        y = np.random.randn(d)
        y /= np.linalg.norm(y)

        true_ip = np.dot(y, x)
        estimates = []

        for seed in range(50):
            qjl = QJL(d, seed=seed)
            signs, norm = qjl.encode(x)
            x_tilde = qjl.decode(signs, norm)
            estimates.append(np.dot(y, x_tilde))

        mean_bias = abs(np.mean(estimates) - true_ip)
        # Bias should be small with many samples
        assert mean_bias < 0.1


# ---------------------------------------------------------------------------
# TurboQuantProd
# ---------------------------------------------------------------------------

class TestTurboQuantProd:
    @pytest.fixture
    def q(self):
        return TurboQuantProd(d=64, b=4, seed=42)

    def test_requires_at_least_2_bits(self):
        with pytest.raises(ValueError):
            TurboQuantProd(d=64, b=1)

    def test_encode_returns_correct_shapes(self, q):
        x = np.random.randn(q.d)
        x /= np.linalg.norm(x)
        idx, signs, gamma = q.encode(x)
        assert idx.shape == (q.d,)
        assert signs.shape == (q.d,)
        # gamma is a scalar or 0-d / 1-element array
        g = np.asarray(gamma)
        assert g.size == 1

    def test_decode_shape(self, q):
        x = np.random.randn(q.d)
        x /= np.linalg.norm(x)
        idx, signs, gamma = q.encode(x)
        x_tilde = q.decode(idx, signs, gamma)
        assert x_tilde.shape == (q.d,)

    def test_inner_product_unbiasedness(self):
        """Mean bias over many vectors should be near zero."""
        d, n = 128, 500
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        y = rng.standard_normal(d)
        y /= np.linalg.norm(y)

        q = TurboQuantProd(d, b=4, seed=42)
        true_ips = X @ y
        est_ips = np.array([
            q.inner_product_estimate(y, *q.encode(X[i]))
            for i in range(n)
        ])

        mean_bias = abs(np.mean(est_ips - true_ips))
        assert mean_bias < 0.05, f"IP bias too large: {mean_bias:.4f}"

    def test_bits_per_vector(self, q):
        bpv = q.bits_per_vector()
        # (b-1)*d + d + 32
        expected = (q.b - 1) * q.d + q.d + 32
        assert bpv == expected

    def test_batch_encode_decode(self, q):
        X = np.random.randn(100, q.d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        idx, signs, gammas = q.encode(X)
        assert idx.shape == (100, q.d)
        assert signs.shape == (100, q.d)
        assert gammas.shape == (100, 1) or len(gammas) == 100

        X_tilde = q.decode(idx, signs, gammas)
        assert X_tilde.shape == (100, q.d)

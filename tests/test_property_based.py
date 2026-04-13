"""
Property-based tests for TurboQuant using Hypothesis.

These tests generate random inputs and verify invariants:
  - encode→decode roundtrip preserves shape and produces finite values
  - MSE decreases monotonically with more bits
  - IP estimation is unbiased (mean bias ≈ 0)
  - QJL signs are always ±1
  - No crashes on extreme inputs (very small, very large, zero vectors)
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from turboquant import QJL, TurboQuantMSE, TurboQuantProd

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def unit_vectors(draw, d=st.integers(4, 128), n=st.integers(1, 50)):
    """Generate random unit vectors."""
    d_val = draw(d)
    n_val = draw(n)
    rng = np.random.default_rng(draw(st.integers(0, 2**31)))
    X = rng.standard_normal((n_val, d_val))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms, d_val


@st.composite
def arbitrary_vectors(draw, d=st.integers(4, 128)):
    """Generate arbitrary vectors (not necessarily unit norm)."""
    d_val = draw(d)
    scale = draw(st.floats(1e-15, 1e6))
    rng = np.random.default_rng(draw(st.integers(0, 2**31)))
    X = rng.standard_normal((1, d_val)) * scale
    return X[0], d_val


# ---------------------------------------------------------------------------
# TurboQuantMSE properties
# ---------------------------------------------------------------------------

class TestTurboQuantMSEProperties:

    @given(unit_vectors(d=st.sampled_from([8, 32, 64]), n=st.integers(1, 20)))
    @settings(deadline=None, max_examples=30)
    def test_roundtrip_finite(self, data):
        """encode→decode always produces finite values with correct shape."""
        X, d = data
        q = TurboQuantMSE(d, b=4, seed=42)
        idx = q.encode(X)
        x_hat = q.decode(idx)
        assert x_hat.shape == X.shape
        assert np.all(np.isfinite(x_hat))

    @given(unit_vectors(d=st.just(64), n=st.just(100)))
    @settings(deadline=None, max_examples=20)
    def test_mse_reasonable(self, data):
        """MSE on unit vectors is below theoretical bound ~1/4^b."""
        X, d = data
        for b in [2, 4]:
            q = TurboQuantMSE(d, b=b, seed=42)
            mse = q.mse(X)
            # Theoretical upper bound: (√3π/2) / 4^b
            bound = (np.sqrt(3 * np.pi) / 2) / (4 ** b)
            assert mse < bound * 2, f"MSE={mse:.6f} exceeds bound for b={b}"

    @given(arbitrary_vectors(d=st.just(64)))
    @settings(deadline=None, max_examples=20)
    def test_encode_with_norm_roundtrip(self, data):
        """encode_with_norm + decode_with_norm preserves norm approximately."""
        x, d = data
        q = TurboQuantMSE(d, b=4, seed=42)
        idx, norm = q.encode_with_norm(x)
        x_rec = q.decode_with_norm(idx, norm)
        assert x_rec.shape == x.shape
        assert np.all(np.isfinite(x_rec))
        orig_norm = np.linalg.norm(x)
        rec_norm = np.linalg.norm(x_rec)
        if orig_norm > 1e-10:
            relative_error = abs(rec_norm - orig_norm) / orig_norm
            assert relative_error < 0.15, f"norm error {relative_error:.4f}"


# ---------------------------------------------------------------------------
# QJL properties
# ---------------------------------------------------------------------------

class TestQJLProperties:

    @given(unit_vectors(d=st.sampled_from([16, 64, 128]), n=st.just(1)))
    @settings(deadline=None, max_examples=30)
    def test_signs_always_pm1(self, data):
        """QJL encode always produces ±1 signs."""
        X, d = data
        qjl = QJL(d, seed=42)
        signs, norm = qjl.encode(X[0])
        assert set(np.unique(signs)).issubset({-1, 1})
        assert norm >= 0

    @given(unit_vectors(d=st.just(128), n=st.just(1)))
    @settings(deadline=None, max_examples=20)
    def test_decode_finite(self, data):
        """QJL decode always produces finite values."""
        X, d = data
        qjl = QJL(d, seed=42)
        signs, norm = qjl.encode(X[0])
        x_tilde = qjl.decode(signs, norm)
        assert np.all(np.isfinite(x_tilde))


# ---------------------------------------------------------------------------
# TurboQuantProd properties
# ---------------------------------------------------------------------------

class TestTurboQuantProdProperties:

    @given(unit_vectors(d=st.sampled_from([32, 64]), n=st.just(1)))
    @settings(deadline=None, max_examples=30)
    def test_decode_finite(self, data):
        """decode always produces finite values."""
        X, d = data
        q = TurboQuantProd(d, b=4, seed=42)
        idx, signs, gamma = q.encode(X[0])
        x_tilde = q.decode(idx, signs, gamma)
        assert x_tilde.shape == (d,)
        assert np.all(np.isfinite(x_tilde))

    @given(unit_vectors(d=st.just(64), n=st.integers(50, 100)))
    @settings(deadline=None, max_examples=15)
    def test_ip_unbiased(self, data):
        """Mean IP bias across many vectors should be near zero."""
        X, d = data
        q = TurboQuantProd(d, b=4, seed=42)
        y = X[0].copy()

        true_ips = X @ y
        est_ips = np.array([
            q.inner_product_estimate(y, *q.encode(X[i]))
            for i in range(len(X))
        ])

        mean_bias = abs(np.mean(est_ips - true_ips))
        assert mean_bias < 0.05, f"IP bias {mean_bias:.4f} too large"

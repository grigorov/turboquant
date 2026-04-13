"""
TurboQuant: Near-Optimal Vector Quantization for LLM Inference
==============================================================
Based on: https://arxiv.org/html/2504.19874v1

Two quantization schemes:
  - TurboQuantMSE:  minimizes MSE between original and reconstructed vectors
  - TurboQuantProd: provides unbiased inner-product estimates with minimal variance

Key ideas:
  1. Apply a random rotation (QR decomposition) → coordinates become nearly i.i.d.
  2. Apply per-coordinate Lloyd-Max quantizer optimised for the Beta/Gaussian marginal
  3. For inner products: add a QJL residual term to remove multiplicative bias (~2/π)

Usage
-----
    import numpy as np
    from turboquant import TurboQuantMSE, TurboQuantProd

    d, b = 256, 4           # dimension, bits per coordinate
    x = np.random.randn(d); x /= np.linalg.norm(x)   # unit vector

    # MSE quantization
    q = TurboQuantMSE(d, b, seed=0)
    idx   = q.encode(x)            # (d,) uint16 indices, b bits each
    x_hat = q.decode(idx)          # (d,) float64 reconstruction

    # Inner-product optimised quantization
    p = TurboQuantProd(d, b, seed=0)
    idx, signs, gamma = p.encode(x)
    ip_est = p.inner_product_estimate(y, idx, signs, gamma)   # ≈ x·y, unbiased
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn
from scipy.stats import norm as scipy_norm

# ---------------------------------------------------------------------------
# Helper: Lloyd-Max quantizer for a 1-D distribution
# ---------------------------------------------------------------------------

def _lloyd_max_gaussian(n_centroids: int, sigma: float,
                        max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """
    Lloyd-Max optimal scalar quantizer for X ~ N(0, sigma²).

    Returns sorted centroid array of length n_centroids.
    """
    # Initialise with Gaussian quantiles
    q = np.linspace(0.5 / n_centroids, 1.0 - 0.5 / n_centroids, n_centroids)
    centroids = scipy_norm.ppf(q, scale=sigma)

    for _ in range(max_iter):
        prev = centroids.copy()

        # Decision boundaries: midpoints between adjacent centroids
        boundaries = np.empty(n_centroids + 1)
        boundaries[0]  = -np.inf
        boundaries[-1] =  np.inf
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        # Update each centroid to conditional mean E[X | boundary interval]
        for i in range(n_centroids):
            lo, hi = boundaries[i], boundaries[i + 1]
            phi_lo = scipy_norm.pdf(lo, scale=sigma)
            phi_hi = scipy_norm.pdf(hi, scale=sigma)
            prob   = scipy_norm.cdf(hi, scale=sigma) - scipy_norm.cdf(lo, scale=sigma)
            if prob > 1e-15:
                centroids[i] = sigma ** 2 * (phi_lo - phi_hi) / prob
            else:
                centroids[i] = 0.5 * (lo + hi) if np.isfinite(lo + hi) else 0.0

        if np.max(np.abs(centroids - prev)) < tol:
            break

    return centroids


def _lloyd_max_beta_sphere(n_centroids: int, d: int,
                           max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """
    Lloyd-Max optimal scalar quantizer for the marginal distribution of a
    coordinate of a uniformly random unit vector in ℝ^d.

        f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2),  x ∈ (-1,1)

    For d > 50 delegates to the Gaussian approximation (N(0, 1/d)).
    """
    if d > 50:
        return _lloyd_max_gaussian(n_centroids, sigma=1.0 / np.sqrt(d), max_iter=max_iter, tol=tol)

    coeff = gamma_fn(d / 2) / (np.sqrt(np.pi) * gamma_fn((d - 1) / 2))
    alpha = (d - 3) / 2  # exponent in (1 - x²)^alpha

    def pdf(x: float) -> float:
        v = 1.0 - x * x
        return coeff * v ** alpha if v > 0 else 0.0

    def pdf_x(x: float) -> float:
        return x * pdf(x)

    # Initialise with quantiles of N(0, 1/d)
    sigma = 1.0 / np.sqrt(d)
    q = np.linspace(0.5 / n_centroids, 1.0 - 0.5 / n_centroids, n_centroids)
    centroids = np.clip(scipy_norm.ppf(q, scale=sigma), -0.99, 0.99)

    for _ in range(max_iter):
        prev = centroids.copy()

        boundaries = np.empty(n_centroids + 1)
        boundaries[0]  = -1.0
        boundaries[-1] =  1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        for i in range(n_centroids):
            lo, hi = boundaries[i], boundaries[i + 1]
            prob, _ = quad(pdf,   lo, hi, limit=50)
            mean, _ = quad(pdf_x, lo, hi, limit=50)
            centroids[i] = mean / prob if prob > 1e-15 else 0.5 * (lo + hi)

        if np.max(np.abs(centroids - prev)) < tol:
            break

    return centroids


# ---------------------------------------------------------------------------
# TurboQuantMSE
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """
    MSE-optimised vector quantizer (Algorithm 1 in TurboQuant paper).

    Parameters
    ----------
    d    : vector dimension
    b    : bits per coordinate  (total storage = b·d bits per vector)
    seed : random seed for the rotation matrix

    Assumes unit-norm input vectors. For non-unit vectors, store the L2 norm
    separately and rescale after dequantisation (use ``encode_with_norm``).
    """

    def __init__(self, d: int, b: int, seed: Optional[int] = None) -> None:
        if b < 1 or b > 16:
            raise ValueError(f"b must be in [1, 16], got {b}")

        self.d = d
        self.b = b
        self.n_centroids = 2 ** b

        rng = np.random.default_rng(seed)

        # Random rotation matrix via QR decomposition (uniformly random orthogonal)
        A = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(A)
        # Ensure uniform distribution (correct sign of diagonal of R)
        Q *= np.sign(np.diag(R))
        self.rotation: np.ndarray = Q          # shape (d, d)

        # Optimal codebook for the sphere marginal distribution
        self.centroids: np.ndarray = _lloyd_max_beta_sphere(self.n_centroids, d)

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize vector(s) to centroid indices.

        Parameters
        ----------
        x : (d,) or (n, d) array  — unit-norm vectors

        Returns
        -------
        indices : (d,) or (n, d) uint16 array
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]

        # Rotate: y = Π · x,  shape (n, d)
        y = x @ self.rotation.T

        # Nearest centroid per coordinate: broadcast (n,d,1) vs (1,1,k)
        dist = (y[:, :, None] - self.centroids[None, None, :]) ** 2  # (n, d, k)
        indices = dist.argmin(axis=-1).astype(np.uint16)              # (n, d)

        return indices[0] if squeeze else indices

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Reconstruct vector(s) from centroid indices.

        Parameters
        ----------
        indices : (d,) or (n, d) uint16 array

        Returns
        -------
        x_hat : (d,) or (n, d) float64 array
        """
        squeeze = indices.ndim == 1
        if squeeze:
            indices = indices[None, :]

        # Look up centroids and rotate back: x̃ = Πᵀ · ỹ
        y_hat = self.centroids[indices]        # (n, d)
        x_hat = y_hat @ self.rotation          # (n, d)

        return x_hat[0] if squeeze else x_hat

    # ------------------------------------------------------------------
    # Convenience helpers for non-unit vectors
    # ------------------------------------------------------------------

    def encode_with_norm(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize arbitrary (non-unit) vectors.

        Returns (indices, norms) where norms are float32 scalars per vector.
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]

        norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
        x_unit = x / np.maximum(norms, 1e-12)
        indices = self.encode(x_unit)

        return (indices[0], norms[0]) if squeeze else (indices, norms)

    def decode_with_norm(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """Reconstruct vectors encoded via ``encode_with_norm``."""
        x_unit = self.decode(indices)
        return x_unit * norms

    # ------------------------------------------------------------------
    # Distortion measurement
    # ------------------------------------------------------------------

    def mse(self, x: np.ndarray) -> float:
        """Average per-coordinate MSE on a batch of unit vectors."""
        x_hat = self.decode(self.encode(x))
        return float(np.mean((x - x_hat) ** 2))


# ---------------------------------------------------------------------------
# QJL (Quantized Johnson-Lindenstrauss)
# ---------------------------------------------------------------------------

class QJL:
    """
    Quantized Johnson-Lindenstrauss transform for unbiased inner-product estimation.

        Encode:  z = sign(S · x)            ∈ {-1, +1}^d
        Decode:  x̃ = (√(π/2) / d) · γ · Sᵀ · z

    where γ = ‖x‖₂ is stored as a scalar.

    Property: E[⟨y, x̃⟩] = ⟨y, x⟩  (unbiased for any fixed y)
    Variance:  Var(⟨y, x̃⟩) ≤ (π/2d) · ‖y‖₂²
    """

    def __init__(self, d: int, seed: Optional[int] = None) -> None:
        self.d = d
        rng = np.random.default_rng(seed)
        self.S: np.ndarray = rng.standard_normal((d, d))   # shape (d, d)
        self._scale = np.sqrt(np.pi / 2.0) / d

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        x : (d,) or (n, d) array

        Returns
        -------
        (signs, norms) where signs ∈ {-1,+1}^d and norms = ‖x‖₂
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]

        norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
        z = np.sign(x @ self.S.T)
        z[z == 0] = 1.0   # break ties

        if squeeze:
            return z[0].astype(np.int8), norms[0]
        return z.astype(np.int8), norms

    def decode(self, signs: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        signs : (d,) or (n, d) int8 array  (values ±1)
        norms : scalar or (n,) float array

        Returns
        -------
        x̃ : (d,) or (n, d) float64 array
        """
        squeeze = signs.ndim == 1
        if squeeze:
            signs = signs[None, :]
            norms = np.atleast_1d(norms)

        norms = np.asarray(norms).reshape(-1, 1)
        x_tilde = (signs.astype(np.float64) @ self.S) * (self._scale * norms)

        return x_tilde[0] if squeeze else x_tilde


# ---------------------------------------------------------------------------
# TurboQuantProd
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """
    Inner-product optimised vector quantizer (Algorithm 2 in TurboQuant paper).

    Combines MSE quantisation (b-1 bits) with a QJL residual (1 bit/coord)
    to achieve unbiased inner-product estimation:

        E[⟨y, x̃⟩] = ⟨y, x⟩  for any fixed query y

    Parameters
    ----------
    d    : vector dimension
    b    : total bits per coordinate  (must be ≥ 2)
    seed : random seed
    """

    def __init__(self, d: int, b: int, seed: Optional[int] = None) -> None:
        if b < 2:
            raise ValueError("TurboQuantProd requires b ≥ 2")

        self.d = d
        self.b = b

        rng = np.random.default_rng(seed)
        seed_mse = int(rng.integers(0, 2 ** 31))
        seed_qjl = int(rng.integers(0, 2 ** 31))

        # MSE quantiser with (b-1) bits
        self.mse = TurboQuantMSE(d, b - 1, seed=seed_mse)

        # QJL for the residual
        self.qjl = QJL(d, seed=seed_qjl)

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray,
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quantize vector(s) for inner-product estimation.

        Parameters
        ----------
        x : (d,) or (n, d) array — unit-norm vectors

        Returns
        -------
        (mse_indices, qjl_signs, residual_norms)
          mse_indices    : (d,) or (n, d) uint16
          qjl_signs      : (d,) or (n, d) int8  (values ±1)
          residual_norms : scalar or (n,) float32
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]

        # Step 1: MSE quantisation with (b-1) bits
        mse_idx = self.mse.encode(x)
        x_hat   = self.mse.decode(mse_idx)

        # Step 2: Residual
        residual = x - x_hat

        # Step 3: QJL on residual
        qjl_signs, res_norms = self.qjl.encode(residual)

        if squeeze:
            return mse_idx[0], qjl_signs[0], res_norms[0]
        return mse_idx, qjl_signs, res_norms

    def decode(self, mse_indices: np.ndarray,
               qjl_signs: np.ndarray,
               residual_norms: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximation for inner-product estimation.

            x̃ = x̂_mse + (√(π/2)/d) · ‖r‖₂ · Sᵀ · z

        Parameters
        ----------
        mse_indices    : (d,) or (n, d) uint16
        qjl_signs      : (d,) or (n, d) int8
        residual_norms : scalar or (n,) float32

        Returns
        -------
        x̃ : (d,) or (n, d) float64
        """
        squeeze = mse_indices.ndim == 1
        if squeeze:
            mse_indices    = mse_indices[None, :]
            qjl_signs      = qjl_signs[None, :]
            residual_norms = np.atleast_1d(residual_norms)

        x_hat_mse    = self.mse.decode(mse_indices)
        residual_hat = self.qjl.decode(qjl_signs, residual_norms)

        result = x_hat_mse + residual_hat

        return result[0] if squeeze else result

    def inner_product_estimate(self, y: np.ndarray,
                               mse_indices: np.ndarray,
                               qjl_signs: np.ndarray,
                               residual_norms: np.ndarray) -> float:
        """
        Compute unbiased inner-product estimate ⟨y, x⟩ from quantised x.

        Parameters
        ----------
        y              : (d,) query vector
        mse_indices    : (d,) uint16
        qjl_signs      : (d,) int8
        residual_norms : scalar float32

        Returns
        -------
        float estimate of ⟨y, x⟩
        """
        x_tilde = self.decode(mse_indices, qjl_signs, residual_norms)
        return float(np.dot(y, x_tilde))

    # ------------------------------------------------------------------
    # Storage information
    # ------------------------------------------------------------------

    def bits_per_vector(self) -> int:
        """Total bits used to store one quantised vector."""
        # (b-1) bits × d coords for MSE part + 1 bit × d coords for QJL
        # + 32 bits for the residual norm float
        return (self.b - 1) * self.d + self.d + 32


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    import time

    np.random.seed(42)
    d   = 256
    n   = 1000
    rng = np.random.default_rng(42)

    # Random unit vectors
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    print("=" * 60)
    print(f"TurboQuant demo  |  d={d}  n={n}")
    print("=" * 60)

    # ------------------------------------------------------------------ MSE
    print("\n--- TurboQuantMSE ---")
    for b in [1, 2, 4, 8]:
        t0 = time.perf_counter()
        q   = TurboQuantMSE(d, b, seed=0)
        t_setup = time.perf_counter() - t0

        t0 = time.perf_counter()
        idx   = q.encode(X)
        x_hat = q.decode(idx)
        t_enc = time.perf_counter() - t0

        mse = float(np.mean((X - x_hat) ** 2))
        print(f"  b={b:2d}  MSE={mse:.5f}  setup={t_setup:.2f}s  enc+dec={t_enc*1000:.1f}ms")

    # ------------------------------------------------------------------ Prod
    print("\n--- TurboQuantProd (inner product) ---")
    y = rng.standard_normal(d)
    y /= np.linalg.norm(y)

    true_ips = X @ y   # (n,) true inner products

    for b in [2, 4, 8]:
        q = TurboQuantProd(d, b, seed=0)

        mse_idx, qjl_signs, res_norms = q.encode(X)
        est_ips = np.array([
            q.inner_product_estimate(y, mse_idx[i], qjl_signs[i], res_norms[i])
            for i in range(n)
        ])

        bias = float(np.mean(est_ips - true_ips))
        rmse = float(np.sqrt(np.mean((est_ips - true_ips) ** 2)))
        print(f"  b={b:2d}  IP bias={bias:+.5f}  RMSE={rmse:.5f}"
              f"  bits/vec={q.bits_per_vector()}")

    # ------------------------------------------------------------------ Unbiasedness check
    print("\n--- Unbiasedness verification (b=4, n=5000) ---")
    n2  = 5000
    X2  = rng.standard_normal((n2, d))
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True)

    q   = TurboQuantProd(d, 4, seed=0)
    mse_idx, qjl_signs, res_norms = q.encode(X2)

    true_ips2 = X2 @ y
    est_ips2  = np.array([
        q.inner_product_estimate(y, mse_idx[i], qjl_signs[i], res_norms[i])
        for i in range(n2)
    ])
    bias2 = float(np.mean(est_ips2 - true_ips2))
    print(f"  Mean bias over {n2} vectors: {bias2:+.6f}  (should be ≈ 0)")

    print("\nDone.")


if __name__ == "__main__":
    _demo()

"""
TurboQuant Numba JIT-accelerated encode/decode.

Provides drop-in replacements for TurboQuantMSE encode/decode that use
Numba's just-in-time compilation for faster batch processing.

Usage
-----
    from turboquant_numba import TurboQuantMSEJIT

    q = TurboQuantMSEJIT(d=256, b=4, seed=42)
    idx = q.encode(X)   # JIT-compiled batch encode
    x_hat = q.decode(idx)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from turboquant import TurboQuantMSE

# Try to import numba; fall back gracefully if not available
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    # Dummy decorators for fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(n):
        return range(n)


# ---------------------------------------------------------------------------
# JIT-compiled kernels
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _encode_kernel(
        x: np.ndarray,       # (n, d)
        rotation: np.ndarray,  # (d, d)
        centroids: np.ndarray,  # (k,)
        d: int,
        k: int,
    ) -> np.ndarray:
        """JIT-parallel encode: nearest centroid per coordinate."""
        n = x.shape[0]
        indices = np.empty((n, d), dtype=np.uint16)

        for i in prange(n):
            # y = rotation @ x[i]
            y = np.empty(d, dtype=np.float64)
            for r in range(d):
                s = 0.0
                for c in range(d):
                    s += rotation[r, c] * x[i, c]
                y[r] = s

            # Nearest centroid per coordinate
            for j in range(d):
                best_idx = 0
                best_dist = (y[j] - centroids[0]) ** 2
                for c in range(1, k):
                    dist = (y[j] - centroids[c]) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = c
                indices[i, j] = np.uint16(best_idx)

        return indices

    @njit(parallel=True, cache=True)
    def _decode_kernel(
        indices: np.ndarray,   # (n, d)
        centroids: np.ndarray,  # (k,)
        rotation: np.ndarray,   # (d, d)
        d: int,
    ) -> np.ndarray:
        """JIT-parallel decode: lookup centroids + inverse rotation."""
        n = indices.shape[0]
        result = np.empty((n, d), dtype=np.float64)

        for i in prange(n):
            # Lookup centroids
            y_hat = np.empty(d, dtype=np.float64)
            for j in range(d):
                y_hat[j] = centroids[indices[i, j]]

            # x_hat = rotation.T @ y_hat
            for c in range(d):
                s = 0.0
                for r in range(d):
                    s += rotation[r, c] * y_hat[r]
                result[i, c] = s

        return result


# ---------------------------------------------------------------------------
# JIT-accelerated TurboQuantMSE
# ---------------------------------------------------------------------------

class TurboQuantMSEJIT(TurboQuantMSE):
    """
    TurboQuantMSE with Numba JIT-compiled encode/decode for faster batching.

    Falls back to the parent NumPy implementation if Numba is not installed.
    """

    def __init__(self, d: int, b: int, seed: Optional[int] = None) -> None:
        super().__init__(d, b, seed)

        if not _HAS_NUMBA:
            import warnings
            warnings.warn(
                "Numba not available. Falling back to NumPy encode/decode. "
                "Install numba for JIT acceleration: pip install numba",
                UserWarning,
            )

    def encode(self, x: np.ndarray) -> np.ndarray:
        """JIT-accelerated batch encode."""
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]

        if _HAS_NUMBA:
            indices = _encode_kernel(
                x, self.rotation, self.centroids, self.d, self.n_centroids
            )
        else:
            indices = super().encode(x)

        return indices[0] if squeeze else indices

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """JIT-accelerated batch decode."""
        squeeze = indices.ndim == 1
        if squeeze:
            indices = indices[None, :]

        if _HAS_NUMBA:
            result = _decode_kernel(
                indices, self.centroids, self.rotation, self.d
            )
        else:
            result = super().decode(indices)

        return result[0] if squeeze else result

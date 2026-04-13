"""
TurboQuant Sparse Vector Quantization.

Efficient quantization for sparse vectors (vectors with many zero coordinates).
Instead of quantizing all coordinates, only non-zero coordinates are encoded,
saving significant storage and compute for sparse data.

Usage
-----
    from turboquant_sparse import SparseQuantizer

    # Create a sparse vector (90% zeros)
    import numpy as np
    x = np.zeros(4096)
    nonzero_idx = np.random.choice(4096, size=410, replace=False)
    x[nonzero_idx] = np.random.randn(410)

    # Quantize
    q = SparseQuantizer(d=4096, b=4, density_threshold=1e-6, seed=42)
    sparse_encoding = q.encode(x)
    x_hat = q.decode(sparse_encoding)

    # Batch mode
    X = np.zeros((100, 4096))
    for i in range(100):
        idx = np.random.choice(4096, size=400, replace=False)
        X[i, idx] = np.random.randn(400)
    enc_batch = q.encode_batch(X)
    X_hat = q.decode_batch(enc_batch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse

from turboquant import TurboQuantMSE, TurboQuantProd


@dataclass
class SparseEncoding:
    """
    Encoded representation of a sparse vector.

    Attributes
    ----------
    indices : ndarray of int
        Indices of non-zero coordinates.
    values : ndarray of float
        Original values at non-zero coordinates.
    quantized_indices : ndarray of uint16
        Quantized indices (from TurboQuantMSE on the non-zero sub-vector).
    norm : float32
        L2 norm of the non-zero sub-vector (for non-unit sparse vectors).
    """
    indices: np.ndarray
    values: np.ndarray
    quantized_indices: np.ndarray
    norm: np.float32


class SparseQuantizer:
    """
    Quantizer optimized for sparse vectors.

    For a sparse vector x with nnz non-zero elements:
      1. Extract non-zero values and their indices.
      2. Quantize the non-zero sub-vector using TurboQuantMSE/Prod with
         dimension = nnz (not d), saving (d - nnz) * b bits.
      3. Store indices as uint32 (or delta-encoded for further savings).

    This is especially effective for:
      - MoE gating vectors
      - Sparse attention patterns
      - Pruned neural network weights
      - One-hot / multi-hot encodings

    Parameters
    ----------
    d : int
        Full vector dimension (including zeros).
    b : int
        Bits per non-zero coordinate.
    density_threshold : float
        Values with absolute magnitude below this are treated as zero.
    prod : bool
        Use TurboQuantProd for non-zero sub-vectors (unbiased IP).
        Only applicable when expected nnz >= 2.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        d: int,
        b: int,
        density_threshold: float = 1e-6,
        prod: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.d = d
        self.b = b
        self.density_threshold = density_threshold
        self.prod = prod
        self.seed = seed

        # We create quantizers lazily per expected nnz to avoid
        # creating 4096 different quantizers.
        self._cache: dict[int, object] = {}

    def _get_quantizer(self, nnz: int):
        """Get or create a quantizer for a sub-vector of size nnz."""
        if nnz in self._cache:
            return self._cache[nnz]

        if nnz == 0:
            q = None
        elif nnz == 1:
            q = TurboQuantMSE(1, self.b, seed=self.seed)
        elif self.prod and nnz >= 2:
            q = TurboQuantProd(nnz, self.b, seed=self.seed)
        else:
            q = TurboQuantMSE(nnz, self.b, seed=self.seed)

        self._cache[nnz] = q
        return q

    # ------------------------------------------------------------------
    # Single vector encode/decode
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> SparseEncoding:
        """
        Encode a sparse vector.

        Parameters
        ----------
        x : (d,) array
            Input vector (may contain many zeros).

        Returns
        -------
        encoding : SparseEncoding
        """
        assert x.shape[0] == self.d, f"Expected dim {self.d}, got {x.shape[0]}"

        # Find non-zero coordinates
        mask = np.abs(x) > self.density_threshold
        indices = np.flatnonzero(mask).astype(np.uint32)
        values = x[mask]
        nnz = len(indices)

        if nnz == 0:
            return SparseEncoding(
                indices=np.array([], dtype=np.uint32),
                values=np.array([], dtype=np.float64),
                quantized_indices=np.array([], dtype=np.uint16),
                norm=np.float32(0.0),
            )

        # Quantize the non-zero sub-vector
        q = self._get_quantizer(nnz)
        norm = np.float32(np.linalg.norm(values))

        if norm > 1e-12:
            values_unit = values / norm
        else:
            values_unit = values

        if isinstance(q, TurboQuantProd):
            quantized = q.encode(values_unit)
            # quantized is (mse_idx, signs, residual_norm)
            # Pack into a single uint16 array for storage
            mse_idx, signs, res_norm = quantized
            # Encode: mse_idx (uint16) + signs as packed bits
            packed = np.concatenate([
                mse_idx.astype(np.uint16),
                ((signs + 1) // 2).astype(np.uint16),  # -1,1 -> 0,1
            ])
            quantized_indices = packed
        else:
            quantized_indices = q.encode(values_unit)

        return SparseEncoding(
            indices=indices,
            values=values,
            quantized_indices=quantized_indices,
            norm=norm,
        )

    def decode(self, encoding: SparseEncoding) -> np.ndarray:
        """
        Decode a sparse encoding back to a full vector.

        Parameters
        ----------
        encoding : SparseEncoding

        Returns
        -------
        x_hat : (d,) array
            Reconstructed vector.
        """
        x_hat = np.zeros(self.d, dtype=np.float64)
        nnz = len(encoding.indices)

        if nnz == 0:
            return x_hat

        q = self._get_quantizer(nnz)

        if isinstance(q, TurboQuantProd):
            # Unpack
            half = len(encoding.quantized_indices) // 2
            mse_idx = encoding.quantized_indices[:half]
            signs_packed = encoding.quantized_indices[half:]
            signs = (signs_packed.astype(np.int8) * 2 - 1).astype(np.int8)
            res_norm = encoding.norm  # approximate; stored in norm field

            # For Prod, we need the residual norm. Since we store the full
            # values, we can reconstruct more accurately.
            values_hat = q.decode(mse_idx, signs, np.float32(1.0))
        else:
            values_hat = q.decode(encoding.quantized_indices)

        x_hat[encoding.indices] = values_hat * encoding.norm
        return x_hat

    # ------------------------------------------------------------------
    # Batch encode/decode
    # ------------------------------------------------------------------

    def encode_batch(self, X: np.ndarray) -> list[SparseEncoding]:
        """Encode a batch of sparse vectors."""
        return [self.encode(X[i]) for i in range(X.shape[0])]

    def decode_batch(self, encodings: list[SparseEncoding]) -> np.ndarray:
        """Decode a batch of sparse encodings."""
        return np.array([self.decode(enc) for enc in encodings])

    # ------------------------------------------------------------------
    # Sparse matrix (scipy) support
    # ------------------------------------------------------------------

    def encode_sparse_matrix(
        self,
        X: sparse.spmatrix,
        axis: int = 0,
    ) -> list[SparseEncoding]:
        """
        Encode a scipy sparse matrix row by row.

        Parameters
        ----------
        X : scipy sparse matrix
            Input sparse matrix (CSR format recommended).
        axis : int
            Axis to encode along (0 = rows, 1 = columns).

        Returns
        -------
        encodings : list of SparseEncoding
        """
        if axis == 1:
            X = X.T

        X_csr = X.tocsr()
        encodings = []
        for i in range(X_csr.shape[0]):
            row = X_csr[i].toarray().ravel()
            encodings.append(self.encode(row))
        return encodings

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def bits_for_encoding(self, encoding: SparseEncoding) -> int:
        """
        Compute the number of bits required for a sparse encoding.

        Bits = nnz * 32 (indices) + len(quantized_indices) * b + 32 (norm)
        """
        nnz = len(encoding.indices)
        q_bits = len(encoding.quantized_indices) * self.b
        index_bits = nnz * 32  # uint32 indices
        norm_bits = 32
        return q_bits + index_bits + norm_bits

    def compression_vs_dense(self, encoding: SparseEncoding) -> float:
        """
        Compression ratio vs dense quantization of the full vector.

        Returns dense_bits / sparse_bits.
        """
        dense_bits = self.d * self.b
        sparse_bits = self.bits_for_encoding(encoding)
        if sparse_bits == 0:
            return float("inf")
        return dense_bits / sparse_bits

    def sparsity(self, x: np.ndarray) -> float:
        """Fraction of zero coordinates."""
        return float(np.mean(np.abs(x) <= self.density_threshold))

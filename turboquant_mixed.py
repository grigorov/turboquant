"""
TurboQuant Mixed-Precision Quantization.

Supports adaptive bit allocation across layers or coordinate groups,
enabling different bit-widths for different parts of a vector or
different layers of a neural network.

Usage
-----
    from turboquant_mixed import MixedPrecisionQuantizer

    # Strategy 1: per-layer bit allocation
    q = MixedPrecisionQuantizer(
        layer_dims=[4096, 4096, 4096],   # 3 layers of different sizes
        layer_bits=[4, 2, 8],             # 4-bit, 2-bit, 8-bit
        seed=42,
    )
    idx = q.encode(layer_weights)

    # Strategy 2: per-group bit allocation (split vector into groups)
    q = MixedPrecisionQuantizer.from_groups(
        d=4096,
        group_sizes=[2048, 1024, 1024],   # split vector into 3 groups
        group_bits=[4, 2, 8],              # each group gets different bits
        seed=42,
    )
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from turboquant import TurboQuantMSE, TurboQuantProd


class MixedPrecisionConfig:
    """
    Configuration for mixed-precision quantization.

    Two modes:
      - "layer": different layers (sub-quantizers) handle different vectors
      - "group": a single vector is split into groups, each quantized at different bits
    """

    mode: str  # "layer" or "group"
    sizes: list[int]     # layer dims (mode="layer") or group sizes (mode="group")
    bits: list[int]      # bits per layer/group
    seeds: list[int]     # seed per layer/group

    def __init__(
        self,
        mode: str,
        sizes: Sequence[int],
        bits: Sequence[int],
        seed: Optional[int] = None,
    ) -> None:
        if mode not in ("layer", "group"):
            raise ValueError(f"mode must be 'layer' or 'group', got '{mode}'")
        if len(sizes) != len(bits):
            raise ValueError("sizes and bits must have the same length")
        for b in bits:
            if b < 1 or b > 16:
                raise ValueError(f"bits must be in [1, 16], got {b}")

        self.mode = mode
        self.sizes = list(sizes)
        self.bits = list(bits)

        rng = np.random.default_rng(seed)
        self.seeds = [int(rng.integers(0, 2**31)) for _ in range(len(self.sizes))]

    @property
    def total_bits(self) -> int:
        """Total bits per (vector or layer-set)."""
        return sum(s * b for s, b in zip(self.sizes, self.bits))


class MixedPrecisionQuantizer:
    """
    Mixed-precision vector quantizer.

    Supports two strategies:

    **Layer mode** — multiple sub-quantizers, each handling vectors of a
    specific dimension at a specific bit-width. Useful for quantizing
    different layers of a neural network at different precisions.

    **Group mode** — a single vector is partitioned into contiguous groups
    of coordinates, each group quantized at a different bit-width. Useful
    for allocating more bits to "important" coordinates (e.g., high-magnitude
    weights or attention heads).

    Parameters
    ----------
    config : MixedPrecisionConfig
        Quantization configuration.
    prod : bool
        If True, use TurboQuantProd (unbiased IP) for each sub-quantizer.
        If False, use TurboQuantMSE (MSE-optimized).
    """

    def __init__(self, config: MixedPrecisionConfig, prod: bool = False) -> None:
        self.config = config

        if prod:
            self.quantizers: list = [
                TurboQuantMSE(size, bits, seed=seed)
                if bits == 1
                else TurboQuantProd(size, bits, seed=seed)
                for size, bits, seed in zip(config.sizes, config.bits, config.seeds)
            ]
        else:
            self.quantizers = [
                TurboQuantMSE(size, bits, seed=seed)
                for size, bits, seed in zip(config.sizes, config.bits, config.seeds)
            ]

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_layers(
        cls,
        layer_dims: Sequence[int],
        layer_bits: Sequence[int],
        seed: Optional[int] = None,
        prod: bool = False,
    ) -> "MixedPrecisionQuantizer":
        """
        Create a layer-mode quantizer.

        Each layer has its own dimension and bit-width.

        Parameters
        ----------
        layer_dims : list of int
            Dimension of each layer (e.g., [4096, 1024, 4096]).
        layer_bits : list of int
            Bit-width for each layer (e.g., [4, 2, 8]).
        seed : int or None
            Master seed for reproducibility.
        prod : bool
            Use TurboQuantProd for layers with b >= 2.
        """
        config = MixedPrecisionConfig("layer", layer_dims, layer_bits, seed)
        return cls(config, prod=prod)

    @classmethod
    def from_groups(
        cls,
        d: int,
        group_sizes: Sequence[int],
        group_bits: Sequence[int],
        seed: Optional[int] = None,
        prod: bool = False,
    ) -> "MixedPrecisionQuantizer":
        """
        Create a group-mode quantizer.

        A vector of dimension `d` is split into contiguous groups.

        Parameters
        ----------
        d : int
            Total vector dimension.
        group_sizes : list of int
            Size of each group. Must sum to `d`.
        group_bits : list of int
            Bit-width for each group.
        seed : int or None
            Master seed for reproducibility.
        prod : bool
            Use TurboQuantProd for groups with b >= 2.
        """
        if sum(group_sizes) != d:
            raise ValueError(
                f"group_sizes must sum to d={d}, got {sum(group_sizes)}"
            )
        config = MixedPrecisionConfig("group", group_sizes, group_bits, seed)
        return cls(config, prod=prod)

    @classmethod
    def from_importance(
        cls,
        d: int,
        n_groups: int = 4,
        bits_range: tuple[int, int] = (2, 8),
        seed: Optional[int] = None,
        prod: bool = False,
    ) -> "MixedPrecisionQuantizer":
        """
        Create a group-mode quantizer with equal-sized groups.

        The vector is split into `n_groups` equal parts (or nearly equal),
        each quantized at a different bit-width from low to high.
        Useful when you want to experiment with importance-based allocation
        without knowing the exact importance map.

        Parameters
        ----------
        d : int
            Vector dimension.
        n_groups : int
            Number of groups (default 4).
        bits_range : (min_bits, max_bits)
            Bit-width range across groups.
        seed : int or None
            Master seed.
        prod : bool
            Use TurboQuantProd for groups with b >= 2.
        """
        lo, hi = bits_range
        group_bits = list(np.linspace(lo, hi, n_groups).round().astype(int))
        base_size = d // n_groups
        remainder = d % n_groups
        group_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_groups)]
        return cls.from_groups(d, group_sizes, group_bits, seed=seed, prod=prod)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode_layer_batch(
        self,
        layer_data: Sequence[np.ndarray],
    ) -> list:
        """
        Encode a list of layer weight matrices.

        Each layer is flattened to 1D and quantized independently.

        Parameters
        ----------
        layer_data : list of arrays
            Each array is a layer weight tensor (any shape).
            Must have the same number of elements as layer_dims.

        Returns
        -------
        encoded : list
            One encoded result per layer (indices or (idx, signs, norm) tuples).
        """
        if len(layer_data) != len(self.config.sizes):
            raise ValueError(
                f"Expected {len(self.config.sizes)} layers, got {len(layer_data)}"
            )

        results = []
        for i, (layer, q) in enumerate(zip(layer_data, self.quantizers)):
            flat = layer.ravel()
            n = flat.shape[0]
            if n != self.config.sizes[i]:
                raise ValueError(
                    f"Layer {i}: expected {self.config.sizes[i]} elements, got {n}"
                )

            # Normalize to unit norm
            norm = float(np.linalg.norm(flat))
            if norm > 1e-12:
                flat_unit = flat / norm
            else:
                flat_unit = flat

            result = q.encode(flat_unit)
            results.append((result, np.float32(norm)))

        return results

    def decode_layer_batch(
        self,
        encoded: list,
        original_shapes: Sequence[tuple],
    ) -> list[np.ndarray]:
        """
        Decode layer quantized results back to original shapes.

        Parameters
        ----------
        encoded : list
            Output of ``encode_layer_batch``.
        original_shapes : list of tuples
            Original shape of each layer.

        Returns
        -------
        layers : list of arrays
            Reconstructed layer weight tensors.
        """
        results = []
        for i, ((enc_data, norm), shape) in enumerate(zip(encoded, original_shapes)):
            q = self.quantizers[i]

            if isinstance(enc_data, tuple):
                # TurboQuantProd: (idx, signs, gamma)
                x_flat = q.decode(enc_data[0], enc_data[1], enc_data[2])
            else:
                # TurboQuantMSE: just indices
                x_flat = q.decode(enc_data)

            x_flat = x_flat * norm
            results.append(x_flat.reshape(shape))

        return results

    def encode_group(self, x: np.ndarray) -> list:
        """
        Encode a single vector using group-mode quantization.

        Parameters
        ----------
        x : (d,) array
            Input vector.

        Returns
        -------
        encoded : list
            One result per group.
        """
        if x.shape[0] != sum(self.config.sizes):
            raise ValueError(
                f"Expected vector of dim {sum(self.config.sizes)}, got {x.shape[0]}"
            )

        results = []
        offset = 0
        for i, (q, size) in enumerate(zip(self.quantizers, self.config.sizes)):
            group = x[offset : offset + size]
            norm = float(np.linalg.norm(group))
            if norm > 1e-12:
                group_unit = group / norm
            else:
                group_unit = group

            result = q.encode(group_unit)
            results.append((result, np.float32(norm)))
            offset += size

        return results

    def decode_group(self, encoded: list) -> np.ndarray:
        """
        Decode group-mode encoded result.

        Parameters
        ----------
        encoded : list
            Output of ``encode_group``.

        Returns
        -------
        x_hat : (d,) array
            Reconstructed vector.
        """
        parts = []
        for i, (enc_data, norm) in enumerate(encoded):
            q = self.quantizers[i]

            if isinstance(enc_data, tuple):
                x_part = q.decode(enc_data[0], enc_data[1], enc_data[2])
            else:
                x_part = q.decode(enc_data)

            parts.append(x_part * norm)

        return np.concatenate(parts)

    def encode_group_batch(
        self,
        X: np.ndarray,
    ) -> list:
        """
        Encode a batch of vectors using group-mode.

        Parameters
        ----------
        X : (n, d) array
            Input vectors.

        Returns
        -------
        encoded : list of lists
            encoded[i] is the encoding for X[i].
        """
        return [self.encode_group(X[i]) for i in range(X.shape[0])]

    def decode_group_batch(
        self,
        encoded_batch: list,
    ) -> np.ndarray:
        """
        Decode a batch of group-encoded vectors.

        Parameters
        ----------
        encoded_batch : list of lists
            Output of ``encode_group_batch``.

        Returns
        -------
        X_hat : (n, d) array
        """
        return np.array([self.decode_group(enc) for enc in encoded_batch])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def total_bits_per_vector(self) -> int:
        """Total bits for one quantized vector (group mode)."""
        if self.config.mode == "group":
            return sum(s * b for s, b in zip(self.config.sizes, self.config.bits))
        else:
            raise ValueError("total_bits_per_vector only makes sense in group mode")

    def avg_bits_per_coordinate(self) -> float:
        """Average bits per coordinate across all groups."""
        total = sum(s * b for s, b in zip(self.config.sizes, self.config.bits))
        d = sum(self.config.sizes)
        return total / d

    def compression_ratio(self) -> float:
        """Compression ratio vs float64."""
        d = sum(self.config.sizes)
        raw_bits = d * 64
        return raw_bits / self.total_bits_per_vector()

    def bit_allocation_summary(self) -> str:
        """Human-readable summary of bit allocation."""
        label = "Group" if self.config.mode == "group" else "Layer"
        lines = [f"Mode: {self.config.mode}"]
        for i, (s, b) in enumerate(zip(self.config.sizes, self.config.bits)):
            lines.append(f"  {label} {i}: size={s:>6d}, bits={b:>2d} ({s*b:>6d} bits total)")
        if self.config.mode == "group":
            lines.append(f"  Total bits: {self.total_bits_per_vector()}")
            lines.append(f"  Avg bits/coord: {self.avg_bits_per_coordinate():.2f}")
            lines.append(f"  Compression: {self.compression_ratio():.1f}x")
        else:
            total = self.total_bits_all_layers()
            lines.append(f"  Total bits (all layers): {total}")
        return "\n".join(lines)

    def total_bits_all_layers(self) -> int:
        """Total bits across all layers (layer mode)."""
        return sum(s * b for s, b in zip(self.config.sizes, self.config.bits))

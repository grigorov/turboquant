"""
TurboQuant LLM Integration Backend.

Provides adapters for quantizing and serving LLM weights through
popular inference frameworks (llama.cpp, vLLM) with a unified API.

Usage
-----
    from turboquant_llm import TurboQuantBackend

    # llama.cpp-style GGUF quantization
    backend = TurboQuantBackend.create("gguf")
    q = backend.quantizer(d=4096, b=4, seed=42)

    # vLLM-style KV-cache quantization
    backend = TurboQuantBackend.create("vllm")
    q = backend.quantizer(d=4096, b=4, seed=42)

    # Quantize a weight matrix
    W = ...  # numpy array from HuggingFace model
    encoded = q.encode(W)
    W_rec = q.decode(encoded)
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from turboquant import TurboQuantMSE, TurboQuantProd


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Abstract interface for LLM framework integration."""

    @abstractmethod
    def quantizer(self, d: int, b: int, seed: Optional[int] = None):
        """Create a quantizer compatible with this backend."""

    @abstractmethod
    def encode_weight(self, weight: np.ndarray, quantizer) -> bytes:
        """Encode a weight tensor to the backend's binary format."""

    @abstractmethod
    def decode_weight(self, data: bytes, d: int, b: int, quantizer) -> np.ndarray:
        """Decode a backend binary blob back to a weight tensor."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""


# ---------------------------------------------------------------------------
# GGUF / llama.cpp-style backend
# ---------------------------------------------------------------------------

class GGUFSimpleBackend(LLMBackend):
    """
    Simplified GGUF-like binary format for llama.cpp integration.

    Format per tensor:
        [header: 16 bytes]
          - magic (4 bytes): b'TQGG'
          - version (4 bytes): uint32
          - n_dims (2 bytes): uint16
          - d (2 bytes): uint16  (first dimension)
          - n_elements (4 bytes): uint32
        [data: variable]
          - quantization data
    """

    MAGIC = b"TQGG"
    VERSION = 1

    def _chunked_quantize(self, flat: np.ndarray, d: int, b: int, seed: int, quantizer_factory):
        """
        Quantize a potentially large vector by chunking.

        For very large dimensions, we split into chunks of size `chunk_d`
        and quantize each chunk independently.
        """
        chunk_d = min(d, 4096)  # Max chunk dimension
        n_chunks = (d + chunk_d - 1) // chunk_d

        if n_chunks == 1:
            q = quantizer_factory(d, b, seed=seed)
            norm = float(np.linalg.norm(flat))
            if norm > 1e-12:
                flat_unit = flat / norm
            else:
                flat_unit = flat
            encoded = q.encode(flat_unit)
            return encoded, norm

        # Chunked quantization
        all_encoded = []
        norms = []
        offset = 0
        for i in range(n_chunks):
            chunk_size = min(chunk_d, d - offset)
            chunk = flat[offset : offset + chunk_size]
            chunk_seed = seed + i

            q = quantizer_factory(chunk_size, b, seed=chunk_seed)
            norm = float(np.linalg.norm(chunk))
            if norm > 1e-12:
                chunk_unit = chunk / norm
            else:
                chunk_unit = chunk

            encoded = q.encode(chunk_unit)
            all_encoded.append((encoded, np.float32(norm)))
            offset += chunk_size

        return all_encoded, None

    def _chunked_dequantize(self, encoded, d: int, b: int, seed: int, quantizer_factory):
        """Dequantize a chunked result."""
        if isinstance(encoded, list):
            # Chunked case
            parts = []
            offset = 0
            for i, (enc_data, norm) in enumerate(encoded):
                chunk_size_norm = norm
                chunk_seed = seed + i

                # Infer chunk size from encoded data
                if isinstance(enc_data, tuple):
                    chunk_size = len(enc_data[0])
                    q = quantizer_factory(chunk_size, b, seed=chunk_seed)
                    part = q.decode(enc_data[0], enc_data[1], np.float32(enc_data[2]))
                else:
                    chunk_size = len(enc_data)
                    q = quantizer_factory(chunk_size, b, seed=chunk_seed)
                    part = q.decode(enc_data)

                parts.append(part * chunk_size_norm)
                offset += chunk_size

            return np.concatenate(parts)
        else:
            # Single quantizer case
            if isinstance(encoded, tuple):
                q = quantizer_factory(d, b, seed=seed)
                return q.decode(encoded[0], encoded[1], np.float32(encoded[2]))
            else:
                q = quantizer_factory(d, b, seed=seed)
                return q.decode(encoded)

    def quantizer(self, d: int, b: int, seed: Optional[int] = None):
        if b >= 2:
            return TurboQuantProd(d, b, seed=seed)
        return TurboQuantMSE(d, b, seed=seed)

    def encode_weight(self, weight: np.ndarray, quantizer) -> bytes:
        flat = weight.ravel().astype(np.float64)
        d = flat.shape[0]

        encoded, _ = self._encode_flat(flat, d, quantizer)

        header = struct.pack(
            "<4sIHHI",
            self.MAGIC,
            self.VERSION,
            1,
            min(d, 65535),
            d,
        )
        return header + encoded

    def _encode_flat(self, flat: np.ndarray, d: int, quantizer):
        """Encode a flat array, possibly in chunks. Returns (data_bytes, norm_or_None)."""
        chunk_d = min(d, 4096)
        n_chunks = (d + chunk_d - 1) // chunk_d

        if n_chunks == 1:
            norm = float(np.linalg.norm(flat))
            if norm > 1e-12:
                flat_unit = flat / norm
            else:
                flat_unit = flat

            encoded = quantizer.encode(flat_unit)
            return self._pack_encoded(encoded, norm), None
        else:
            data = b""
            offset = 0
            seed = 42  # default seed for chunks
            for i in range(n_chunks):
                chunk_size = min(chunk_d, d - offset)
                chunk = flat[offset : offset + chunk_size]
                chunk_seed = seed + i

                # Create chunk quantizer
                if hasattr(quantizer, 'prod') or isinstance(quantizer, TurboQuantProd):
                    q = TurboQuantProd(chunk_size, quantizer.b, seed=chunk_seed) if hasattr(quantizer, 'b') else TurboQuantMSE(chunk_size, 4, seed=chunk_seed)
                else:
                    q = TurboQuantMSE(chunk_size, quantizer.b, seed=chunk_seed)

                norm = float(np.linalg.norm(chunk))
                if norm > 1e-12:
                    chunk_unit = chunk / norm
                else:
                    chunk_unit = chunk

                encoded = q.encode(chunk_unit)
                chunk_data = self._pack_encoded(encoded, norm)
                data += struct.pack("<I", len(chunk_data))
                data += chunk_data
                offset += chunk_size

            return data, "chunked"

    def _pack_encoded(self, encoded, norm: float) -> bytes:
        """Pack encoded result into bytes."""
        if isinstance(encoded, tuple):
            mse_idx, signs, res_norm = encoded
            data = struct.pack("<I", len(mse_idx))
            data += mse_idx.tobytes()
            data += struct.pack("<I", len(signs))
            data += signs.tobytes()
            data += struct.pack("<f", float(np.asarray(res_norm).item()))
        else:
            data = struct.pack("<I", len(encoded))
            data += encoded.tobytes()
        data += struct.pack("<f", float(norm))
        return data

    def decode_weight(self, data: bytes, d: int, b: int, quantizer) -> np.ndarray:
        header_size = 16
        header = data[:header_size]
        magic, version, n_dims, dim0, n_elements = struct.unpack(
            "<4sIHHI", header
        )
        assert magic == self.MAGIC, f"Invalid magic: {magic}"

        payload = data[header_size:]
        return self._decode_flat(payload, n_elements, b, quantizer)

    def _decode_flat(self, payload: bytes, d: int, b: int, quantizer) -> np.ndarray:
        """Decode a flat array, possibly from chunks."""
        chunk_d = min(d, 4096)
        n_chunks = (d + chunk_d - 1) // chunk_d

        if n_chunks == 1:
            return self._unpack_and_decode(payload, 0, d, b, quantizer)
        else:
            parts = []
            pos = 0
            seed = 42
            for i in range(n_chunks):
                chunk_size = min(chunk_d, d - len(np.concatenate(parts)) if parts else d)
                chunk_data_len = struct.unpack_from("<I", payload, pos)[0]
                pos += 4
                chunk_data = payload[pos : pos + chunk_data_len]
                pos += chunk_data_len

                chunk_seed = seed + i
                if hasattr(quantizer, 'b'):
                    q_cls = TurboQuantProd if b >= 2 else TurboQuantMSE
                else:
                    q_cls = TurboQuantMSE
                q = q_cls(chunk_size, b, seed=chunk_seed)

                part = self._unpack_and_decode(chunk_data, 0, chunk_size, b, q)
                parts.append(part)

            return np.concatenate(parts)

    def _unpack_and_decode(self, data: bytes, pos: int, d: int, b: int, quantizer) -> np.ndarray:
        """Unpack bytes and decode to a vector."""
        idx_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        mse_idx = np.frombuffer(data[pos : pos + idx_len * 2], dtype=np.uint16)
        pos += idx_len * 2

        if b >= 2:
            signs_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            signs = np.frombuffer(data[pos : pos + signs_len], dtype=np.int8)
            pos += signs_len
            res_norm = struct.unpack_from("<f", data, pos)[0]
            pos += 4
            x_unit = quantizer.decode(mse_idx, signs, np.float32(res_norm))
        else:
            x_unit = quantizer.decode(mse_idx)

        norm = struct.unpack_from("<f", data, pos)[0]
        return x_unit * norm

    @property
    def name(self) -> str:
        return "gguf"


# ---------------------------------------------------------------------------
# vLLM-style KV-cache backend
# ---------------------------------------------------------------------------

class VLLMCacheBackend(LLMBackend):
    """
    vLLM-style KV-cache quantization backend.

    KV-cache vectors are typically short (sequence_length × head_dim) and
    benefit from per-token quantization.

    Format:
        [header: 12 bytes]
          - magic (4 bytes): b'TQKV'
          - version (4 bytes): uint32
          - n_tokens (4 bytes): uint32
        [per-token data]
          - token_size (4 bytes)
          - token_quant_data
    """

    MAGIC = b"TQKV"
    VERSION = 1

    def quantizer(self, d: int, b: int, seed: Optional[int] = None):
        if b >= 2:
            return TurboQuantProd(d, b, seed=seed)
        return TurboQuantMSE(d, b, seed=seed)

    def encode_weight(self, kv_cache: np.ndarray, quantizer) -> bytes:
        """
        Encode KV-cache tensor.

        Parameters
        ----------
        kv_cache : (n_tokens, d) array
            KV-cache for all tokens in a sequence.
        """
        if kv_cache.ndim == 1:
            kv_cache = kv_cache[None, :]
        n_tokens, d = kv_cache.shape

        header = struct.pack("<4sII", self.MAGIC, self.VERSION, n_tokens)

        tokens_data = b""
        for i in range(n_tokens):
            token = kv_cache[i].astype(np.float64)
            norm = float(np.linalg.norm(token))
            if norm > 1e-12:
                token_unit = token / norm
            else:
                token_unit = token

            encoded = quantizer.encode(token_unit)

            token_data = b""
            if isinstance(encoded, tuple):
                mse_idx, signs, res_norm = encoded
                token_data += struct.pack("<I", len(mse_idx))
                token_data += mse_idx.tobytes()
                token_data += struct.pack("<I", len(signs))
                token_data += signs.tobytes()
                token_data += struct.pack("<f", float(np.asarray(res_norm).item()))
            else:
                token_data += struct.pack("<I", len(encoded))
                token_data += encoded.tobytes()
            token_data += struct.pack("<f", float(norm))

            tokens_data += struct.pack("<I", len(token_data))
            tokens_data += token_data

        return header + tokens_data

    def decode_weight(self, data: bytes, d: int, b: int, quantizer) -> np.ndarray:
        header = struct.unpack("<4sII", data[:12])
        assert header[0] == self.MAGIC
        n_tokens = header[2]

        tokens = []
        pos = 12
        for _ in range(n_tokens):
            token_size = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            token_data = data[pos : pos + token_size]
            pos += token_size

            tpos = 0
            idx_len = struct.unpack_from("<I", token_data, tpos)[0]
            tpos += 4
            mse_idx = np.frombuffer(
                token_data[tpos : tpos + idx_len * 2], dtype=np.uint16
            )
            tpos += idx_len * 2

            if b >= 2:
                signs_len = struct.unpack_from("<I", token_data, tpos)[0]
                tpos += 4
                signs = np.frombuffer(
                    token_data[tpos : tpos + signs_len], dtype=np.int8
                )
                tpos += signs_len
                res_norm = struct.unpack_from("<f", token_data, tpos)[0]
                tpos += 4
                x_unit = quantizer.decode(mse_idx, signs, np.float32(res_norm))
            else:
                x_unit = quantizer.decode(mse_idx)

            norm = struct.unpack_from("<f", token_data, tpos)[0]
            tokens.append(x_unit * norm)

        result = np.array(tokens)
        return result if n_tokens > 1 else result[0]

    @property
    def name(self) -> str:
        return "vllm"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Registry of available backends
_BACKENDS: dict[str, type[LLMBackend]] = {
    "gguf": GGUFSimpleBackend,
    "vllm": VLLMCacheBackend,
}


class TurboQuantBackend:
    """
    Factory for creating LLM framework integration backends.

    Usage
    -----
        backend = TurboQuantBackend.create("gguf")
        q = backend.quantizer(d=4096, b=4)
        data = backend.encode_weight(weight, q)
        weight_rec = backend.decode_weight(data, 4096, 4, q)
    """

    @classmethod
    def create(cls, name: str) -> LLMBackend:
        """
        Create a backend by name.

        Parameters
        ----------
        name : str
            Backend name: "gguf" or "vllm".

        Returns
        -------
        backend : LLMBackend
        """
        if name not in _BACKENDS:
            raise ValueError(
                f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}"
            )
        return _BACKENDS[name]()

    @classmethod
    def list_backends(cls) -> list[str]:
        """List available backend names."""
        return list(_BACKENDS.keys())

    @classmethod
    def register(cls, name: str, backend_cls: type[LLMBackend]) -> None:
        """
        Register a custom backend.

        Parameters
        ----------
        name : str
            Backend name.
        backend_cls : type[LLMBackend]
            Backend class.
        """
        _BACKENDS[name] = backend_cls


# ---------------------------------------------------------------------------
# Convenience: HuggingFace-style layer quantization
# ---------------------------------------------------------------------------

@dataclass
class LayerInfo:
    """Metadata about a model layer."""
    name: str
    shape: tuple[int, ...]
    dtype: str
    param_count: int


@dataclass
class QuantizedModel:
    """A fully quantized model ready for inference."""
    layers: dict[str, bytes]       # layer_name -> binary blob
    config: dict[str, Any]         # quantization config
    layer_info: dict[str, LayerInfo]

    def total_bits(self) -> int:
        return sum(len(v) * 8 for v in self.layers.values())

    def summary(self) -> str:
        lines = ["Quantized Model Summary:"]
        total_params = 0
        total_bits = 0
        for name, info in self.layer_info.items():
            params = info.param_count
            bits = len(self.layers[name]) * 8
            total_params += params
            total_bits += bits
            bits_per_param = bits / params if params > 0 else 0
            lines.append(
                f"  {name:<40s} shape={str(info.shape):<20s} "
                f"{params:>8d} params  {bits_per_param:.1f} bits/param"
            )
        lines.append(
            f"  {'TOTAL':<40s} {'':>20s} {total_params:>8d} params  "
            f"{total_bits / total_params:.1f} bits/param avg"
        )
        return "\n".join(lines)


def quantize_model_layers(
    state_dict: dict[str, np.ndarray],
    backend_name: str = "gguf",
    bits_per_layer: Optional[dict[str, int]] = None,
    default_bits: int = 4,
    seed: int = 42,
    max_chunk_size: int = 4096,
) -> QuantizedModel:
    """
    Quantize all layers of a model state dict.

    For large weight tensors, automatically splits into chunks of
    `max_chunk_size` to avoid OOM when creating rotation matrices.

    Parameters
    ----------
    state_dict : dict
        {layer_name: weight_array} from a HuggingFace model.
    backend_name : str
        Backend to use ("gguf" or "vllm").
    bits_per_layer : dict, optional
        {layer_name: bits} for mixed-precision.
    default_bits : int
        Default bits per coordinate.
    seed : int
        Random seed.
    max_chunk_size : int
        Maximum chunk size for large weights (default 4096).

    Returns
    -------
    quantized : QuantizedModel
    """
    backend = TurboQuantBackend.create(backend_name)
    layers = {}
    layer_info = {}
    rng = np.random.default_rng(seed)

    for name, weight in state_dict.items():
        flat = weight.ravel().astype(np.float64)
        d = flat.shape[0]
        b = bits_per_layer.get(name, default_bits) if bits_per_layer else default_bits
        layer_seed = int(rng.integers(0, 2**31))

        n_chunks = (d + max_chunk_size - 1) // max_chunk_size

        if n_chunks == 1:
            # Small weight: direct quantization
            q = backend.quantizer(d, b, seed=layer_seed)
            encoded = backend.encode_weight(weight, q)
            layers[name] = encoded
        else:
            # Large weight: chunked quantization
            chunk_data = b""
            for i in range(n_chunks):
                offset = i * max_chunk_size
                chunk_size = min(max_chunk_size, d - offset)
                chunk = flat[offset : offset + chunk_size]
                chunk_seed = layer_seed + i

                q = backend.quantizer(chunk_size, b, seed=chunk_seed)
                chunk_encoded = backend.encode_weight(chunk, q)
                chunk_data += struct.pack("<I", len(chunk_encoded))
                chunk_data += chunk_encoded

            # Store with chunk header
            header = struct.pack("<II", n_chunks, d)
            layers[name] = header + chunk_data

        layer_info[name] = LayerInfo(
            name=name,
            shape=weight.shape,
            dtype=str(weight.dtype),
            param_count=weight.size,
        )

    return QuantizedModel(
        layers=layers,
        config={
            "backend": backend_name,
            "default_bits": default_bits,
            "seed": seed,
            "max_chunk_size": max_chunk_size,
        },
        layer_info=layer_info,
    )


def reconstruct_model_layers(
    quantized_model: QuantizedModel,
    state_dict_template: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Reconstruct weights from a quantized model.
    """
    backend = TurboQuantBackend.create(quantized_model.config["backend"])
    max_chunk_size = quantized_model.config.get("max_chunk_size", 4096)
    result = {}

    rng = np.random.default_rng(quantized_model.config.get("seed", 42))

    for name, orig_weight in state_dict_template.items():
        d = orig_weight.size
        b = quantized_model.config["default_bits"]
        layer_seed = int(rng.integers(0, 2**31))
        n_chunks = (d + max_chunk_size - 1) // max_chunk_size

        data = quantized_model.layers[name]

        if n_chunks == 1:
            q = backend.quantizer(d, b, seed=layer_seed)
            weight_flat = backend.decode_weight(data, d, b, q)
        else:
            # Chunked: skip header
            pos = 8  # n_chunks (4) + total_d (4)
            parts = []
            for i in range(n_chunks):
                chunk_size = min(max_chunk_size, d - sum(len(p) for p in parts))
                chunk_seed = layer_seed + i
                chunk_data_len = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                chunk_data = data[pos : pos + chunk_data_len]
                pos += chunk_data_len

                q = backend.quantizer(chunk_size, b, seed=chunk_seed)
                part = backend.decode_weight(chunk_data, chunk_size, b, q)
                parts.append(part)
            weight_flat = np.concatenate(parts)

        result[name] = weight_flat.reshape(orig_weight.shape).astype(orig_weight.dtype)

    return result

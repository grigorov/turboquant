#!/usr/bin/env python3
"""
TurboQuant benchmarks: MSE, throughput, memory, inner-product quality.

Usage:
    python benchmark.py [--d 256] [--n 10000] [--bits 1 2 4 8]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from turboquant import TurboQuantMSE, TurboQuantProd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_unit_batch(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@dataclass
class MseBenchmark:
    d: int
    b: int
    n: int
    mse: float
    setup_s: float
    encode_ms: float
    decode_ms: float
    bits_per_vec: int


@dataclass
class IpbBenchmark:
    d: int
    b: int
    n: int
    ip_bias: float
    ip_rmse: float
    bits_per_vec: int


# ---------------------------------------------------------------------------
# MSE benchmarks
# ---------------------------------------------------------------------------


def benchmark_mse(d: int, b: int, n: int, seed: int = 0) -> MseBenchmark:
    X = make_unit_batch(n, d, seed)

    t0 = time.perf_counter()
    q = TurboQuantMSE(d, b, seed=seed)
    setup_s = time.perf_counter() - t0

    # Encode (warmup + timed)
    _ = q.encode(X[:1])  # warmup
    t1 = time.perf_counter()
    idx = q.encode(X)
    encode_ms = (time.perf_counter() - t1) / n * 1000  # per vector

    # Decode
    t2 = time.perf_counter()
    _ = q.decode(idx)
    decode_ms = (time.perf_counter() - t2) / n * 1000

    mse = q.mse(X)

    return MseBenchmark(
        d=d, b=b, n=n, mse=mse, setup_s=setup_s,
        encode_ms=encode_ms, decode_ms=decode_ms,
        bits_per_vec=b * d,
    )


# ---------------------------------------------------------------------------
# Inner-product benchmarks
# ---------------------------------------------------------------------------


def benchmark_ip(d: int, b: int, n: int, seed: int = 0) -> IpbBenchmark:
    rng = np.random.default_rng(seed)
    X = make_unit_batch(n, d, seed)
    y = rng.standard_normal(d)
    y /= np.linalg.norm(y)

    q = TurboQuantProd(d, b, seed=seed)

    mse_idx, qjl_signs, res_norms = q.encode(X)

    true_ips = X @ y
    est_ips = np.array([
        q.inner_product_estimate(y, mse_idx[i], qjl_signs[i], res_norms[i])
        for i in range(n)
    ])

    bias = float(np.mean(est_ips - true_ips))
    rmse = float(np.sqrt(np.mean((est_ips - true_ips) ** 2)))

    return IpbBenchmark(
        d=d, b=b, n=n, ip_bias=bias, ip_rmse=rmse,
        bits_per_vec=q.bits_per_vector(),
    )


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def memory_summary(d: int, b_mse: int, b_prod: int) -> None:
    raw_bytes = d * 8  # float64
    mse_bytes = (b_mse * d) / 8
    prod_bytes = ((b_prod - 1) * d + d + 32) / 8

    print("\n--- Memory per vector ---")
    print(f"  Raw (float64)       : {raw_bytes:>8} bytes  ({raw_bytes/1024:.1f} KB)")
    print(f"  MSE b={b_mse}         : {mse_bytes:>8.1f} bytes  ({mse_bytes/1024:.1f} KB)  "
          f"({100 * mse_bytes / raw_bytes:.1f}%)")
    print(f"  Prod b={b_prod}      : {prod_bytes:>8.1f} bytes  ({prod_bytes/1024:.1f} KB)  "
          f"({100 * prod_bytes / raw_bytes:.1f}%)")


# ---------------------------------------------------------------------------
# Throughput analysis
# ---------------------------------------------------------------------------


def throughput_summary(results: list[MseBenchmark]) -> None:
    print("\n--- Throughput (vectors/sec) ---")
    for r in results:
        enc_vps = 1000 / r.encode_ms
        dec_vps = 1000 / r.decode_ms
        print(f"  MSE b={r.b:2d}  encode={enc_vps:>8.0f} v/s  decode={dec_vps:>8.0f} v/s")


# ---------------------------------------------------------------------------
# Quality vs compression
# ---------------------------------------------------------------------------


def quality_summary(mse_results: list[MseBenchmark], ip_results: list[IpbBenchmark]) -> None:
    print("\n--- Quality vs Compression ---")
    d = mse_results[0].d if mse_results else (ip_results[0].d if ip_results else 64)
    raw_bits = d * 64  # float64 bits equivalent
    print(f"  {'Method':<10} {'b':>3} {'MSE':>10} {'IP bias':>10} {'IP RMSE':>10} "
          f"{'Bits/vec':>10} {'Ratio':>8}")
    for r in mse_results:
        ratio = raw_bits / r.bits_per_vec
        print(f"  {'MSE':<10} {r.b:>3} {r.mse:>10.6f} {'—':>10} {'—':>10} "
              f"{r.bits_per_vec:>10} {ratio:>7.1f}x")
    for r in ip_results:
        ratio = raw_bits / r.bits_per_vec
        print(f"  {'Prod':<10} {r.b:>3} {'—':>10} {r.ip_bias:>+10.6f} {r.ip_rmse:>10.6f} "
              f"{r.bits_per_vec:>10} {ratio:>7.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant benchmarks")
    parser.add_argument("--d", type=int, default=256, help="Vector dimension")
    parser.add_argument("--n", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Bits per coordinate")
    args = parser.parse_args()

    d, n = args.d, args.n
    print("=" * 70)
    print(f"TurboQuant benchmarks  |  d={d}  n={n}")
    print("=" * 70)

    # MSE benchmarks
    print("\n--- TurboQuantMSE ---")
    mse_results: list[MseBenchmark] = []
    for b in args.bits:
        r = benchmark_mse(d, b, n)

        # MSE decay rate: should be ~4× per additional bit
        # Compute per-bit decay: (mse_prev / mse_curr) ^ (1 / delta_bits)
        if len(mse_results) > 0:
            prev = mse_results[-1]
            delta_b = b - prev.b
            if delta_b > 0 and r.mse > 0 and prev.mse > 0:
                per_bit_decay = (prev.mse / r.mse) ** (1.0 / delta_b)
                decay_str = f"{per_bit_decay:.1f}×/bit"
            else:
                decay_str = "—"
        else:
            decay_str = "—"

        mse_results.append(r)
        print(f"  b={b:2d}  MSE={r.mse:.6f}  "
              f"setup={r.setup_s:.2f}s  "
              f"enc={r.encode_ms:.2f}ms  dec={r.decode_ms:.2f}ms  "
              f"bits/vec={r.bits_per_vec}  decay={decay_str}")

    # IP benchmarks
    print("\n--- TurboQuantProd ---")
    ip_results: list[IpbBenchmark] = []
    for b in [bt for bt in args.bits if bt >= 2]:
        r = benchmark_ip(d, b, n)
        ip_results.append(r)
        bias_ok = abs(r.ip_bias) < 0.01
        bias_str = "✓" if bias_ok else "✗"
        print(f"  b={b:2d}  IP bias={r.ip_bias:+.6f}  RMSE={r.ip_rmse:.6f}  "
              f"bits/vec={r.bits_per_vec}  unbiased={bias_str}")

    # Summaries
    memory_summary(d, min(args.bits), max(bt for bt in args.bits if bt >= 2))
    throughput_summary(mse_results)
    quality_summary(mse_results, ip_results)

    print("\nDone.")


if __name__ == "__main__":
    main()

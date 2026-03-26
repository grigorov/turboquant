#!/usr/bin/env python3
"""
TurboQuant fuzzer: correctness invariant checker.

Invariants:
  1. encode→decode roundtrip (shape, no crash)
  2. MSE monotonicity: MSE(b+1) ≤ MSE(b) × 1.1
  3. IP unbiasedness: |mean(est_ip - true_ip)| < 0.05
  4. IP variance: var(est_ip - true_ip) ≤ (π/2d)·‖y‖² × 2.0  [skipped for d<16]
  5. Edge cases: b=1/16, d=2/4, zero/near-zero vectors

Usage:
    python fuzzer.py [--iters N] [--seed S]
"""

import argparse
import math
import sys
import numpy as np

from turboquant import TurboQuantMSE, TurboQuantProd

D_CHOICES = [4, 8, 16, 32, 64, 128, 256]
N_VECTORS_RANGE = (500, 2000)


def make_unit_batch(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n random unit vectors of dimension d."""
    X = rng.standard_normal((n, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def check_roundtrip(q: TurboQuantMSE, X: np.ndarray) -> tuple[bool, str]:
    """Invariant 1: encode→decode produces correct shape, no crash."""
    try:
        idx = q.encode(X)
        x_hat = q.decode(idx)
    except Exception as e:
        return False, f"roundtrip crashed: {e}"
    if x_hat.shape != X.shape:
        return False, f"shape mismatch: {x_hat.shape} != {X.shape}"
    return True, "OK"


def check_mse_monotone(d: int, b: int, X: np.ndarray, seed: int) -> tuple[bool, str]:
    """Invariant 2: MSE(b+1) ≤ MSE(b) × 1.1."""
    if b >= 8:
        return True, "skipped(b>=8)"
    q_b  = TurboQuantMSE(d, b,     seed=seed)
    q_b1 = TurboQuantMSE(d, b + 1, seed=seed)
    mse_b  = q_b.mse(X)
    mse_b1 = q_b1.mse(X)
    if mse_b1 > mse_b * 1.1:
        return False, f"MSE({b+1})={mse_b1:.5f} > MSE({b})={mse_b:.5f} × 1.1"
    return True, f"MSE({b})={mse_b:.5f} MSE({b+1})={mse_b1:.5f}"


def check_ip_bias(q: TurboQuantProd, X: np.ndarray, y: np.ndarray) -> tuple[bool, str]:
    """Invariant 3: |mean(est_ip - true_ip)| < 0.05."""
    true_ips = X @ y
    est_ips = np.array([
        q.inner_product_estimate(y, *q.encode(X[i]))
        for i in range(len(X))
    ])
    bias = float(np.mean(est_ips - true_ips))
    if abs(bias) >= 0.05:
        return False, f"ip_bias={bias:+.5f}  (threshold=0.05)"
    return True, f"ip_bias={bias:+.5f}"


def check_ip_variance(q: TurboQuantProd, X: np.ndarray, y: np.ndarray) -> tuple[bool, str]:
    """Invariant 4: var(est_ip - true_ip) ≤ (π/2d)·‖y‖² × 2.0. Skipped for d < 16."""
    d = q.d
    if d < 16:
        return True, "skipped(d<16)"
    true_ips = X @ y
    est_ips = np.array([
        q.inner_product_estimate(y, *q.encode(X[i]))
        for i in range(len(X))
    ])
    var = float(np.var(est_ips - true_ips))
    threshold = (math.pi / (2 * d)) * float(np.dot(y, y)) * 2.0
    if var > threshold:
        return False, f"ip_var={var:.5f} > threshold={threshold:.5f}"
    return True, f"ip_var={var:.5f} threshold={threshold:.5f}"


def check_edge_cases(d: int, seed: int) -> tuple[bool, str]:
    """Invariant 5: edge cases don't crash, output shapes are correct."""
    failures = []

    # b=1 и b=8 для MSE (b=16 требует 65536 вызовов scipy.quad — слишком медленно)
    for b in [1, 8]:
        try:
            q = TurboQuantMSE(d, b, seed=seed)
            x = np.zeros(d); x[0] = 1.0
            x_hat = q.decode(q.encode(x))
            if x_hat.shape != (d,):
                failures.append(f"MSE b={b}: wrong shape {x_hat.shape}")
        except Exception as e:
            failures.append(f"MSE b={b}: {e}")

    # b=2 минимум для Prod
    try:
        q = TurboQuantProd(d, 2, seed=seed)
        x = np.zeros(d); x[0] = 1.0
        idx, signs, gamma = q.encode(x)
        if idx.shape != (d,):
            failures.append(f"Prod b=2: wrong idx shape {idx.shape}")
    except Exception as e:
        failures.append(f"Prod b=2: {e}")

    # Нулевой вектор
    try:
        q = TurboQuantMSE(d, 4, seed=seed)
        x_hat = q.decode(q.encode(np.zeros(d)))
        if x_hat.shape != (d,):
            failures.append(f"zero vector: wrong shape {x_hat.shape}")
    except Exception as e:
        failures.append(f"zero vector: {e}")

    # Почти нулевой вектор (норма ~ 1e-13)
    try:
        q = TurboQuantMSE(d, 4, seed=seed)
        x_hat = q.decode(q.encode(np.full(d, 1e-14)))
        if x_hat.shape != (d,):
            failures.append(f"near-zero: wrong shape {x_hat.shape}")
    except Exception as e:
        failures.append(f"near-zero: {e}")

    if failures:
        return False, "failures: " + "; ".join(failures)
    return True, "OK"


def run_iteration(iter_idx: int, master_rng: np.random.Generator) -> bool:
    d   = int(master_rng.choice(D_CHOICES))
    b   = int(master_rng.integers(1, 9))   # 1..8
    b_p = int(master_rng.integers(2, 9))   # 2..8
    n   = int(master_rng.integers(N_VECTORS_RANGE[0], N_VECTORS_RANGE[1] + 1))
    seed      = int(master_rng.integers(0, 2**31))
    iter_seed = int(master_rng.integers(0, 2**31))

    X = make_unit_batch(n, d, np.random.default_rng(iter_seed))
    y = X[0].copy()

    q_mse  = TurboQuantMSE(d, b,   seed=seed)
    q_prod = TurboQuantProd(d, b_p, seed=seed)

    checks = [
        ("roundtrip", check_roundtrip(q_mse, X)),
        ("mse_mono",  check_mse_monotone(d, b, X, seed)),
        ("ip_bias",   check_ip_bias(q_prod, X, y)),
        ("ip_var",    check_ip_variance(q_prod, X, y)),
        ("edge",      check_edge_cases(d, seed)),
    ]

    passed = all(ok for _, (ok, _) in checks)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] iter={iter_idx:02d}  d={d}  b={b}/{b_p}  n={n}")
    for name, (ok, msg) in checks:
        marker = "OK" if ok else f"FAIL: {msg}"
        print(f"       {name}={marker}")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant correctness fuzzer")
    parser.add_argument("--iters", type=int, default=50,   help="iterations (default: 50)")
    parser.add_argument("--seed",  type=int, default=None, help="master seed (default: random)")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(0, 2**31))
    print(f"Master seed: {seed}  iters: {args.iters}\n")
    master_rng = np.random.default_rng(seed)

    failures = 0
    for i in range(1, args.iters + 1):
        if not run_iteration(i, master_rng):
            failures += 1

    # Отдельные прогоны для малых d
    print("\n--- Small-d edge cases ---")
    for d_small in [2, 4]:
        ok, msg = check_edge_cases(d_small, seed=0)
        print(f"[{'PASS' if ok else 'FAIL'}] d={d_small}  {msg}")
        if not ok:
            failures += 1

    total = args.iters + 2
    print(f"\nResults: {total - failures}/{total} passed")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

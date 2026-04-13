# TurboQuant — Near-Optimal Vector Quantization

Python and Rust implementations of the **TurboQuant** algorithm.
Source: [arxiv.org/html/2504.19874v1](https://arxiv.org/html/2504.19874v1)

[![CI](https://github.com/grigorov/turboquant/actions/workflows/ci.yml/badge.svg)](https://github.com/grigorov/turboquant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Table of Contents

- [What is TurboQuant](#what-is-turboquant)
- [Mathematical Background](#mathematical-background)
- [Algorithms](#algorithms)
- [Theoretical Guarantees](#theoretical-guarantees)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Demo Results](#demo-results)
- [Advanced Modules](#advanced-modules)
  - [Mixed-Precision](#mixed-precision-quantization)
  - [Sparse Vectors](#sparse-vector-quantization)
  - [Numba JIT](#numba-jit-acceleration)
  - [LLM Integration](#llm-integration)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Code Structure](#code-structure)
- [Rust Implementation](#rust-implementation)
  - [Python Bindings](#python-bindings-via-maturin)
- [Fuzzer](#fuzzer)

---

## What is TurboQuant

TurboQuant is a vector quantization scheme designed for efficient large language model (LLM) inference. It compresses real-valued vectors to a few bits per coordinate while:

- minimizing mean squared error of reconstruction (MSE mode);
- providing **unbiased** inner product estimates (IP mode).

Practical applications include quantization of transformer weights and activations, vector databases, and approximate nearest neighbor (ANN) search.

---

## Mathematical Background

### Marginal Distribution of Sphere Coordinates

Let vector **x** lie on the unit sphere S^{d-1}. After applying a random orthogonal rotation matrix Π, the coordinates of y = Π·x become nearly independent, each following the marginal distribution:

```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 − x²)^((d−3)/2),   x ∈ (−1, 1)
```

This is a beta-like distribution. As d → ∞ it converges to a normal:

```
f_X(x) → N(0, 1/d)
```

This coordinate independence after rotation allows applying a **scalar** quantizer to each coordinate independently — without the quality loss that would occur with naive per-coordinate quantization without rotation.

### Lloyd-Max Quantizer

For each coordinate, a one-dimensional optimization problem is solved:

```
min_{c₁ ≤ ... ≤ c_{2^b}}  Σᵢ ∫ |x − cᵢ|² · f_X(x) dx
```

Lloyd's iteration algorithm:

1. **Initialization**: centroids are quantiles of distribution f_X.
2. **Cell boundaries**: midpoints between adjacent centroids.
3. **Update**: each centroid = conditional expectation of f_X over its cell.
4. Repeat until convergence (||Δc|| < 10⁻¹⁰).

For d > 50, the Gaussian approximation N(0, 1/d) is used — it is more accurate at large dimensions and significantly faster (analytical formulas instead of numerical integration).

### Inner Product Problem and QJL

The MSE-optimal quantizer introduces a **multiplicative bias** in the inner product estimate: E[⟨y, x̃_mse⟩] ≈ (2/π)·⟨y, x⟩. This is unacceptable for nearest neighbor search and attention mechanisms.

The solution is the **Quantized Johnson-Lindenstrauss (QJL)** transform:

```
Encode:  z = sign(S · x)              ∈ {−1, +1}^d
Decode:  x̃ = (√(π/2) / d) · ‖x‖₂ · Sᵀ · z
```

where S ∈ ℝ^{d×d} is a matrix with N(0, 1) entries.

**Unbiasedness property:**

```
E[⟨y, x̃⟩] = ⟨y, x⟩   for any fixed y
```

**Variance:** Var(⟨y, x̃⟩) ≤ (π/2d) · ‖y‖₂²

---

## Algorithms

### TurboQuantMSE — MSE Minimization

**Initialization:**
1. Random rotation matrix Π ∈ ℝ^{d×d} via QR decomposition of a random N(0,1) matrix.
2. Compute optimal codebook {c₁, ..., c_{2^b}} via Lloyd-Max algorithm.

**Encoding Quant_MSE(x):**
```
y    ← Π · x                          # rotate
idxⱼ ← argmin_k |yⱼ − cₖ|,  j=1..d  # nearest centroid
output: idx  (b bits per coordinate)
```

**Decoding DeQuant_MSE(idx):**
```
ỹⱼ ← c_{idxⱼ},  j=1..d     # look up centroids
x̃  ← Πᵀ · ỹ                # inverse rotation
output: x̃
```

**Memory cost:** b·d bits per vector.

---

### TurboQuantProd — Unbiased Inner Product

Combines MSE quantization with a QJL residual.

**Initialization:**
1. Create TurboQuantMSE with bit width (b−1).
2. Random projection matrix S ∈ ℝ^{d×d} with N(0,1) entries.

**Encoding Quant_Prod(x):**
```
idx  ← Quant_MSE(x)                  # step 1: MSE quantization (b-1 bits)
r    ← x − DeQuant_MSE(idx)          # step 2: residual
z    ← sign(S · r)                   # step 3: QJL on residual
γ    ← ‖r‖₂                          # store residual norm
output: (idx, z, γ)
```

**Decoding DeQuant_Prod(idx, z, γ):**
```
x̃_mse ← DeQuant_MSE(idx)
x̃_qjl ← (√(π/2) / d) · γ · Sᵀ · z
output: x̃_mse + x̃_qjl
```

**Memory cost:** b·d bits + 32 bits (float for γ) per vector.

---

## Theoretical Guarantees

| Method | Distortion | Lower Bound |
|---|---|---|
| TurboQuantMSE | D_mse ≤ (√3π/2) / 4^b | D_mse ≥ 1 / 4^b |
| TurboQuantProd | D_prod ≤ (√3π²·‖y‖²/d) / 4^b | D_prod ≥ (1/d) / 4^b |

TurboQuantMSE achieves within ~2.7× of the theoretical optimum.
TurboQuantProd is near-optimal for inner products.

**Unbiasedness of TurboQuantProd:**
```
E[⟨y, x̃⟩] = ⟨y, x⟩
```

---

## Installation

### Pure Python (no compilation)

Requirements: Python 3.9+, NumPy, SciPy.

```bash
pip install numpy scipy
```

The file `turboquant.py` requires no installation — just copy it into your project.

### With Rust bindings (recommended, maximum performance)

Requirements: Python 3.9+, Rust 1.75+ (edition 2021), `maturin`.

```bash
# Install maturin
pip install maturin

# Build and install from source
maturin develop --release

# Or build a wheel package
maturin build --release
pip install target/wheels/*.whl
```

After installation, Rust bindings are available via the `turboquant_rs` module:

```python
from turboquant_rs import TurboQuantMse, TurboQuantProd, QuantizedProd
```

> **Note:** The pure Python version (`turboquant.py`) continues to work unchanged. Rust bindings are an optional acceleration layer.

---

## Usage

### Rust bindings (recommended)

```python
from turboquant_rs import TurboQuantMse, TurboQuantProd

d, b = 256, 4

# MSE quantization
q = TurboQuantMse(d=d, b=b, seed=42)

x = [float(v) for v in np.random.randn(d)]
x_norm = x / np.linalg.norm(x)

idx   = q.encode(x_norm.tolist())   # List[u16] — centroid indices
x_hat = q.decode(idx)               # List[f64] — reconstruction

# TurboQuantProd for unbiased inner product estimation
q_prod = TurboQuantProd(d=d, b=b, seed=42)
qv = q_prod.encode(x_norm.tolist())
ip_est = q_prod.inner_product_estimate(x_norm.tolist(), qv)
```

### Pure Python (without Rust)

```python
import numpy as np
from turboquant import TurboQuantMSE

d, b = 256, 4   # dimension and bits per coordinate

# Create quantizer (computes rotation matrix and codebook)
q = TurboQuantMSE(d=d, b=b, seed=42)

# Single vector (unit norm)
x = np.random.randn(d)
x /= np.linalg.norm(x)

idx   = q.encode(x)      # (d,) uint16 — centroid indices
x_hat = q.decode(idx)    # (d,) float64 — reconstructed vector

mse = np.mean((x - x_hat) ** 2)
print(f"MSE = {mse:.6f}")

# Batch of vectors
X = np.random.randn(1000, d)
X /= np.linalg.norm(X, axis=1, keepdims=True)

IDX   = q.encode(X)      # (1000, d) uint16
X_hat = q.decode(IDX)    # (1000, d) float64
```

### Quantization of Arbitrary (Non-Unit) Vectors

```python
x_raw = np.random.randn(d) * 5.0    # arbitrary norm

idx, norm = q.encode_with_norm(x_raw)
x_rec     = q.decode_with_norm(idx, norm)
```

### Unbiased Inner Product Estimation

```python
from turboquant import TurboQuantProd

q = TurboQuantProd(d=256, b=4, seed=42)

# Encode database (stored vectors)
X = np.random.randn(1000, 256)
X /= np.linalg.norm(X, axis=1, keepdims=True)

mse_idx, qjl_signs, res_norms = q.encode(X)

# Estimate inner product with query y
y = np.random.randn(256)
y /= np.linalg.norm(y)

ip_true = X @ y   # true inner products

for i in range(len(X)):
    ip_est = q.inner_product_estimate(y, mse_idx[i], qjl_signs[i], res_norms[i])
    # E[ip_est] == ip_true[i]
```

### Direct Decoding (Vector Reconstruction)

```python
X_tilde = q.decode(mse_idx, qjl_signs, res_norms)  # (1000, 256)
```

---

## API

### `TurboQuantMSE(d, b, seed=None)`

| Parameter | Type | Description |
|---|---|---|
| `d` | int | Vector dimension |
| `b` | int | Bits per coordinate (1–16) |
| `seed` | int or None | Random seed |

| Method | Description |
|---|---|
| `encode(x)` | Vector(s) → uint16 indices |
| `decode(idx)` | Indices → reconstructed float64 vectors |
| `encode_with_norm(x)` | For non-unit vectors: returns (indices, norms) |
| `decode_with_norm(idx, norms)` | Inverse of `encode_with_norm` |
| `mse(x)` | Compute mean MSE on a batch |

---

### `TurboQuantProd(d, b, seed=None)`

| Parameter | Type | Description |
|---|---|---|
| `d` | int | Vector dimension |
| `b` | int | Bits per coordinate (≥ 2) |
| `seed` | int or None | Random seed |

| Method | Description |
|---|---|
| `encode(x)` | Vector(s) → (mse_idx, qjl_signs, res_norms) |
| `decode(mse_idx, qjl_signs, res_norms)` | Reconstruct vector |
| `inner_product_estimate(y, mse_idx, qjl_signs, res_norms)` | Unbiased estimate of ⟨y, x⟩ |
| `bits_per_vector()` | Number of bits per compressed vector |

---

### `QJL(d, seed=None)`

Helper class used internally by `TurboQuantProd`.

| Method | Description |
|---|---|
| `encode(x)` | x → (±1 signs, norms) |
| `decode(signs, norms)` | Unbiased reconstruction |

---

## Demo Results

Running `python3 turboquant.py` on d=256, n=1000 random unit vectors:

```
============================================================
TurboQuant demo  |  d=256  n=1000
============================================================

--- TurboQuantMSE ---
  b= 1  MSE=0.00141  setup=0.01s  enc+dec=5.9ms
  b= 2  MSE=0.00046  setup=0.01s  enc+dec=6.8ms
  b= 4  MSE=0.00004  setup=0.21s  enc+dec=8.1ms
  b= 8  MSE=0.00000  setup=3.29s  enc+dec=133.5ms

--- TurboQuantProd (inner product) ---
  b= 2  IP bias=+0.00122  RMSE=0.04685  bits/vec=544
  b= 4  IP bias=-0.00007  RMSE=0.01389  bits/vec=1056
  b= 8  IP bias=+0.00002  RMSE=0.00135  bits/vec=2080

--- Unbiasedness verification (b=4, n=5000) ---
  Mean bias over 5000 vectors: -0.000153  (should be ≈ 0)
```

MSE decreases ~4× with each additional bit (matches theoretical 1/4^b).
Inner product estimation bias is practically zero.

---

## Code Structure

```
turboquant.py                    # quantization library
├── _lloyd_max_gaussian()        # Lloyd-Max for N(0, σ²)
├── _lloyd_max_beta_sphere()     # Lloyd-Max for sphere marginal
├── TurboQuantMSE                # Algorithm 1: MSE quantization
│   ├── __init__                 #   rotation + codebook
│   ├── encode / decode          #   batch encode/decode
│   └── encode_with_norm /       #   support for non-unit vectors
│       decode_with_norm
├── QJL                          # Quantized Johnson-Lindenstrauss
│   ├── encode                   #   x → sign(S·x)
│   └── decode                   #   z → (√(π/2)/d)·γ·Sᵀ·z
├── TurboQuantProd               # Algorithm 2: IP quantization
│   ├── __init__                 #   TurboQuantMSE(b-1) + QJL
│   ├── encode / decode          #   (idx, signs, residual norm)
│   ├── inner_product_estimate   #   unbiased estimate of ⟨y, x⟩
│   └── bits_per_vector          #   compressed vector size in bits
└── _demo()                      # demo and benchmarks

fuzzer.py                        # correctness fuzzer (5 invariants)
├── make_unit_batch()            # generate random unit vectors
├── check_roundtrip()            # invariant 1: encode→decode
├── check_mse_monotone()         # invariant 2: MSE decreases with b
├── check_ip_bias()              # invariant 3: IP unbiasedness
├── check_ip_variance()          # invariant 4: IP variance within theory
├── check_edge_cases()           # invariant 5: edge cases
├── run_iteration()              # one random iteration
└── main()                       # CLI: --iters, --seed
```

---

## Rust Implementation

The Rust implementation lives in the `rust/` directory as a `turboquant_rs` library crate with Python bindings via PyO3/maturin.

Dependencies: `rand 0.8`, `rand_distr 0.4`, `rayon 1.10` (parallelism), `pyo3 0.25` (Python bindings).

### Installation (Rust)

Requirements: Rust 1.75+ (edition 2021), Cargo.

```bash
cd rust
cargo build --release
```

Run the demo:

```bash
cargo run --release
```

#### Python bindings (via maturin)

```bash
# In project root
pip install maturin
maturin develop --release
```

After installation:

```python
from turboquant_rs import TurboQuantMse, TurboQuantProd, QuantizedProd

# Check Rust bindings availability
import turboquant_rs
print(turboquant_rs.__has_rust__)  # True
```

### Usage (Rust)

#### MSE Quantization

```rust
use turboquant::TurboQuantMse;

let d = 256;
let b = 4;
let q = TurboQuantMse::new(d, b, Some(42));

// Single vector (flat slice of length d)
let x: Vec<f64> = /* unit vector of length d */;
let idx: Vec<u16> = q.encode(&x);      // centroid indices
let x_hat: Vec<f64> = q.decode(&idx);  // reconstructed vector

// Batch: flat buffer of n*d elements
let x_batch: Vec<f64> = /* n*d elements */;
let idx_batch = q.encode(&x_batch);    // Vec<u16> of length n*d
let x_hat_batch = q.decode(&idx_batch);
```

#### Unbiased Inner Product Estimation

```rust
use turboquant::TurboQuantProd;

let q = TurboQuantProd::new(256, 4, Some(42));

// Encode a database vector
let qv = q.encode(&x);  // QuantizedVec { mse_idx, qjl_signs, res_norm }

// Estimate inner product with query y
let ip_est: f64 = q.inner_product_estimate(&y, &qv);
// E[ip_est] == ⟨y, x⟩

// Decode (vector reconstruction)
let x_tilde: Vec<f64> = q.decode(&qv);

println!("bits per vector: {}", q.bits_per_vector());
```

### API (Rust)

#### `TurboQuantMse`

| Method | Description |
|---|---|
| `TurboQuantMse::new(d, b, seed)` | Create quantizer: rotation matrix + Lloyd-Max codebook |
| `encode(x: &[f64]) -> Vec<u16>` | Flat vector buffer → centroid indices |
| `decode(idx: &[u16]) -> Vec<f64>` | Indices → reconstructed vectors |
| `encode_with_norm(x) -> (Vec<u16>, f32)` | For non-unit vectors |
| `decode_with_norm(idx, norm) -> Vec<f64>` | Inverse of `encode_with_norm` |
| `mse(x) -> f64` | Compute MSE on a batch |

Constructor parameters:

| Parameter | Type | Description |
|---|---|---|
| `d` | `usize` | Vector dimension |
| `b` | `usize` | Bits per coordinate (1–16) |
| `seed` | `Option<u64>` | Random seed (`None` — non-deterministic) |

#### `TurboQuantProd`

| Method | Description |
|---|---|
| `TurboQuantProd::new(d, b, seed)` | Create quantizer: `TurboQuantMse(b-1)` + QJL matrix |
| `encode(x: &[f64]) -> QuantizedVec` | Vector → `{ mse_idx, qjl_signs, res_norm }` |
| `decode(qv: &QuantizedVec) -> Vec<f64>` | Reconstruct vector |
| `inner_product_estimate(y, qv) -> f64` | Unbiased estimate of ⟨y, x⟩ |
| `bits_per_vector() -> usize` | Compressed vector size in bits |

#### `Qjl`

Helper type used internally by `TurboQuantProd`.

| Method | Description |
|---|---|
| `Qjl::new(d, seed)` | Initialize projection matrix S ∈ ℝ^{d×d} |
| `encode(x) -> (Vec<i8>, f64)` | x → (±1 signs, norm ‖x‖₂) |
| `decode(signs, norm) -> Vec<f64>` | Unbiased reconstruction |

### Python Bindings API

All Rust classes are available from Python via the `turboquant_rs` module:

#### `TurboQuantMse` (Python)

```python
from turboquant_rs import TurboQuantMse

q = TurboQuantMse(d=256, b=4, seed=42)

# Properties
q.d           # dimension
q.b           # bits per coordinate
q.n_centroids # number of centroids (2^b)

# Methods
q.encode(x: List[float]) -> List[int]
q.decode(indices: List[int]) -> List[float]
q.encode_with_norm(x: List[float]) -> Tuple[List[int], float]
q.decode_with_norm(indices: List[int], norm: float) -> List[float]
q.mse(x: List[float]) -> float
```

#### `TurboQuantProd` (Python)

```python
from turboquant_rs import TurboQuantProd

q = TurboQuantProd(d=256, b=4, seed=42)

# Properties
q.d  # dimension
q.b  # bits per coordinate

# Methods
q.encode(x: List[float]) -> QuantizedProd
q.decode(qv: QuantizedProd) -> List[float]
q.inner_product_estimate(y: List[float], qv: QuantizedProd) -> float
```

#### `QuantizedProd` (Python)

```python
# Compressed vector representation
qv.mse_indices    # List[int] — MSE centroid indices
qv.qjl_signs      # List[int]  — QJL signs (±1)
qv.residual_norm  # float      — residual norm

repr(qv)  # human-readable string
```

### Code Structure (Rust)

```
rust/
├── Cargo.toml
└── src/
    ├── lib.rs          # public API + Python bindings (PyO3)
    ├── main.rs         # demo and performance benchmarks
    ├── bin/
    │   └── fuzzer.rs   # correctness fuzzer (5 invariants)
    ├── lloyd.rs        # Lloyd-Max quantizer (Gaussian and sphere variants)
    ├── mse.rs          # TurboQuantMse: rotation + codebook, encode/decode
    ├── qjl.rs          # Qjl: matrix S, sign projection, unbiased decoding
    └── prod.rs         # TurboQuantProd: MSE + QJL residual, IP estimation
```

Python bindings are built via `maturin` and available as `turboquant_rs`.

---

## Fuzzer

Standalone correctness fuzzer for Python and Rust. Generates random vectors and parameters, runs 5 invariants on each iteration.

### Checked Invariants

| # | Invariant | Criterion |
|---|-----------|-----------|
| 1 | **Roundtrip** `decode(encode(x))` does not crash, correct shape | shape matches input |
| 2 | **MSE monotonicity** more bits → lower error | `MSE(b+1) ≤ MSE(b) × 1.1` |
| 3 | **IP unbiasedness** inner product estimate has no systematic bias | `\|mean(est − true)\| < 0.05` |
| 4 | **IP variance** within QJL theoretical bound (for d ≥ 16) | `var ≤ (π/2d)·‖y‖² × 2.0` |
| 5 | **Edge cases** b=1/8, zero and near-zero vectors, small d | no panics, correct shape |

### Fuzzer Python

```bash
# Basic run (50 iterations, random seed)
python fuzzer.py

# With fixed seed and iteration count
python fuzzer.py --iters 100 --seed 42
```

Sample output:

```
Master seed: 42  iters: 3

[PASS] iter=01  d=4  b=7/6  n=1158
       roundtrip=OK
       mse_mono=OK
       ip_bias=ip_bias=-0.00312
       ip_var=skipped(d<16)
       edge=OK
...
--- Small-d edge cases ---
[PASS] d=2  OK
[PASS] d=4  OK

Results: 5/5 passed
```

Exit code: `0` — all passed, `1` — failures present.

### Fuzzer Rust

```bash
# Basic run
cd rust
cargo run --bin fuzzer

# With parameters
cargo run --bin fuzzer -- --iters 100 --seed 42

# Release build (faster)
cargo run --release --bin fuzzer -- --seed 0
```

Sample output:

```
Master seed: 42  iters: 3

[PASS] iter=01  d=64  b=3/4  n=872
       roundtrip=OK
       mse_mono=MSE(3)=0.00234 MSE(4)=0.00089
       ip_bias=ip_bias=+0.00178
       ip_var=ip_var=0.00412 threshold=0.04909
       edge=OK
...
--- Small-d edge cases ---
[PASS] d=2  OK
[PASS] d=4  OK

Results: 5/5 passed
```

---

## Advanced Modules

### Mixed-Precision Quantization

The `turboquant_mixed.py` module allows assigning different bit-widths to different model layers or coordinate groups within a single vector.

```python
from turboquant_mixed import MixedPrecisionQuantizer

# Mode 1: per-layer — each model layer gets its own bit-width
q = MixedPrecisionQuantizer.from_layers(
    layer_dims=[256, 512, 128],
    layer_bits=[4, 2, 8],
    seed=42,
)

# Mode 2: per-group — a vector is split into parts with different bit-widths
q = MixedPrecisionQuantizer.from_groups(
    d=4096,
    group_sizes=[2048, 1024, 1024],
    group_bits=[4, 2, 8],
    seed=42,
)
print(q.bit_allocation_summary())
# Avg bits/coord: 4.50  Compression: 14.2x

# Mode 3: automatic importance-based allocation
q = MixedPrecisionQuantizer.from_importance(
    d=4096, n_groups=4, bits_range=(2, 8), seed=42,
)
# Groups: 2 → 4 → 6 → 8 bits
```

| Mode | Description | Compression |
|---|---|---|
| Layer | Each layer gets its own bit-width | Depends on model |
| Group | Vector split into groups | 12–14× |
| Importance | Automatic 2→8 bit ramp | 12.8× |

### Sparse Vector Quantization

The `turboquant_sparse.py` module is optimized for vectors with many zeros (MoE gating, sparse attention, pruning).

```python
from turboquant_sparse import SparseQuantizer
import numpy as np

# Vector with 90% zeros
d = 4096
x = np.zeros(d)
x[np.random.choice(d, size=d//10)] = np.random.randn(d//10)

q = SparseQuantizer(d, b=4, seed=42)
enc = q.encode(x)
x_hat = q.decode(enc)

print(f"NNZ: {len(enc.indices)}/{d}")
print(f"Compression: {q.compression_vs_dense(enc):.1f}×")
```

| Sparsity | Bits/vector | Compression vs Dense |
|---|---|---|
| 50% | 73 760 | 0.2× |
| 90% | 14 756 | **1.1×** |
| 99% | 1 472 | **11.1×** |

Supports scipy sparse matrices (`encode_sparse_matrix`) and TurboQuantProd mode.

### Numba JIT Acceleration

The `turboquant_numba.py` module provides 2× speedup for encode/decode via JIT compilation.

```python
from turboquant_numba import TurboQuantMSEJIT

q = TurboQuantMSEJIT(d=256, b=4, seed=42)
idx = q.encode(X)   # JIT-compiled, parallel
x_hat = q.decode(idx)
```

| Operation | NumPy | Numba | Speedup |
|---|---|---|---|
| Encode (d=256, n=5000) | 56 ms | 28 ms | **2.0×** |
| Decode | 5 ms | 47 ms | — (overhead) |

Install: `pip install numba` (optional).

### LLM Integration

The `turboquant_llm.py` module provides backend adapters for llama.cpp (GGUF) and vLLM (KV-cache).

```python
from turboquant_llm import (
    TurboQuantBackend,
    quantize_model_layers,
    reconstruct_model_layers,
)

# GGUF-style: model weight quantization
backend = TurboQuantBackend.create("gguf")
q = backend.quantizer(d=4096, b=4, seed=42)
data = backend.encode_weight(weight_matrix, q)
W_rec = backend.decode_weight(data, 4096, 4, q)

# Full model quantization
state_dict = {"layer1.q.weight": W_q, "layer1.k.weight": W_k, ...}
qmodel = quantize_model_layers(
    state_dict,
    backend_name="gguf",
    bits_per_layer={"layer1.q.weight": 4, "layer1.ffn.weight": 8},
    default_bits=2,
    seed=42,
)
print(qmodel.summary())

# VLLM KV-cache: per-token activation quantization
backend_kv = TurboQuantBackend.create("vllm")
q = backend_kv.quantizer(head_dim=128, b=4, seed=42)
data = backend_kv.encode_weight(kv_cache, q)
kv_rec = backend_kv.decode_weight(data, 128, 4, q)
```

| Backend | Format | Purpose |
|---|---|---|
| GGUF (`TQGG`) | Binary blob | llama.cpp-style weights |
| VLLM (`TQKV`) | Per-token binary | KV-cache activations |

---

## Benchmarks

### Python

```bash
python benchmark.py --d 256 --n 5000 --bits 1 2 4 8
```

Results (d=256, n=5000):

| Method | b | MSE | Encode (ms/vec) | Decode (ms/vec) | Bits/vec | Compression |
|---|---|---|---|---|---|---|
| MSE | 1 | 0.00147 | 0.006 | 0.001 | 256 | 64.0× |
| MSE | 2 | 0.00051 | 0.008 | 0.001 | 512 | 32.0× |
| MSE | 4 | 0.00009 | 0.012 | 0.001 | 1024 | 16.0× |
| MSE | 8 | 0.00004 | 0.183 | 0.001 | 2048 | 8.0× |
| Prod | 2 | — | — | — | 544 | 30.1× |
| Prod | 4 | — | — | — | 1056 | 15.5× |
| Prod | 8 | — | — | — | 2080 | 7.9× |

MSE decay: ~2.5–2.9× per bit (theory: ~4×).
IP bias: < 0.001 for all b.

### Rust (criterion)

```bash
cd rust && cargo bench --bench benchmarks
```

| Operation | d=256, n=1000 |
|---|---|
| MSE setup | ~36 ms |
| MSE encode | ~42–46 ms |
| MSE decode | ~43–65 ms |
| Prod encode | ~148–157 ms |
| Prod decode | ~128 ms |

---

## Testing

```bash
# Python unit tests + property-based
python -m pytest tests/ -v          # 33 tests

# Python fuzzer
python fuzzer.py --iters 50         # 5 invariants

# Rust tests (without Python bindings)
cd rust && cargo test --no-default-features  # 15 tests + 3 proptest

# Rust fuzzer
cargo run --no-default-features --bin fuzzer -- --iters 50

# Python bindings (via maturin)
maturin develop --release
python -c "from turboquant_rs import TurboQuantMse; print('OK')"
```

Total: **53+ tests** (33 Python + 15 Rust + 5 fuzzer invariants + Python bindings smoke).
All pass CI on every push.

### CI Pipeline

CI automatically runs:
- **Python pure tests** — testing the original Python version
- **Python+Rust integration** — testing Python bindings via maturin
- **Rust tests** — native Rust tests, clippy, fuzzer

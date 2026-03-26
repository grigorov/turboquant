# TurboQuant — Near-Optimal Vector Quantization

Python and Rust implementations of the **TurboQuant** algorithm.
Source: [arxiv.org/html/2504.19874v1](https://arxiv.org/html/2504.19874v1)

---

## Table of Contents

- [What is TurboQuant](#what-is-turboquant)
- [Mathematical Background](#mathematical-background)
  - [Marginal Distribution of Sphere Coordinates](#marginal-distribution-of-sphere-coordinates)
  - [Lloyd-Max Quantizer](#lloyd-max-quantizer)
  - [Inner Product Problem and QJL](#inner-product-problem-and-qjl)
- [Algorithms](#algorithms)
  - [TurboQuantMSE — MSE Minimization](#turboquantmse--mse-minimization)
  - [TurboQuantProd — Unbiased Inner Product](#turboquantprod--unbiased-inner-product)
- [Theoretical Guarantees](#theoretical-guarantees)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Demo Results](#demo-results)
- [Code Structure](#code-structure)
- [Rust Implementation](#rust-implementation)
  - [Installation (Rust)](#installation-rust)
  - [Usage (Rust)](#usage-rust)
  - [API (Rust)](#api-rust)
  - [Code Structure (Rust)](#code-structure-rust)
- [Fuzzer](#fuzzer)
  - [Python](#fuzzer-python)
  - [Rust](#fuzzer-rust)
  - [Checked Invariants](#checked-invariants)

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

Requirements: Python 3.9+, NumPy, SciPy.

```bash
pip install numpy scipy
```

The file `turboquant.py` requires no installation — just copy it into your project.

---

## Usage

### MSE Quantization

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

The Rust implementation lives in the `rust/` directory as a `turboquant` library crate with an executable binary for the demo.

Dependencies: `rand 0.8`, `rand_distr 0.4` — for generating random rotation and QJL matrices.

### Installation (Rust)

Requirements: Rust 1.85+ (edition 2024), Cargo.

```bash
cd rust
cargo build --release
```

Run the demo:

```bash
cargo run --release
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

### Code Structure (Rust)

```
rust/
├── Cargo.toml
└── src/
    ├── lib.rs          # public API: re-exports TurboQuantMse, TurboQuantProd, Qjl
    ├── main.rs         # demo and performance benchmarks
    ├── bin/
    │   └── fuzzer.rs   # correctness fuzzer (5 invariants)
    ├── lloyd.rs        # Lloyd-Max quantizer (Gaussian and sphere variants)
    ├── mse.rs          # TurboQuantMse: rotation + codebook, encode/decode
    ├── qjl.rs          # Qjl: matrix S, sign projection, unbiased decoding
    └── prod.rs         # TurboQuantProd: MSE + QJL residual, IP estimation
```

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

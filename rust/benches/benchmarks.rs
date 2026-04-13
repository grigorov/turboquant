//! Criterion benchmarks for TurboQuant Rust implementation.
//!
//! Run with: cargo bench

use criterion::{criterion_group, criterion_main, Criterion};
use turboquant::{TurboQuantMse, TurboQuantProd};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn bench_mse_setup(c: &mut Criterion) {
    let d = 256;
    for bits in [1, 2, 4, 8] {
        c.bench_function(&format!("mse_setup_d{d}_b{bits}"), |b| {
            b.iter(|| TurboQuantMse::new(d, bits, Some(42)))
        });
    }
}

fn bench_mse_encode(c: &mut Criterion) {
    let d = 256;
    let n = 1000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let normal = Normal::new(0.0, 1.0).unwrap();

    for bits in [1, 2, 4, 8] {
        let q = TurboQuantMse::new(d, bits, Some(42));
        let x: Vec<f64> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
        let x = normalize_batch(&x, n, d);

        c.bench_function(&format!("mse_encode_d{d}_b{bits}_n{n}"), |b| {
            b.iter(|| q.encode_batch(&x))
        });
    }
}

fn bench_mse_decode(c: &mut Criterion) {
    let d = 256;
    let n = 1000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let normal = Normal::new(0.0, 1.0).unwrap();

    for bits in [1, 2, 4, 8] {
        let q = TurboQuantMse::new(d, bits, Some(42));
        let x: Vec<f64> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
        let x = normalize_batch(&x, n, d);
        let idx = q.encode_batch(&x);

        c.bench_function(&format!("mse_decode_d{d}_b{bits}_n{n}"), |b| {
            b.iter(|| q.decode_batch(&idx))
        });
    }
}

fn bench_prod_encode(c: &mut Criterion) {
    let d = 256;
    let n = 1000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let normal = Normal::new(0.0, 1.0).unwrap();

    for bits in [2, 4, 8] {
        let q = TurboQuantProd::new(d, bits, Some(42));
        let x: Vec<f64> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
        let x = normalize_batch(&x, n, d);

        c.bench_function(&format!("prod_encode_d{d}_b{bits}_n{n}"), |b| {
            b.iter(|| q.encode_batch(&x))
        });
    }
}

fn bench_prod_decode(c: &mut Criterion) {
    let d = 256;
    let n = 1000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let normal = Normal::new(0.0, 1.0).unwrap();

    for bits in [2, 4, 8] {
        let q = TurboQuantProd::new(d, bits, Some(42));
        let x: Vec<f64> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
        let x = normalize_batch(&x, n, d);
        let encoded = q.encode_batch(&x);

        c.bench_function(&format!("prod_decode_d{d}_b{bits}_n{n}"), |b| {
            b.iter(|| q.decode_batch(&encoded))
        });
    }
}

// Helper: normalize batch of vectors
fn normalize_batch(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(x.len());
    for i in 0..n {
        let xi = &x[i * d..(i + 1) * d];
        let norm = xi.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-12 {
            result.extend_from_slice(xi);
        } else {
            result.extend(xi.iter().map(|v| v / norm));
        }
    }
    result
}

criterion_group!(
    benches,
    bench_mse_setup,
    bench_mse_encode,
    bench_mse_decode,
    bench_prod_encode,
    bench_prod_decode,
);
criterion_main!(benches);

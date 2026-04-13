//! TurboQuant fuzzer: correctness invariant checker.
//!
//! Invariants:
//!   1. encode→decode roundtrip (length, no panic)
//!   2. MSE monotonicity: MSE(b+1) ≤ MSE(b) × 1.1
//!   3. IP unbiasedness: |mean(est_ip - true_ip)| < 0.05
//!   4. IP variance: var ≤ (π/2d)·‖y‖² × 2.0  [skipped for d<16]
//!   5. Edge cases: b=1/8, d=2/4, zero/near-zero vectors
//!
//! Usage:
//!   cargo run --bin fuzzer
//!   cargo run --bin fuzzer -- --iters 100 --seed 42

use std::f64::consts::PI;
use std::process;

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use turboquant_rs::{TurboQuantMse, TurboQuantProd};

const D_CHOICES: &[usize] = &[4, 8, 16, 32, 64, 128, 256];

// ---------------------------------------------------------------------------
// Вспомогательные функции
// ---------------------------------------------------------------------------

fn make_unit_batch(n: usize, d: usize, seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f64, 1.0).unwrap();
    let mut data: Vec<f64> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
    for i in 0..n {
        let row = &mut data[i * d..(i + 1) * d];
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-12 {
            row.iter_mut().for_each(|v| *v /= norm);
        }
    }
    data
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn mean_f64(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance_f64(xs: &[f64]) -> f64 {
    let m = mean_f64(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
}

// ---------------------------------------------------------------------------
// Invariant checks
// ---------------------------------------------------------------------------

/// 1. encode→decode roundtrip: корректная длина, без паники.
fn check_roundtrip(q: &TurboQuantMse, x_batch: &[f64], n: usize, d: usize) -> (bool, String) {
    let idx = q.encode(x_batch);
    if idx.len() != n * d {
        return (false, format!("idx.len()={} != {}", idx.len(), n * d));
    }
    let x_hat = q.decode(&idx);
    if x_hat.len() != n * d {
        return (false, format!("x_hat.len()={} != {}", x_hat.len(), n * d));
    }
    (true, "OK".to_string())
}

/// 2. MSE монотонность: MSE(b+1) ≤ MSE(b) × 1.1.
fn check_mse_monotone(d: usize, b: usize, x_batch: &[f64], seed: u64) -> (bool, String) {
    if b >= 8 {
        return (true, "skipped(b>=8)".to_string());
    }
    let q_b  = TurboQuantMse::new(d, b,     Some(seed));
    let q_b1 = TurboQuantMse::new(d, b + 1, Some(seed));
    let mse_b  = q_b.mse(x_batch);
    let mse_b1 = q_b1.mse(x_batch);
    if mse_b1 > mse_b * 1.1 {
        return (false, format!("MSE({})={:.5} > MSE({})={:.5} × 1.1", b + 1, mse_b1, b, mse_b));
    }
    (true, format!("MSE({b})={mse_b:.5} MSE({})={mse_b1:.5}", b + 1))
}

/// 3. IP несмещённость: |mean(est_ip - true_ip)| < 0.05.
fn check_ip_bias(q: &TurboQuantProd, x_batch: &[f64], y: &[f64], n: usize, d: usize) -> (bool, String) {
    let errors: Vec<f64> = (0..n).map(|i| {
        let xi = &x_batch[i * d..(i + 1) * d];
        let true_ip = dot(xi, y);
        let qv = q.encode(xi);
        q.inner_product_estimate(y, &qv) - true_ip
    }).collect();

    let bias = mean_f64(&errors);
    if bias.abs() >= 0.05 {
        return (false, format!("ip_bias={bias:+.5}  (threshold=0.05)"));
    }
    (true, format!("ip_bias={bias:+.5}"))
}

/// 4. IP дисперсия: var ≤ (π/2d)·‖y‖² × 2.0. Пропускается для d < 16.
fn check_ip_variance(q: &TurboQuantProd, x_batch: &[f64], y: &[f64], n: usize, d: usize) -> (bool, String) {
    if d < 16 {
        return (true, "skipped(d<16)".to_string());
    }
    let errors: Vec<f64> = (0..n).map(|i| {
        let xi = &x_batch[i * d..(i + 1) * d];
        let true_ip = dot(xi, y);
        let qv = q.encode(xi);
        q.inner_product_estimate(y, &qv) - true_ip
    }).collect();

    let var = variance_f64(&errors);
    let y_norm_sq = dot(y, y);
    let threshold = (PI / (2.0 * d as f64)) * y_norm_sq * 2.0;
    if var > threshold {
        return (false, format!("ip_var={var:.5} > threshold={threshold:.5}"));
    }
    (true, format!("ip_var={var:.5} threshold={threshold:.5}"))
}

/// 5. Граничные случаи: корректная длина вывода, без паники.
/// b=8 вместо b=16 — Lloyd-Max для 65536 центроидов слишком медленный.
fn check_edge_cases(d: usize, seed: u64) -> (bool, String) {
    let mut failures: Vec<String> = Vec::new();

    // b=1 и b=8 для MSE
    for b in [1_usize, 8] {
        let q = TurboQuantMse::new(d, b, Some(seed));
        let mut x = vec![0.0_f64; d];
        x[0] = 1.0;
        let x_hat = q.decode(&q.encode(&x));
        if x_hat.len() != d {
            failures.push(format!("MSE b={b}: len={}", x_hat.len()));
        }
    }

    // b=2 минимум для Prod
    {
        let q = TurboQuantProd::new(d, 2, Some(seed));
        let mut x = vec![0.0_f64; d];
        x[0] = 1.0;
        let qv = q.encode(&x);
        let x_tilde = q.decode(&qv);
        if x_tilde.len() != d {
            failures.push(format!("Prod b=2: len={}", x_tilde.len()));
        }
    }

    // Нулевой вектор
    {
        let q = TurboQuantMse::new(d, 4, Some(seed));
        let x = vec![0.0_f64; d];
        let x_hat = q.decode(&q.encode(&x));
        if x_hat.len() != d {
            failures.push(format!("zero vector: len={}", x_hat.len()));
        }
    }

    // Почти нулевой вектор (норма ~ 1e-13)
    {
        let q = TurboQuantMse::new(d, 4, Some(seed));
        let x = vec![1e-14_f64; d];
        let x_hat = q.decode(&q.encode(&x));
        if x_hat.len() != d {
            failures.push(format!("near-zero: len={}", x_hat.len()));
        }
    }

    if failures.is_empty() {
        (true, "OK".to_string())
    } else {
        (false, format!("failures: {}", failures.join("; ")))
    }
}

// ---------------------------------------------------------------------------
// Одна итерация фаззера
// ---------------------------------------------------------------------------

fn run_iteration(iter_idx: usize, master_rng: &mut rand::rngs::StdRng) -> bool {
    let d   = D_CHOICES[master_rng.gen_range(0..D_CHOICES.len())];
    let b   = master_rng.gen_range(1_usize..9);  // 1..8
    let b_p = master_rng.gen_range(2_usize..9);  // 2..8
    let n   = master_rng.gen_range(500_usize..2001);
    let seed: u64      = master_rng.r#gen();
    let iter_seed: u64 = master_rng.r#gen();

    let x_batch = make_unit_batch(n, d, iter_seed);
    let y = x_batch[..d].to_vec();

    let q_mse  = TurboQuantMse::new(d, b,   Some(seed));
    let q_prod = TurboQuantProd::new(d, b_p, Some(seed));

    let (ok1, msg1) = check_roundtrip(&q_mse, &x_batch, n, d);
    let (ok2, msg2) = check_mse_monotone(d, b, &x_batch, seed);
    let (ok3, msg3) = check_ip_bias(&q_prod, &x_batch, &y, n, d);
    let (ok4, msg4) = check_ip_variance(&q_prod, &x_batch, &y, n, d);
    let (ok5, msg5) = check_edge_cases(d, seed);

    let passed = ok1 && ok2 && ok3 && ok4 && ok5;
    let status = if passed { "PASS" } else { "FAIL" };

    println!("[{status}] iter={iter_idx:02}  d={d}  b={b}/{b_p}  n={n}");
    for (name, ok, msg) in [
        ("roundtrip", ok1, &msg1),
        ("mse_mono",  ok2, &msg2),
        ("ip_bias",   ok3, &msg3),
        ("ip_var",    ok4, &msg4),
        ("edge",      ok5, &msg5),
    ] {
        let marker = if ok { "OK".to_string() } else { format!("FAIL: {msg}") };
        println!("       {name}={marker}");
    }

    passed
}

// ---------------------------------------------------------------------------
// Парсинг аргументов
// ---------------------------------------------------------------------------

fn parse_args() -> (usize, Option<u64>) {
    let args: Vec<String> = std::env::args().collect();
    let mut iters = 50_usize;
    let mut seed: Option<u64> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                iters = args[i + 1].parse().expect("--iters requires a number");
                i += 2;
            }
            "--seed" => {
                seed = Some(args[i + 1].parse().expect("--seed requires a number"));
                i += 2;
            }
            _ => { i += 1; }
        }
    }
    (iters, seed)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let (iters, seed_opt) = parse_args();
    let seed: u64 = seed_opt.unwrap_or_else(|| {
        rand::rngs::StdRng::from_entropy().r#gen()
    });

    println!("Master seed: {seed}  iters: {iters}\n");
    let mut master_rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut failures = 0_usize;
    for i in 1..=iters {
        if !run_iteration(i, &mut master_rng) {
            failures += 1;
        }
    }

    // Прогоны для малых d
    println!("\n--- Small-d edge cases ---");
    for d_small in [2_usize, 4] {
        let (ok, msg) = check_edge_cases(d_small, 0);
        let status = if ok { "PASS" } else { "FAIL" };
        println!("[{status}] d={d_small}  {msg}");
        if !ok {
            failures += 1;
        }
    }

    let total = iters + 2;
    println!("\nResults: {}/{total} passed", total - failures);
    process::exit(if failures == 0 { 0 } else { 1 });
}

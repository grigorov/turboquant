//! TurboQuant — демо и замеры производительности.

use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use turboquant::{TurboQuantMse, TurboQuantProd};
use turboquant::mse::normalize;

fn randn_unit_batch(n: usize, d: usize, seed: u64) -> Vec<f64> {
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

fn main() {
    let d: usize = 256;
    let n: usize = 1_000;

    let x_batch = randn_unit_batch(n, d, 42);

    println!("{}", "=".repeat(60));
    println!("TurboQuant demo  |  d={d}  n={n}");
    println!("{}", "=".repeat(60));

    // ---------------------------------------------------------------
    // TurboQuantMse
    // ---------------------------------------------------------------
    println!("\n--- TurboQuantMse ---");

    for b in [1_usize, 2, 4, 8] {
        let t0 = Instant::now();
        let q = TurboQuantMse::new(d, b, Some(0));
        let t_setup = t0.elapsed();

        let t0 = Instant::now();
        let idx   = q.encode(&x_batch);
        let x_hat = q.decode(&idx);
        let t_enc = t0.elapsed();

        let mse = x_batch.iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / (n * d) as f64;

        println!(
            "  b={b:2}  MSE={mse:.5}  setup={:.2}s  enc+dec={:.1}ms",
            t_setup.as_secs_f64(),
            t_enc.as_secs_f64() * 1000.0,
        );
    }

    // ---------------------------------------------------------------
    // TurboQuantProd — оценка скалярного произведения
    // ---------------------------------------------------------------
    println!("\n--- TurboQuantProd (inner product) ---");

    let mut rng = rand::rngs::StdRng::seed_from_u64(99);
    let normal = Normal::new(0.0_f64, 1.0).unwrap();
    let y_raw: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
    let y = normalize(&y_raw);

    let true_ips: Vec<f64> = (0..n)
        .map(|i| {
            let xi = &x_batch[i * d..(i + 1) * d];
            xi.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
        })
        .collect();

    for b in [2_usize, 4, 8] {
        let q = TurboQuantProd::new(d, b, Some(0));

        let est_ips: Vec<f64> = (0..n)
            .map(|i| {
                let xi = &x_batch[i * d..(i + 1) * d];
                let qv = q.encode(xi);
                q.inner_product_estimate(&y, &qv)
            })
            .collect();

        let bias = est_ips.iter().zip(true_ips.iter()).map(|(e, t)| e - t).sum::<f64>() / n as f64;
        let rmse = (est_ips.iter().zip(true_ips.iter())
            .map(|(e, t)| (e - t).powi(2))
            .sum::<f64>() / n as f64).sqrt();

        println!(
            "  b={b:2}  IP bias={bias:+.5}  RMSE={rmse:.5}  bits/vec={}",
            q.bits_per_vector()
        );
    }

    // ---------------------------------------------------------------
    // Проверка несмещённости (n=5000)
    // ---------------------------------------------------------------
    println!("\n--- Проверка несмещённости (b=4, n=5000) ---");
    let n2 = 5_000;
    let x2 = randn_unit_batch(n2, d, 7);
    let q = TurboQuantProd::new(d, 4, Some(0));

    let true_ips2: Vec<f64> = (0..n2)
        .map(|i| {
            let xi = &x2[i * d..(i + 1) * d];
            xi.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
        })
        .collect();

    let bias2: f64 = (0..n2)
        .map(|i| {
            let xi = &x2[i * d..(i + 1) * d];
            let qv = q.encode(xi);
            q.inner_product_estimate(&y, &qv) - true_ips2[i]
        })
        .sum::<f64>() / n2 as f64;

    println!("  Среднее смещение на {n2} векторах: {bias2:+.6}  (должно быть ≈ 0)");
    println!("\nDone.");
}

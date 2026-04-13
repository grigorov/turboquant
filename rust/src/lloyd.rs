//! Квантизатор Ллойда–Макса для одномерных распределений.
//!
//! Решает задачу: min_{c₁…c_k} Σᵢ ∫ |x − cᵢ|² · f(x) dx
//! итерационным методом (условные математические ожидания).

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Вспомогательные функции для нормального распределения N(0, sigma²)
// ---------------------------------------------------------------------------

/// PDF гауссова распределения N(0, sigma²)
#[inline]
fn gaussian_pdf(x: f64, sigma: f64) -> f64 {
    let c = 1.0 / (sigma * (2.0 * PI).sqrt());
    c * (-(x * x) / (2.0 * sigma * sigma)).exp()
}

/// CDF гауссова распределения N(0, sigma²), через функцию ошибок
#[inline]
fn gaussian_cdf(x: f64, sigma: f64) -> f64 {
    0.5 * (1.0 + libm_erf(x / (sigma * 2.0_f64.sqrt())))
}

/// Приближение функции ошибок erf(x).
/// Формула Абрамовица–Стегуна 7.1.27, точность ~1.5e-7.
///
/// Ключевое условие корректности: сумма коэффициентов a_i = 1.0,
/// что гарантирует erf(0) = 0.
fn libm_erf(x: f64) -> f64 {
    // Коэффициенты A&S 7.1.27 (сумма ≈ 1.0 → erf(0) = 0)
    const P: f64 = 0.3275911;
    const A: [f64; 5] = [
         0.254829592,
        -0.284496736,
         1.421413741,
        -1.453152027,
         1.061405429,
    ];
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    // Полином через схему Горнера: t*(a0 + t*(a1 + ... + t*a4))
    let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    sign * (1.0 - poly * (-ax * ax).exp())
}

// ---------------------------------------------------------------------------
// Lloyd-Max для N(0, sigma²)
// ---------------------------------------------------------------------------

/// Вычисляет оптимальные центроиды квантизатора Ллойда–Макса
/// для распределения N(0, sigma²).
///
/// Возвращает отсортированный вектор из `n_centroids` значений.
pub fn lloyd_max_gaussian(n_centroids: usize, sigma: f64) -> Vec<f64> {
    const MAX_ITER: usize = 300;
    const TOL: f64 = 1e-10;

    // Инициализация: квантили N(0, sigma²)
    let mut centroids: Vec<f64> = (0..n_centroids)
        .map(|i| {
            let p = (i as f64 + 0.5) / n_centroids as f64;
            sigma * probit(p)
        })
        .collect();

    for _ in 0..MAX_ITER {
        let prev = centroids.clone();

        // Границы ячеек: середины между соседними центроидами
        let mut bounds = vec![f64::NEG_INFINITY];
        for i in 0..n_centroids - 1 {
            bounds.push(0.5 * (centroids[i] + centroids[i + 1]));
        }
        bounds.push(f64::INFINITY);

        // Обновление каждого центроида: E[X | lo < X < hi]
        for i in 0..n_centroids {
            let lo = bounds[i];
            let hi = bounds[i + 1];

            let phi_lo = if lo.is_finite() { gaussian_pdf(lo, sigma) } else { 0.0 };
            let phi_hi = if hi.is_finite() { gaussian_pdf(hi, sigma) } else { 0.0 };

            let cdf_lo = if lo.is_finite() { gaussian_cdf(lo, sigma) } else { 0.0 };
            let cdf_hi = if hi.is_finite() { gaussian_cdf(hi, sigma) } else { 1.0 };

            let prob = cdf_hi - cdf_lo;
            if prob > 1e-15 {
                centroids[i] = sigma * sigma * (phi_lo - phi_hi) / prob;
            } else {
                centroids[i] = 0.5 * (lo + hi);
            }
        }

        // Проверка сходимости
        let delta: f64 = centroids
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        if delta < TOL {
            break;
        }
    }

    centroids
}

// ---------------------------------------------------------------------------
// Lloyd-Max для маргинального распределения координат единичной сферы
//
//   f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 − x²)^((d−3)/2)
//
// При d > 50 делегируем гауссовскому приближению.
// ---------------------------------------------------------------------------

/// Логарифм гамма-функции (Ланцош-приближение).
fn ln_gamma(x: f64) -> f64 {
    // Коэффициенты Ланцоша (g=7, n=9)
    let g = 7.0_f64;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Формула отражения: Γ(x)·Γ(1-x) = π/sin(πx)
        PI.ln() - (PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        for (i, &ci) in c[1..].iter().enumerate() {
            a += ci / (x + i as f64 + 1.0);
        }
        let t = x + g + 0.5;
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Нормировочный коэффициент для маргинального PDF сферы.
fn sphere_pdf_coeff(d: usize) -> f64 {
    // Γ(d/2) / (√π · Γ((d-1)/2))
    let ln_c = ln_gamma(d as f64 / 2.0) - 0.5 * PI.ln() - ln_gamma((d as f64 - 1.0) / 2.0);
    ln_c.exp()
}

/// PDF маргинального распределения координаты на S^{d-1}.
#[inline]
fn sphere_pdf(x: f64, coeff: f64, alpha: f64) -> f64 {
    let v = 1.0 - x * x;
    if v <= 0.0 {
        0.0
    } else {
        coeff * v.powf(alpha)
    }
}

/// Численное интегрирование ∫_{lo}^{hi} f(x) dx методом Симпсона.
fn integrate_simpson<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64, n: usize) -> f64 {
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (hi - lo) / n as f64;
    let mut sum = f(lo) + f(hi);
    for i in 1..n {
        let x = lo + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

/// Вычисляет оптимальные центроиды квантизатора Ллойда–Макса
/// для маргинального распределения координат S^{d-1}.
///
/// При d > 50 использует гауссовское приближение N(0, 1/d).
pub fn lloyd_max_sphere(n_centroids: usize, d: usize) -> Vec<f64> {
    const MAX_ITER: usize = 300;
    const TOL: f64 = 1e-10;
    const SIMPSON_N: usize = 200;

    if d > 50 {
        return lloyd_max_gaussian(n_centroids, 1.0 / (d as f64).sqrt());
    }

    let sigma = 1.0 / (d as f64).sqrt();
    let coeff = sphere_pdf_coeff(d);
    let alpha = (d as f64 - 3.0) / 2.0;

    // Инициализация: квантили N(0, 1/d), зажатые в (-0.99, 0.99)
    let mut centroids: Vec<f64> = (0..n_centroids)
        .map(|i| {
            let p = (i as f64 + 0.5) / n_centroids as f64;
            (sigma * probit(p)).clamp(-0.99, 0.99)
        })
        .collect();

    for _ in 0..MAX_ITER {
        let prev = centroids.clone();

        let mut bounds = vec![-1.0_f64];
        for i in 0..n_centroids - 1 {
            bounds.push(0.5 * (centroids[i] + centroids[i + 1]));
        }
        bounds.push(1.0_f64);

        for i in 0..n_centroids {
            let lo = bounds[i];
            let hi = bounds[i + 1];

            let pdf = |x: f64| sphere_pdf(x, coeff, alpha);
            let pdf_x = |x: f64| x * sphere_pdf(x, coeff, alpha);

            let prob = integrate_simpson(&pdf, lo, hi, SIMPSON_N);
            if prob > 1e-15 {
                let mean = integrate_simpson(&pdf_x, lo, hi, SIMPSON_N);
                centroids[i] = mean / prob;
            } else {
                centroids[i] = 0.5 * (lo + hi);
            }
        }

        let delta: f64 = centroids
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        if delta < TOL {
            break;
        }
    }

    centroids
}

// ---------------------------------------------------------------------------
// Probit (обратная CDF нормального распределения) — метод Боде
// ---------------------------------------------------------------------------

/// Приближение probit(p) = Φ⁻¹(p) (рациональное приближение, точность ~1e-9).
pub fn probit(p: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ];

    let p_low  = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lloyd_gaussian_symmetry() {
        let c = lloyd_max_gaussian(4, 1.0);
        assert_eq!(c.len(), 4);
        // Должна быть симметрия: c[0] ≈ -c[3], c[1] ≈ -c[2]
        assert!((c[0] + c[3]).abs() < 1e-6, "не симметрично: {:?}", c);
        assert!((c[1] + c[2]).abs() < 1e-6, "не симметрично: {:?}", c);
    }

    #[test]
    fn test_lloyd_sphere_symmetry() {
        let c = lloyd_max_sphere(4, 256);
        assert_eq!(c.len(), 4);
        assert!((c[0] + c[3]).abs() < 1e-6, "не симметрично: {:?}", c);
    }

    #[test]
    fn test_probit_symmetry() {
        let v = probit(0.975);
        let expected = 1.959964;  // ~1.96 (стандартное нормальное)
        assert!((v - expected).abs() < 1e-4, "probit(0.975) = {v}");
    }

    // -----------------------------------------------------------------------
    // Proptest
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_lloyd_centroids_sorted(n in 2usize..32, sigma in 0.1f64..5.0) {
                let c = lloyd_max_gaussian(n, sigma);
                assert_eq!(c.len(), n);
                for i in 1..n {
                    prop_assert!(c[i] > c[i-1], "центроиды не отсортированы");
                }
            }

            #[test]
            fn prop_lloyd_symmetry(n in 2usize..16) {
                let c = lloyd_max_gaussian(n, 1.0);
                // Симметрия относительно нуля: c[i] ≈ -c[n-1-i]
                for i in 0..n/2 {
                    let diff = (c[i] + c[n - 1 - i]).abs();
                    prop_assert!(diff < 1e-6, "нарушена симметрия: c[{i}]={}", c[i]);
                }
            }

            #[test]
            fn prop_sphere_centroids_in_bounds(n in 2usize..16, d in 4usize..200) {
                let c = lloyd_max_sphere(n, d);
                assert_eq!(c.len(), n);
                for &v in &c {
                    prop_assert!((-1.0..=1.0).contains(&v), "центроид вне [-1,1]: {v}");
                }
            }
        }
    }
}

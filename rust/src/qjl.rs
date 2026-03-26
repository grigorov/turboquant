//! QJL (Quantized Johnson-Lindenstrauss) — несмещённое преобразование для
//! оценки скалярных произведений.
//!
//! Encode:  z = sign(S · x)                 ∈ {−1, +1}^d
//! Decode:  x̃ = (√(π/2) / d) · ‖x‖₂ · Sᵀ · z
//!
//! Свойство: E[⟨y, x̃⟩] = ⟨y, x⟩  для любого фиксированного y.
//! Дисперсия: Var(⟨y, x̃⟩) ≤ (π / 2d) · ‖y‖₂²

use std::f64::consts::PI;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// ---------------------------------------------------------------------------
// Вспомогательное: умножение S·x и Sᵀ·z
// ---------------------------------------------------------------------------

#[inline]
fn matvec(s: &[f64], x: &[f64], d: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; d];
    for i in 0..d {
        y[i] = (0..d).map(|j| s[i * d + j] * x[j]).sum();
    }
    y
}

#[inline]
fn matvec_t(s: &[f64], z: &[i8], d: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; d];
    for j in 0..d {
        y[j] = (0..d).map(|i| s[i * d + j] * z[i] as f64).sum();
    }
    y
}

// ---------------------------------------------------------------------------
// Qjl
// ---------------------------------------------------------------------------

/// Quantized Johnson-Lindenstrauss преобразование.
pub struct Qjl {
    /// Размерность
    pub d: usize,
    /// Матрица проекций S ∈ ℝ^{d×d}, элементы ~ N(0,1), row-major
    s: Vec<f64>,
    /// (√(π/2)) / d
    scale: f64,
}

impl Qjl {
    /// Создаёт новый QJL-преобразователь.
    pub fn new(d: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None    => rand::rngs::StdRng::from_entropy(),
        };
        let normal = Normal::new(0.0, 1.0).unwrap();
        let s: Vec<f64> = (0..d * d).map(|_| normal.sample(&mut rng)).collect();
        let scale = (PI / 2.0).sqrt() / d as f64;
        Self { d, s, scale }
    }

    /// Кодирует вектор x в знаки и сохраняет его L2-норму.
    ///
    /// Возвращает (знаки ±1 как i8, норма как f32).
    pub fn encode(&self, x: &[f64]) -> (Vec<i8>, f32) {
        assert_eq!(x.len(), self.d);
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt() as f32;
        let sx = matvec(&self.s, x, self.d);
        let signs: Vec<i8> = sx.iter().map(|&v| if v >= 0.0 { 1_i8 } else { -1_i8 }).collect();
        (signs, norm)
    }

    /// Декодирует знаки + норму в приближённый вектор.
    ///
    /// x̃ = scale · γ · Sᵀ · z
    pub fn decode(&self, signs: &[i8], norm: f32) -> Vec<f64> {
        assert_eq!(signs.len(), self.d);
        let st_z = matvec_t(&self.s, signs, self.d);
        let gamma = norm as f64;
        st_z.iter().map(|v| v * self.scale * gamma).collect()
    }

    /// Оценка скалярного произведения ⟨y, x⟩ через декодированный x̃.
    pub fn inner_product_estimate(&self, y: &[f64], signs: &[i8], norm: f32) -> f64 {
        let x_tilde = self.decode(signs, norm);
        y.iter().zip(x_tilde.iter()).map(|(a, b)| a * b).sum()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn randn_unit(d: usize, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let v: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_signs_are_pm1() {
        let d = 64;
        let q = Qjl::new(d, Some(0));
        let x = randn_unit(d, 1);
        let (signs, _) = q.encode(&x);
        for s in &signs {
            assert!(*s == 1 || *s == -1, "знаки должны быть ±1");
        }
    }

    /// Проверка несмещённости: E[⟨y, x̃⟩] ≈ ⟨y, x⟩
    #[test]
    fn test_unbiased_inner_product() {
        let d = 128;
        let n_trials = 500;

        let y = randn_unit(d, 42);
        let x = randn_unit(d, 7);
        let true_ip: f64 = y.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let mut sum_est = 0.0_f64;
        for trial in 0..n_trials {
            let q = Qjl::new(d, Some(trial as u64));
            let (signs, norm) = q.encode(&x);
            sum_est += q.inner_product_estimate(&y, &signs, norm);
        }
        let mean_est = sum_est / n_trials as f64;

        // Стандартное отклонение одной оценки ≈ sqrt(π/(2d)) ≈ 0.111 для d=128.
        // Стандартная ошибка среднего (500 попыток) ≈ 0.005.
        // Проверяем абсолютную погрешность (не относительную), порог = 4 сигмы.
        let abs_error = (mean_est - true_ip).abs();
        assert!(abs_error < 0.025,
            "смещение слишком большое: est={mean_est:.4}, true={true_ip:.4}, abs_err={abs_error:.4}");
    }
}

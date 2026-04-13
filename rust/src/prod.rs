//! TurboQuantProd — IP-оптимизированный векторный квантизатор (Алгоритм 2).
//!
//! Комбинирует MSE-квантование (b-1 бит) с QJL-остатком (1 бит/координата),
//! обеспечивая несмещённую оценку скалярного произведения:
//!
//!   E[⟨y, x̃⟩] = ⟨y, x⟩  для любого фиксированного y.
//!
//! Шаги кодирования:
//!   1. MSE-квантование с (b-1) битами: idx, x̂_mse
//!   2. Остаток r = x − x̂_mse
//!   3. QJL на остатке: z = sign(S·r), γ = ‖r‖₂
//!
//! Декодирование:
//!   x̃ = x̂_mse + (√(π/2)/d) · γ · Sᵀ · z

use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::mse::TurboQuantMse;
use crate::qjl::Qjl;

/// Сжатое представление одного вектора.
pub struct QuantizedProd {
    /// Индексы MSE-центроидов (длина d)
    pub mse_indices: Vec<u16>,
    /// QJL-знаки ±1 (длина d)
    pub qjl_signs: Vec<i8>,
    /// L2-норма остатка
    pub residual_norm: f32,
}

/// IP-оптимизированный векторный квантизатор.
pub struct TurboQuantProd {
    /// Размерность
    pub d: usize,
    /// Бит на координату (≥ 2)
    pub b: usize,
    /// MSE-квантизатор с (b-1) битами
    mse: TurboQuantMse,
    /// QJL для остатка
    qjl: Qjl,
}

impl TurboQuantProd {
    /// Создаёт новый IP-квантизатор.
    ///
    /// # Параметры
    /// - `d`    — размерность векторов
    /// - `b`    — общее количество бит на координату (≥ 2)
    /// - `seed` — зерно для воспроизводимости
    pub fn new(d: usize, b: usize, seed: Option<u64>) -> Self {
        assert!(b >= 2, "TurboQuantProd требует b ≥ 2");

        // Независимые зёрна для MSE и QJL
        let (seed_mse, seed_qjl) = match seed {
            Some(s) => (s, s.wrapping_add(0xDEAD_BEEF)),
            None    => {
                let mut rng = rand::rngs::StdRng::from_entropy();
                (rng.r#gen(), rng.r#gen())
            }
        };

        let mse = TurboQuantMse::new(d, b - 1, Some(seed_mse));
        let qjl = Qjl::new(d, Some(seed_qjl));

        Self { d, b, mse, qjl }
    }

    // ------------------------------------------------------------------
    // Кодирование / декодирование одного вектора
    // ------------------------------------------------------------------

    /// Кодирует единичный вектор (длина `d`).
    pub fn encode(&self, x: &[f64]) -> QuantizedProd {
        assert_eq!(x.len(), self.d);

        // Шаг 1: MSE-квантование с (b-1) битами
        let mse_indices = self.mse.encode(x);
        let x_hat = self.mse.decode(&mse_indices);

        // Шаг 2: Остаток r = x − x̂
        let residual: Vec<f64> = x.iter().zip(x_hat.iter()).map(|(a, b)| a - b).collect();

        // Шаг 3: QJL на остатке
        let (qjl_signs, residual_norm) = self.qjl.encode(&residual);

        QuantizedProd { mse_indices, qjl_signs, residual_norm }
    }

    /// Декодирует сжатое представление обратно в вектор.
    ///
    /// x̃ = x̂_mse + (√(π/2)/d) · γ · Sᵀ · z
    pub fn decode(&self, q: &QuantizedProd) -> Vec<f64> {
        let x_hat_mse = self.mse.decode(&q.mse_indices);
        let residual_hat = self.qjl.decode(&q.qjl_signs, q.residual_norm);

        x_hat_mse.iter().zip(residual_hat.iter()).map(|(a, b)| a + b).collect()
    }

    // ------------------------------------------------------------------
    // Батчевые версии
    // ------------------------------------------------------------------

    /// Кодирует батч из n векторов (row-major, длина n*d).
    pub fn encode_batch(&self, x: &[f64]) -> Vec<QuantizedProd> {
        assert_eq!(x.len() % self.d, 0);
        let n = x.len() / self.d;
        (0..n)
            .into_par_iter()
            .map(|i| self.encode(&x[i * self.d..(i + 1) * self.d]))
            .collect()
    }

    /// Декодирует батч.
    pub fn decode_batch(&self, qs: &[QuantizedProd]) -> Vec<f64> {
        qs.par_iter()
            .flat_map(|q| self.decode(q))
            .collect()
    }

    // ------------------------------------------------------------------
    // Оценка скалярного произведения
    // ------------------------------------------------------------------

    /// Вычисляет несмещённую оценку ⟨y, x⟩ из сжатого x.
    ///
    /// E[результат] = ⟨y, x⟩
    pub fn inner_product_estimate(&self, y: &[f64], q: &QuantizedProd) -> f64 {
        let x_tilde = self.decode(q);
        y.iter().zip(x_tilde.iter()).map(|(a, b)| a * b).sum()
    }

    // ------------------------------------------------------------------
    // Метаданные
    // ------------------------------------------------------------------

    /// Количество бит для хранения одного сжатого вектора.
    ///
    /// (b-1)*d бит для MSE + d бит для QJL + 32 бита для нормы остатка.
    pub fn bits_per_vector(&self) -> usize {
        (self.b - 1) * self.d + self.d + 32
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
    fn test_decode_shape() {
        let d = 64;
        let q = TurboQuantProd::new(d, 4, Some(0));
        let x = randn_unit(d, 1);
        let qv = q.encode(&x);
        let x_tilde = q.decode(&qv);
        assert_eq!(x_tilde.len(), d);
    }

    #[test]
    fn test_bits_per_vector() {
        let d = 256;
        let b = 4;
        let q = TurboQuantProd::new(d, b, Some(0));
        // (b-1)*d + d + 32 = b*d + 32
        assert_eq!(q.bits_per_vector(), b * d + 32);
    }

    /// Проверка несмещённости: среднее смещение оценки IP близко к нулю
    #[test]
    fn test_unbiased_ip() {
        let d = 128;
        let n = 200;

        let y = randn_unit(d, 99);
        let q = TurboQuantProd::new(d, 4, Some(0));

        let mut bias_sum = 0.0_f64;
        for i in 0..n {
            let x = randn_unit(d, i as u64 + 1000);
            let true_ip: f64 = y.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let qv = q.encode(&x);
            let est_ip = q.inner_product_estimate(&y, &qv);
            bias_sum += est_ip - true_ip;
        }
        let mean_bias = bias_sum / n as f64;

        assert!(mean_bias.abs() < 0.05,
            "среднее смещение слишком большое: {mean_bias:.4}");
    }

    #[test]
    fn test_batch_encode_decode() {
        let d = 64;
        let n = 10;
        let q = TurboQuantProd::new(d, 4, Some(0));
        let xs: Vec<f64> = (0..n).flat_map(|i| randn_unit(d, i as u64)).collect();

        let batch = q.encode_batch(&xs);
        assert_eq!(batch.len(), n);

        let decoded = q.decode_batch(&batch);
        assert_eq!(decoded.len(), n * d);
    }
}

//! TurboQuantMSE — MSE-оптимизированный векторный квантизатор (Алгоритм 1).
//!
//! Шаги:
//! 1. Случайная ортогональная матрица поворота Π через QR-разложение.
//! 2. Оптимальный кодебук через квантизатор Ллойда–Макса для маргинала S^{d-1}.
//! 3. encode: y = Π·x → ближайший центроид на каждой координате.
//! 4. decode: поднять центроиды → x̃ = Πᵀ·ỹ.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::lloyd::lloyd_max_sphere;

// ---------------------------------------------------------------------------
// QR-разложение (Хаусхолдеровы отражения, без внешних зависимостей)
// ---------------------------------------------------------------------------

/// Вычисляет Q из QR-разложения матрицы A (in-place, алгоритм Хаусхолдера).
/// Возвращает ортогональную матрицу Q размером n×n.
fn qr_decompose(a: &[f64], n: usize) -> Vec<f64> {
    // Работаем с копией в row-major порядке
    let mut r = a.to_vec();
    // Q начинается как единичная матрица
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }

    for k in 0..n {
        // Вектор x = R[k:, k]
        let len = n - k;
        let mut x: Vec<f64> = (0..len).map(|i| r[(k + i) * n + k]).collect();

        // norm2(x)
        let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_x < 1e-14 {
            continue;
        }

        // v = x + sign(x[0]) * norm(x) * e_1
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * norm_x;
        let norm_v = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_v < 1e-14 {
            continue;
        }
        let v: Vec<f64> = x.iter().map(|v| v / norm_v).collect();

        // R ← (I - 2vvᵀ) · R  (применяем к нижней правой части)
        for i in k..n {
            let dot: f64 = (0..len).map(|j| v[j] * r[(k + j) * n + i]).sum();
            for j in 0..len {
                r[(k + j) * n + i] -= 2.0 * dot * v[j];
            }
        }

        // Q ← Q · (I - 2vvᵀ)  (применяем к правым len столбцам)
        for i in 0..n {
            let dot: f64 = (0..len).map(|j| v[j] * q[i * n + (k + j)]).sum();
            for j in 0..len {
                q[i * n + (k + j)] -= 2.0 * dot * v[j];
            }
        }
    }

    // Корректировка знаков: умножаем столбцы Q на sign(diag(R))
    for k in 0..n {
        let s = if r[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
        for i in 0..n {
            q[i * n + k] *= s;
        }
    }

    q
}

// ---------------------------------------------------------------------------
// Матричное умножение y = A · x  (row-major A размером m×n)
// ---------------------------------------------------------------------------

#[inline]
fn matvec(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; m];
    for i in 0..m {
        y[i] = (0..n).map(|j| a[i * n + j] * x[j]).sum();
    }
    y
}

/// Умножение Aᵀ · x
#[inline]
fn matvec_t(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for j in 0..n {
        y[j] = (0..m).map(|i| a[i * n + j] * x[i]).sum();
    }
    y
}

// ---------------------------------------------------------------------------
// TurboQuantMse
// ---------------------------------------------------------------------------

/// MSE-оптимизированный векторный квантизатор.
///
/// # Пример
/// ```
/// use turboquant::TurboQuantMse;
/// use turboquant::mse::normalize;
///
/// let q = TurboQuantMse::new(64, 4, Some(42));
/// let x: Vec<f64> = (0..64).map(|i| i as f64).collect();
/// let x = normalize(&x);
/// let idx = q.encode(&x);
/// let x_hat = q.decode(&idx);
/// ```
pub struct TurboQuantMse {
    /// Размерность
    pub d: usize,
    /// Бит на координату
    pub b: usize,
    /// Количество центроидов = 2^b
    pub n_centroids: usize,
    /// Ортогональная матрица поворота Π, row-major (d×d)
    rotation: Vec<f64>,
    /// Оптимальные центроиды (длина n_centroids, отсортированы)
    centroids: Vec<f64>,
}

impl TurboQuantMse {
    /// Создаёт новый квантизатор.
    ///
    /// # Параметры
    /// - `d`    — размерность векторов
    /// - `b`    — бит на координату (1–16)
    /// - `seed` — фиксированное зерно для воспроизводимости
    pub fn new(d: usize, b: usize, seed: Option<u64>) -> Self {
        assert!(b >= 1 && b <= 16, "b должен быть в диапазоне [1, 16]");

        let n_centroids = 1_usize << b;

        // Случайная ортогональная матрица через QR-разложение случайной N(0,1) матрицы
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None    => rand::rngs::StdRng::from_entropy(),
        };
        let normal = Normal::new(0.0, 1.0).unwrap();
        let a: Vec<f64> = (0..d * d).map(|_| normal.sample(&mut rng)).collect();
        let rotation = qr_decompose(&a, d);

        // Оптимальные центроиды Ллойда–Макса для маргинала S^{d-1}
        let centroids = lloyd_max_sphere(n_centroids, d);

        Self { d, b, n_centroids, rotation, centroids }
    }

    /// Кодирует батч векторов в индексы центроидов.
    ///
    /// # Параметры
    /// - `x` — срез единичных векторов длиной `d` (один вектор)
    ///          или длиной `n * d` (батч из n векторов, row-major)
    ///
    /// Возвращает вектор u16-индексов той же длины что и `x`.
    pub fn encode(&self, x: &[f64]) -> Vec<u16> {
        assert_eq!(x.len() % self.d, 0, "длина x должна делиться на d");
        let n = x.len() / self.d;
        let mut indices = Vec::with_capacity(x.len());

        for i in 0..n {
            let xi = &x[i * self.d..(i + 1) * self.d];
            // y = Π · x
            let y = matvec(&self.rotation, xi, self.d, self.d);
            // Ближайший центроид для каждой координаты
            for &yj in &y {
                let idx = self.nearest_centroid(yj);
                indices.push(idx as u16);
            }
        }

        indices
    }

    /// Декодирует индексы центроидов обратно в векторы.
    ///
    /// # Параметры
    /// - `indices` — вектор u16-индексов (длина `n * d`)
    ///
    /// Возвращает реконструированные векторы той же длины (row-major).
    pub fn decode(&self, indices: &[u16]) -> Vec<f64> {
        assert_eq!(indices.len() % self.d, 0);
        let n = indices.len() / self.d;
        let mut result = Vec::with_capacity(indices.len());

        for i in 0..n {
            let idx_i = &indices[i * self.d..(i + 1) * self.d];
            // Восстановить центроиды
            let y_hat: Vec<f64> = idx_i.iter().map(|&k| self.centroids[k as usize]).collect();
            // x̃ = Πᵀ · ỹ
            let x_hat = matvec_t(&self.rotation, &y_hat, self.d, self.d);
            result.extend_from_slice(&x_hat);
        }

        result
    }

    /// Кодирует произвольный (ненормированный) вектор.
    /// Сохраняет L2-норму отдельно.
    /// Возвращает (indices, norm).
    pub fn encode_with_norm(&self, x: &[f64]) -> (Vec<u16>, f32) {
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt() as f32;
        let scale = if norm > 1e-12 { 1.0 / norm as f64 } else { 1.0 };
        let x_unit: Vec<f64> = x.iter().map(|v| v * scale).collect();
        (self.encode(&x_unit), norm)
    }

    /// Декодирует вектор, закодированный через `encode_with_norm`.
    pub fn decode_with_norm(&self, indices: &[u16], norm: f32) -> Vec<f64> {
        let x_unit = self.decode(indices);
        x_unit.iter().map(|v| v * norm as f64).collect()
    }

    /// Среднее MSE на батче единичных векторов.
    pub fn mse(&self, x: &[f64]) -> f64 {
        let x_hat = self.decode(&self.encode(x));
        let n = x.len();
        x.iter().zip(x_hat.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64
    }

    // ------------------------------------------------------------------
    // Внутренние вспомогательные методы
    // ------------------------------------------------------------------

    /// Бинарный поиск ближайшего центроида для значения y.
    #[inline]
    fn nearest_centroid(&self, y: f64) -> usize {
        // Центроиды отсортированы → двоичный поиск
        match self.centroids.binary_search_by(|c| c.partial_cmp(&y).unwrap()) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) if i >= self.n_centroids => self.n_centroids - 1,
            Err(i) => {
                let d_left  = (y - self.centroids[i - 1]).abs();
                let d_right = (y - self.centroids[i]).abs();
                if d_left <= d_right { i - 1 } else { i }
            }
        }
    }

}

// ---------------------------------------------------------------------------
// Вспомогательная функция нормализации
// ---------------------------------------------------------------------------

/// Нормализует вектор до единичной L2-нормы.
pub fn normalize(x: &[f64]) -> Vec<f64> {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return x.to_vec();
    }
    x.iter().map(|v| v / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_unit_vec(d: usize, seed_offset: u64) -> Vec<f64> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed_offset);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let v: Vec<f64> = (0..d).map(|_| normal.sample(&mut rng)).collect();
        normalize(&v)
    }

    #[test]
    fn test_encode_decode_shape() {
        let d = 32;
        let q = TurboQuantMse::new(d, 4, Some(0));
        let x = make_unit_vec(d, 1);
        let idx = q.encode(&x);
        assert_eq!(idx.len(), d);
        let x_hat = q.decode(&idx);
        assert_eq!(x_hat.len(), d);
    }

    #[test]
    fn test_mse_decreases_with_bits() {
        let d = 128;
        let n = 50;
        let xs: Vec<f64> = (0..n).flat_map(|i| make_unit_vec(d, i as u64)).collect();

        let mse4 = TurboQuantMse::new(d, 4, Some(0)).mse(&xs);
        let mse8 = TurboQuantMse::new(d, 8, Some(0)).mse(&xs);

        assert!(mse8 < mse4, "MSE должно убывать при увеличении b: {mse4} vs {mse8}");
    }

    #[test]
    fn test_encode_with_norm() {
        let d = 64;
        let q = TurboQuantMse::new(d, 4, Some(42));
        let x: Vec<f64> = (0..d).map(|i| i as f64).collect();
        let (idx, norm) = q.encode_with_norm(&x);
        let x_rec = q.decode_with_norm(&idx, norm);
        assert_eq!(x_rec.len(), d);
        let orig_norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rec_norm  = x_rec.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Допускаем до 2% погрешности нормы: вектор [0,1,...,d-1] имеет
        // неравномерное распределение компонент, что даёт чуть больший
        // квантизационный шум, чем у случайного единичного вектора.
        assert!((orig_norm - rec_norm as f64).abs() / orig_norm < 0.02,
            "норма должна сохраняться: {orig_norm} vs {rec_norm}");
    }
}

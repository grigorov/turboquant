//! TurboQuant — Near-Optimal Vector Quantization
//!
//! Реализация алгоритма TurboQuant по статье:
//! <https://arxiv.org/html/2504.19874v1>
//!
//! Два режима квантования:
//! - [`TurboQuantMse`]  — минимизация MSE реконструкции
//! - [`TurboQuantProd`] — несмещённая оценка скалярного произведения

pub mod lloyd;
pub mod mse;
pub mod qjl;
pub mod prod;

pub use mse::TurboQuantMse;
pub use prod::TurboQuantProd;
pub use qjl::Qjl;

// ---------------------------------------------------------------------------
// Python bindings via PyO3 (only when "extension-module" feature is enabled)
// ---------------------------------------------------------------------------

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::exceptions::PyValueError;

/// MSE-optimized vector quantizer for Python.
#[cfg(feature = "extension-module")]
#[pyclass(name = "TurboQuantMse")]
struct PyTurboQuantMse {
    inner: TurboQuantMse,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyTurboQuantMse {
    #[new]
    #[pyo3(signature = (d, b, seed=None))]
    fn new(d: usize, b: usize, seed: Option<u64>) -> Self {
        Self {
            inner: TurboQuantMse::new(d, b, seed),
        }
    }

    /// Encode a list of f64 values (length must be multiple of d).
    /// Returns list of u16 indices.
    fn encode(&self, x: Vec<f64>) -> PyResult<Vec<u16>> {
        if !x.len().is_multiple_of(self.inner.d) {
            return Err(PyValueError::new_err("length of x must be divisible by d"));
        }
        Ok(self.inner.encode(&x))
    }

    /// Decode a list of u16 indices back to f64 values.
    fn decode(&self, indices: Vec<u16>) -> PyResult<Vec<f64>> {
        if !indices.len().is_multiple_of(self.inner.d) {
            return Err(PyValueError::new_err("length of indices must be divisible by d"));
        }
        Ok(self.inner.decode(&indices))
    }

    /// Encode with norm preservation. Returns (indices, norm).
    fn encode_with_norm(&self, x: Vec<f64>) -> PyResult<(Vec<u16>, f32)> {
        if !x.len().is_multiple_of(self.inner.d) {
            return Err(PyValueError::new_err("length of x must be divisible by d"));
        }
        let n = x.len() / self.inner.d;
        if n == 1 {
            let (indices, norm) = self.inner.encode_with_norm(&x);
            Ok((indices, norm))
        } else {
            // For batch, encode each vector separately
            let mut all_indices = Vec::new();
            let mut norm = 0.0f32;
            for i in 0..n {
                let xi = &x[i * self.inner.d..(i + 1) * self.inner.d];
                let (idx, nrm) = self.inner.encode_with_norm(xi);
                all_indices.extend_from_slice(&idx);
                norm = nrm; // last norm
            }
            Ok((all_indices, norm))
        }
    }

    /// Decode with norm preservation.
    fn decode_with_norm(&self, indices: Vec<u16>, norm: f32) -> PyResult<Vec<f64>> {
        if !indices.len().is_multiple_of(self.inner.d) {
            return Err(PyValueError::new_err("length of indices must be divisible by d"));
        }
        Ok(self.inner.decode_with_norm(&indices, norm))
    }

    /// Calculate MSE on a batch of unit vectors.
    fn mse(&self, x: Vec<f64>) -> PyResult<f64> {
        if !x.len().is_multiple_of(self.inner.d) {
            return Err(PyValueError::new_err("length of x must be divisible by d"));
        }
        Ok(self.inner.mse(&x))
    }

    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }

    #[getter]
    fn b(&self) -> usize {
        self.inner.b
    }

    #[getter]
    fn n_centroids(&self) -> usize {
        self.inner.n_centroids
    }
}

/// IP-optimized vector quantizer with unbiased inner product estimation.
#[cfg(feature = "extension-module")]
#[pyclass(name = "TurboQuantProd")]
struct PyTurboQuantProd {
    inner: TurboQuantProd,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyTurboQuantProd {
    #[new]
    #[pyo3(signature = (d, b, seed=None))]
    fn new(d: usize, b: usize, seed: Option<u64>) -> Self {
        Self {
            inner: TurboQuantProd::new(d, b, seed),
        }
    }

    /// Encode a single unit vector. Returns QuantizedProd.
    fn encode(&self, x: Vec<f64>) -> PyResult<PyQuantizedProd> {
        if x.len() != self.inner.d {
            return Err(PyValueError::new_err("length of x must equal d"));
        }
        let q = self.inner.encode(&x);
        Ok(PyQuantizedProd { inner: q })
    }

    /// Decode a QuantizedProd back to a vector.
    fn decode(&self, q: &PyQuantizedProd) -> PyResult<Vec<f64>> {
        Ok(self.inner.decode(&q.inner))
    }

    /// Estimate inner product between y and quantized x.
    fn inner_product_estimate(&self, y: Vec<f64>, q: &PyQuantizedProd) -> PyResult<f64> {
        if y.len() != self.inner.d {
            return Err(PyValueError::new_err("length of y must equal d"));
        }
        Ok(self.inner.inner_product_estimate(&y, &q.inner))
    }

    #[getter]
    fn d(&self) -> usize {
        self.inner.d
    }

    #[getter]
    fn b(&self) -> usize {
        self.inner.b
    }
}

/// Compressed representation of a vector.
#[cfg(feature = "extension-module")]
#[pyclass(name = "QuantizedProd")]
struct PyQuantizedProd {
    inner: prod::QuantizedProd,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyQuantizedProd {
    #[getter]
    fn mse_indices(&self) -> Vec<u16> {
        self.inner.mse_indices.clone()
    }

    #[getter]
    fn qjl_signs(&self) -> Vec<i8> {
        self.inner.qjl_signs.clone()
    }

    #[getter]
    fn residual_norm(&self) -> f32 {
        self.inner.residual_norm
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantizedProd(mse_indices={}, qjl_signs={}, residual_norm={:.4})",
            self.inner.mse_indices.len(),
            self.inner.qjl_signs.len(),
            self.inner.residual_norm
        )
    }
}

/// Python module definition.
#[cfg(feature = "extension-module")]
#[pymodule]
mod turboquant_rs {
    #[pymodule_export]
    use super::PyTurboQuantMse;

    #[pymodule_export]
    use super::PyTurboQuantProd;

    #[pymodule_export]
    use super::PyQuantizedProd;
}

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

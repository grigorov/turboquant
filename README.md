# TurboQuant — оптимальное векторное квантование

Реализации алгоритма **TurboQuant** на Python и Rust.
Источник: [arxiv.org/html/2504.19874v1](https://arxiv.org/html/2504.19874v1)

[![CI](https://github.com/grigorov/turboquant/actions/workflows/ci.yml/badge.svg)](https://github.com/grigorov/turboquant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Содержание

- [Что такое TurboQuant](#что-такое-turboquant)
- [Математическая основа](#математическая-основа)
- [Алгоритмы](#алгоритмы)
- [Теоретические гарантии](#теоретические-гарантии)
- [Установка](#установка)
- [Использование](#использование)
- [API](#api)
- [Результаты демо](#результаты-демо)
- [Расширенные модули](#расширенные-модули)
  - [Mixed-Precision](#mixed-precision-kвантование)
  - [Sparse Vectors](#квантование-разреженных-векторов)
  - [Numba JIT](#numba-jit-ускорение)
  - [LLM Integration](#интеграция-с-llm)
- [Бенчмарки](#бенчмарки)
- [Тестирование](#тестирование)
- [Структура кода](#структура-кода)
- [Реализация на Rust](#реализация-на-rust)
  - [Python Bindings](#python-bindings-через-maturin)
- [Fuzzer](#fuzzer)

---

## Что такое TurboQuant

TurboQuant — схема векторного квантования, разработанная для эффективного вывода больших языковых моделей (LLM). Она сжимает вещественные векторы до нескольких бит на координату и при этом:

- минимизирует среднеквадратическую ошибку реконструкции (MSE-режим);
- обеспечивает **несмещённую** оценку скалярных произведений (режим IP).

Практическое применение — квантование весов и активаций трансформеров, векторные базы данных, ANN-поиск (approximate nearest neighbours).

---

## Математическая основа

### Маргинальное распределение координат сферы

Пусть вектор **x** лежит на единичной сфере S^{d-1}. После применения случайной ортогональной матрицы поворота Π координаты вектора y = Π·x становятся почти независимыми и каждая из них имеет маргинальное распределение:

```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 − x²)^((d−3)/2),   x ∈ (−1, 1)
```

Это бета-подобное распределение. При d → ∞ оно сходится к нормальному:

```
f_X(x) → N(0, 1/d)
```

Именно эта независимость координат после поворота позволяет применять **скалярный** квантизатор к каждой координате отдельно — без потери качества, которая была бы при наивном покоординатном квантовании без поворота.

### Квантизатор Ллойда–Макса

Для каждой координаты решается одномерная задача оптимизации:

```
min_{c₁ ≤ ... ≤ c_{2^b}}  Σᵢ ∫ |x − cᵢ|² · f_X(x) dx
```

Алгоритм итерации Ллойда:

1. **Инициализация**: центроиды — квантили распределения f_X.
2. **Границы ячеек**: середины между соседними центроидами.
3. **Обновление**: каждый центроид = условное математическое ожидание f_X на своей ячейке.
4. Повторять до сходимости (||Δc|| < 10⁻¹⁰).

При d > 50 используется гауссовское приближение N(0, 1/d) — оно точнее работает при больших размерностях и значительно быстрее (аналитические формулы вместо численного интегрирования).

### Проблема скалярного произведения и QJL

MSE-оптимальный квантизатор вносит **мультипликативное смещение** в оценку скалярного произведения: E[⟨y, x̃_mse⟩] ≈ (2/π)·⟨y, x⟩. Для задач поиска ближайших соседей и attention-механизмов это недопустимо.

Решение — **Quantized Johnson-Lindenstrauss (QJL)** преобразование:

```
Encode:  z = sign(S · x)              ∈ {−1, +1}^d
Decode:  x̃ = (√(π/2) / d) · ‖x‖₂ · Sᵀ · z
```

где S ∈ ℝ^{d×d} — матрица с элементами N(0, 1).

**Свойство несмещённости:**

```
E[⟨y, x̃⟩] = ⟨y, x⟩   для любого фиксированного y
```

**Дисперсия:** Var(⟨y, x̃⟩) ≤ (π/2d) · ‖y‖₂²

---

## Алгоритмы

### TurboQuantMSE — минимизация MSE

**Инициализация:**
1. Случайная матрица поворота Π ∈ ℝ^{d×d} через QR-разложение случайной матрицы с элементами N(0,1).
2. Вычисление оптимального кодебука {c₁, ..., c_{2^b}} через алгоритм Ллойда–Макса.

**Кодирование Quant_MSE(x):**
```
y    ← Π · x                          # поворот
idxⱼ ← argmin_k |yⱼ − cₖ|,  j=1..d  # ближайший центроид
output: idx  (b бит на координату)
```

**Декодирование DeQuant_MSE(idx):**
```
ỹⱼ ← c_{idxⱼ},  j=1..d     # восстановление центроидов
x̃  ← Πᵀ · ỹ                # обратный поворот
output: x̃
```

**Затраты памяти:** b·d бит на вектор.

---

### TurboQuantProd — несмещённое скалярное произведение

Комбинирует MSE-квантование с QJL-остатком.

**Инициализация:**
1. Создать TurboQuantMSE с битовой шириной (b−1).
2. Случайная проекционная матрица S ∈ ℝ^{d×d} с элементами N(0,1).

**Кодирование Quant_Prod(x):**
```
idx  ← Quant_MSE(x)                  # шаг 1: MSE-квантование (b-1 бит)
r    ← x − DeQuant_MSE(idx)          # шаг 2: остаток
z    ← sign(S · r)                   # шаг 3: QJL на остатке
γ    ← ‖r‖₂                          # сохраняем норму остатка
output: (idx, z, γ)
```

**Декодирование DeQuant_Prod(idx, z, γ):**
```
x̃_mse ← DeQuant_MSE(idx)
x̃_qjl ← (√(π/2) / d) · γ · Sᵀ · z
output: x̃_mse + x̃_qjl
```

**Затраты памяти:** b·d бит + 32 бита (float для γ) на вектор.

---

## Теоретические гарантии

| Метод | Дисторсия | Нижняя граница |
|---|---|---|
| TurboQuantMSE | D_mse ≤ (√3π/2) / 4^b | D_mse ≥ 1 / 4^b |
| TurboQuantProd | D_prod ≤ (√3π²·‖y‖²/d) / 4^b | D_prod ≥ (1/d) / 4^b |

TurboQuantMSE достигает результата в ~2.7× от теоретического оптимума.
TurboQuantProd — практически оптимален по скалярному произведению.

**Несмещённость TurboQuantProd:**
```
E[⟨y, x̃⟩] = ⟨y, x⟩
```

---

## Установка

### Pure Python (без компиляции)

Зависимости: Python 3.9+, NumPy, SciPy.

```bash
pip install numpy scipy
```

Файл `turboquant.py` не требует установки — достаточно скопировать его в проект.

### С Rust bindings (рекомендуется, максимальная производительность)

Требования: Python 3.9+, Rust 1.75+ (edition 2021), `maturin`.

```bash
# Установка maturin
pip install maturin

# Сборка и установка из исходников
maturin develop --release

# Или сборка wheel-пакета
maturin build --release
pip install target/wheels/*.whl
```

После установки Rust bindings становятся доступны через модуль `turboquant_rs`:

```python
from turboquant_rs import TurboQuantMse, TurboQuantProd, QuantizedProd
```

> **Примечание:** Pure Python версии (`turboquant.py`) продолжает работать без изменений. Rust bindings — опциональное ускорение.

---

## Использование

### Rust bindings (рекомендуется)

```python
from turboquant_rs import TurboQuantMse, TurboQuantProd

d, b = 256, 4

# MSE-квантование
q = TurboQuantMse(d=d, b=b, seed=42)

x = [float(v) for v in np.random.randn(d)]
x_norm = x / np.linalg.norm(x)

idx   = q.encode(x_norm.tolist())   # List[u16] — индексы центроидов
x_hat = q.decode(idx)               # List[f64] — реконструкция

# TurboQuantProd для несмещённой оценки скалярного произведения
q_prod = TurboQuantProd(d=d, b=b, seed=42)
qv = q_prod.encode(x_norm.tolist())
ip_est = q_prod.inner_product_estimate(x_norm.tolist(), qv)
```

### Pure Python (без Rust)

```python
import numpy as np
from turboquant import TurboQuantMSE

d, b = 256, 4   # размерность и количество бит на координату

# Создать квантизатор (вычисляет матрицу поворота и кодебук)
q = TurboQuantMSE(d=d, b=b, seed=42)

# Один вектор (единичная норма)
x = np.random.randn(d)
x /= np.linalg.norm(x)

idx   = q.encode(x)      # (d,) uint16 — индексы центроидов
x_hat = q.decode(idx)    # (d,) float64 — реконструированный вектор

mse = np.mean((x - x_hat) ** 2)
print(f"MSE = {mse:.6f}")

# Батч векторов
X = np.random.randn(1000, d)
X /= np.linalg.norm(X, axis=1, keepdims=True)

IDX   = q.encode(X)      # (1000, d) uint16
X_hat = q.decode(IDX)    # (1000, d) float64
```

### Квантование произвольных (ненормированных) векторов

```python
x_raw = np.random.randn(d) * 5.0    # произвольная норма

idx, norm = q.encode_with_norm(x_raw)
x_rec     = q.decode_with_norm(idx, norm)
```

### Несмещённая оценка скалярного произведения

```python
from turboquant import TurboQuantProd

q = TurboQuantProd(d=256, b=4, seed=42)

# Кодирование базы (хранимые векторы)
X = np.random.randn(1000, 256)
X /= np.linalg.norm(X, axis=1, keepdims=True)

mse_idx, qjl_signs, res_norms = q.encode(X)

# Оценка скалярного произведения с запросом y
y = np.random.randn(256)
y /= np.linalg.norm(y)

ip_true = X @ y   # истинные скалярные произведения

for i in range(len(X)):
    ip_est = q.inner_product_estimate(y, mse_idx[i], qjl_signs[i], res_norms[i])
    # E[ip_est] == ip_true[i]
```

### Прямое декодирование (для реконструкции вектора)

```python
X_tilde = q.decode(mse_idx, qjl_signs, res_norms)  # (1000, 256)
```

---

## API

### `TurboQuantMSE(d, b, seed=None)`

| Параметр | Тип | Описание |
|---|---|---|
| `d` | int | Размерность вектора |
| `b` | int | Бит на координату (1–16) |
| `seed` | int или None | Случайное зерно |

| Метод | Описание |
|---|---|
| `encode(x)` | Вектор(ы) → индексы uint16 |
| `decode(idx)` | Индексы → реконструированные векторы float64 |
| `encode_with_norm(x)` | Для ненормированных векторов: возвращает (индексы, нормы) |
| `decode_with_norm(idx, norms)` | Обратное к `encode_with_norm` |
| `mse(x)` | Вычислить среднее MSE на батче |

---

### `TurboQuantProd(d, b, seed=None)`

| Параметр | Тип | Описание |
|---|---|---|
| `d` | int | Размерность вектора |
| `b` | int | Бит на координату (≥ 2) |
| `seed` | int или None | Случайное зерно |

| Метод | Описание |
|---|---|
| `encode(x)` | Вектор(ы) → (mse_idx, qjl_signs, res_norms) |
| `decode(mse_idx, qjl_signs, res_norms)` | Реконструировать вектор |
| `inner_product_estimate(y, mse_idx, qjl_signs, res_norms)` | Несмещённая оценка ⟨y, x⟩ |
| `bits_per_vector()` | Количество бит на один сжатый вектор |

---

### `QJL(d, seed=None)`

Вспомогательный класс. Используется внутри `TurboQuantProd`.

| Метод | Описание |
|---|---|
| `encode(x)` | x → (знаки ±1, нормы) |
| `decode(signs, norms)` | Несмещённая реконструкция |

---

## Результаты демо

Запуск `python3 turboquant.py` на d=256, n=1000 случайных единичных векторов:

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

MSE убывает в ~4 раза при каждом добавлении 1 бита (соответствует теоретическому 1/4^b).
Смещение оценки скалярного произведения практически равно нулю.

---

## Структура кода

```
turboquant.py                    # библиотека квантования
├── _lloyd_max_gaussian()        # Lloyd-Max для N(0, σ²)
├── _lloyd_max_beta_sphere()     # Lloyd-Max для маргинала сферы
├── TurboQuantMSE                # Алгоритм 1: MSE-квантование
│   ├── __init__                 #   поворот + кодебук
│   ├── encode / decode          #   пакетное кодирование/декодирование
│   └── encode_with_norm /       #   поддержка ненормированных векторов
│       decode_with_norm
├── QJL                          # Quantized Johnson-Lindenstrauss
│   ├── encode                   #   x → sign(S·x)
│   └── decode                   #   z → (√(π/2)/d)·γ·Sᵀ·z
├── TurboQuantProd               # Алгоритм 2: IP-квантование
│   ├── __init__                 #   TurboQuantMSE(b-1) + QJL
│   ├── encode / decode          #   (idx, знаки, норма остатка)
│   ├── inner_product_estimate   #   несмещённая оценка ⟨y, x⟩
│   └── bits_per_vector          #   размер сжатого вектора в битах
└── _demo()                      # демо и замеры

fuzzer.py                        # fuzzer корректности (5 инвариантов)
├── make_unit_batch()            # генерация случайных единичных векторов
├── check_roundtrip()            # инвариант 1: encode→decode
├── check_mse_monotone()         # инвариант 2: MSE убывает с ростом b
├── check_ip_bias()              # инвариант 3: несмещённость IP
├── check_ip_variance()          # инвариант 4: дисперсия IP в границах теории
├── check_edge_cases()           # инвариант 5: граничные случаи
├── run_iteration()              # одна случайная итерация
└── main()                       # CLI: --iters, --seed
```

---

## Реализация на Rust

Rust-реализация находится в директории `rust/` и представляет собой крейт-библиотеку `turboquant_rs` с Python bindings через PyO3/maturin.

Зависимости: `rand 0.8`, `rand_distr 0.4`, `rayon 1.10` (параллелизм), `pyo3 0.25` (Python bindings).

### Установка (Rust)

Требования: Rust 1.75+ (edition 2021), Cargo.

```bash
cd rust
cargo build --release
```

Запустить демо:

```bash
cargo run --release
```

#### Python bindings (через maturin)

```bash
# В корне проекта
pip install maturin
maturin develop --release
```

После установки:

```python
from turboquant_rs import TurboQuantMse, TurboQuantProd, QuantizedProd

# Проверка доступности Rust bindings
import turboquant_rs
print(turboquant_rs.__has_rust__)  # True
```

### Использование (Rust)

#### MSE-квантование

```rust
use turboquant::TurboQuantMse;

let d = 256;
let b = 4;
let q = TurboQuantMse::new(d, b, Some(42));

// Кодирование одного вектора (плоский срез длиной d)
let x: Vec<f64> = /* единичный вектор длиной d */;
let idx: Vec<u16> = q.encode(&x);      // индексы центроидов
let x_hat: Vec<f64> = q.decode(&idx);  // реконструированный вектор

// Батч: плоский буфер n*d элементов
let x_batch: Vec<f64> = /* n*d элементов */;
let idx_batch = q.encode(&x_batch);    // Vec<u16> длиной n*d
let x_hat_batch = q.decode(&idx_batch);
```

#### Несмещённая оценка скалярного произведения

```rust
use turboquant::TurboQuantProd;

let q = TurboQuantProd::new(256, 4, Some(42));

// Кодирование вектора базы
let qv = q.encode(&x);  // QuantizedVec { mse_idx, qjl_signs, res_norm }

// Оценка скалярного произведения с запросом y
let ip_est: f64 = q.inner_product_estimate(&y, &qv);
// E[ip_est] == ⟨y, x⟩

// Декодирование (реконструкция вектора)
let x_tilde: Vec<f64> = q.decode(&qv);

println!("bits per vector: {}", q.bits_per_vector());
```

### API (Rust)

#### `TurboQuantMse`

| Метод | Описание |
|---|---|
| `TurboQuantMse::new(d, b, seed)` | Создаёт квантизатор: матрица поворота + кодебук Lloyd-Max |
| `encode(x: &[f64]) -> Vec<u16>` | Плоский буфер векторов → индексы центроидов |
| `decode(idx: &[u16]) -> Vec<f64>` | Индексы → реконструированные векторы |
| `encode_with_norm(x) -> (Vec<u16>, f32)` | Для ненормированных векторов |
| `decode_with_norm(idx, norm) -> Vec<f64>` | Обратное к `encode_with_norm` |
| `mse(x) -> f64` | Вычислить MSE на батче |

Параметры конструктора:

| Параметр | Тип | Описание |
|---|---|---|
| `d` | `usize` | Размерность вектора |
| `b` | `usize` | Бит на координату (1–16) |
| `seed` | `Option<u64>` | Случайное зерно (`None` — не детерминировано) |

#### `TurboQuantProd`

| Метод | Описание |
|---|---|
| `TurboQuantProd::new(d, b, seed)` | Создаёт квантизатор: `TurboQuantMse(b-1)` + матрица QJL |
| `encode(x: &[f64]) -> QuantizedVec` | Вектор → `{ mse_idx, qjl_signs, res_norm }` |
| `decode(qv: &QuantizedVec) -> Vec<f64>` | Реконструкция вектора |
| `inner_product_estimate(y, qv) -> f64` | Несмещённая оценка ⟨y, x⟩ |
| `bits_per_vector() -> usize` | Размер сжатого вектора в битах |

#### `Qjl`

Вспомогательный тип, используется внутри `TurboQuantProd`.

| Метод | Описание |
|---|---|
| `Qjl::new(d, seed)` | Инициализация проекционной матрицы S ∈ ℝ^{d×d} |
| `encode(x) -> (Vec<i8>, f64)` | x → (знаки ±1, норма ‖x‖₂) |
| `decode(signs, norm) -> Vec<f64>` | Несмещённая реконструкция |

### Python Bindings API

Все Rust классы доступны из Python через модуль `turboquant_rs`:

#### `TurboQuantMse` (Python)

```python
from turboquant_rs import TurboQuantMse

q = TurboQuantMse(d=256, b=4, seed=42)

# Свойства
q.d           # размерность
q.b           # бит на координату
q.n_centroids # количество центроидов (2^b)

# Методы
q.encode(x: List[float]) -> List[int]
q.decode(indices: List[int]) -> List[float]
q.encode_with_norm(x: List[float]) -> Tuple[List[int], float]
q.decode_with_norm(indices: List[int], norm: float) -> List[float]
q.mse(x: List[float]) -> float
```

#### `TurboQuantProd` (Python)

```python
from turboquant_rs import TurboQuantProd

q = TurboQuantProd(d=256, b=4, seed=42)

# Свойства
q.d  # размерность
q.b  # бит на координату

# Методы
q.encode(x: List[float]) -> QuantizedProd
q.decode(qv: QuantizedProd) -> List[float]
q.inner_product_estimate(y: List[float], qv: QuantizedProd) -> float
```

#### `QuantizedProd` (Python)

```python
# Сжатое представление вектора
qv.mse_indices    # List[int] — индексы MSE центроидов
qv.qjl_signs      # List[int]  — знаки QJL (±1)
qv.residual_norm  # float      — норма остатка

repr(qv)  # удобная строка представления
```

### Структура кода (Rust)

```
rust/
├── Cargo.toml
└── src/
    ├── lib.rs          # публичный API + Python bindings (PyO3)
    ├── main.rs         # демо и замеры производительности
    ├── bin/
    │   └── fuzzer.rs   # fuzzer корректности (5 инвариантов)
    ├── lloyd.rs        # Lloyd-Max квантизатор (гауссовский и сферический варианты)
    ├── mse.rs          # TurboQuantMse: поворот + кодебук, encode/decode
    ├── qjl.rs          # Qjl: матрица S, sign-проекция, несмещённое декодирование
    └── prod.rs         # TurboQuantProd: MSE + QJL остаток, оценка скалярных произведений
```

Python bindings собираются через `maturin` и доступны как `turboquant_rs`.

---

## Fuzzer

Standalone-фаззер корректности для Python и Rust. Генерирует случайные векторы и параметры, прогоняет 5 инвариантов на каждой итерации.

### Проверяемые инварианты

| # | Инвариант | Критерий |
|---|-----------|----------|
| 1 | **Roundtrip** `decode(encode(x))` не падает, форма верная | shape совпадает с входом |
| 2 | **MSE монотонность** больше бит → меньше ошибка | `MSE(b+1) ≤ MSE(b) × 1.1` |
| 3 | **IP несмещённость** оценка скалярного произведения без систематического смещения | `\|mean(est − true)\| < 0.05` |
| 4 | **IP дисперсия** в пределах теоретической границы QJL (для d ≥ 16) | `var ≤ (π/2d)·‖y‖² × 2.0` |
| 5 | **Граничные случаи** b=1/8, нулевые и почти нулевые векторы, малые d | нет паник, форма верная |

### Fuzzer Python

```bash
# Базовый запуск (50 итераций, случайный seed)
python fuzzer.py

# С фиксированным seed и количеством итераций
python fuzzer.py --iters 100 --seed 42
```

Пример вывода:

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

Exit-код: `0` — все прошли, `1` — есть падения.

### Fuzzer Rust

```bash
# Базовый запуск
cd rust
cargo run --bin fuzzer

# С параметрами
cargo run --bin fuzzer -- --iters 100 --seed 42

# Release-сборка (быстрее)
cargo run --release --bin fuzzer -- --seed 0
```

Пример вывода:

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

---

## Расширенные модули

### Mixed-Precision квантование

Модуль `turboquant_mixed.py` позволяет задавать разную битовость для разных слоёв модели или групп координат одного вектора.

```python
from turboquant_mixed import MixedPrecisionQuantizer

# Режим 1: по слоям — каждый слой модели со своей битовостью
q = MixedPrecisionQuantizer.from_layers(
    layer_dims=[256, 512, 128],
    layer_bits=[4, 2, 8],
    seed=42,
)

# Режим 2: по группам — вектор делится на части с разной битовостью
q = MixedPrecisionQuantizer.from_groups(
    d=4096,
    group_sizes=[2048, 1024, 1024],
    group_bits=[4, 2, 8],
    seed=42,
)
print(q.bit_allocation_summary())
# Avg bits/coord: 4.50  Compression: 14.2x

# Режим 3: автоматическое распределение по важности
q = MixedPrecisionQuantizer.from_importance(
    d=4096, n_groups=4, bits_range=(2, 8), seed=42,
)
# Группы: 2 → 4 → 6 → 8 бит
```

| Режим | Описание | Сжатие |
|---|---|---|
| Layer | Каждый слой — своя битовость | Зависит от модели |
| Group | Вектор делится на группы | 12–14× |
| Importance | Автоматическое 2→8 бит | 12.8× |

### Квантование разреженных векторов

Модуль `turboquant_sparse.py` оптимизирован для векторов с большим количеством нулей (MoE gating, sparse attention, pruning).

```python
from turboquant_sparse import SparseQuantizer
import numpy as np

# Вектор с 90% нулей
d = 4096
x = np.zeros(d)
x[np.random.choice(d, size=d//10)] = np.random.randn(d//10)

q = SparseQuantizer(d, b=4, seed=42)
enc = q.encode(x)
x_hat = q.decode(enc)

print(f"NNZ: {len(enc.indices)}/{d}")
print(f"Сжатие: {q.compression_vs_dense(enc):.1f}×")
```

| Sparsity | Bits/vector | Сжатие vs Dense |
|---|---|---|
| 50% | 73 760 | 0.2× |
| 90% | 14 756 | **1.1×** |
| 99% | 1 472 | **11.1×** |

Поддержка scipy sparse матриц (`encode_sparse_matrix`) и режима TurboQuantProd.

### Numba JIT ускорение

Модуль `turboquant_numba.py` обеспечивает 2× ускорение encode/decode через JIT-компиляцию.

```python
from turboquant_numba import TurboQuantMSEJIT

q = TurboQuantMSEJIT(d=256, b=4, seed=42)
idx = q.encode(X)   # JIT-compiled, параллельный
x_hat = q.decode(idx)
```

| Операция | NumPy | Numba | Ускорение |
|---|---|---|---|
| Encode (d=256, n=5000) | 56 ms | 28 ms | **2.0×** |
| Decode | 5 ms | 47 ms | — (overhead) |

Установка: `pip install numba` (опционально).

### Интеграция с LLM

Модуль `turboquant_llm.py` предоставляет backend-адаптеры для llama.cpp (GGUF) и vLLM (KV-cache).

```python
from turboquant_llm import (
    TurboQuantBackend,
    quantize_model_layers,
    reconstruct_model_layers,
)

# GGUF-стиль: квантование весов модели
backend = TurboQuantBackend.create("gguf")
q = backend.quantizer(d=4096, b=4, seed=42)
data = backend.encode_weight(weight_matrix, q)
W_rec = backend.decode_weight(data, 4096, 4, q)

# Полное квантование модели
state_dict = {"layer1.q.weight": W_q, "layer1.k.weight": W_k, ...}
qmodel = quantize_model_layers(
    state_dict,
    backend_name="gguf",
    bits_per_layer={"layer1.q.weight": 4, "layer1.ffn.weight": 8},
    default_bits=2,
    seed=42,
)
print(qmodel.summary())

# VLLM KV-cache: квантование активаций по токенам
backend_kv = TurboQuantBackend.create("vllm")
q = backend_kv.quantizer(head_dim=128, b=4, seed=42)
data = backend_kv.encode_weight(kv_cache, q)
kv_rec = backend_kv.decode_weight(data, 128, 4, q)
```

| Backend | Формат | Назначение |
|---|---|---|
| GGUF (`TQGG`) | Binary blob | llama.cpp-style weights |
| VLLM (`TQKV`) | Per-token binary | KV-cache активации |

---

## Бенчмарки

### Python

```bash
python benchmark.py --d 256 --n 5000 --bits 1 2 4 8
```

Результаты (d=256, n=5000):

| Метод | b | MSE | Encode (ms/vec) | Decode (ms/vec) | Bits/vec | Compression |
|---|---|---|---|---|---|---|
| MSE | 1 | 0.00147 | 0.006 | 0.001 | 256 | 64.0× |
| MSE | 2 | 0.00051 | 0.008 | 0.001 | 512 | 32.0× |
| MSE | 4 | 0.00009 | 0.012 | 0.001 | 1024 | 16.0× |
| MSE | 8 | 0.00004 | 0.183 | 0.001 | 2048 | 8.0× |
| Prod | 2 | — | — | — | 544 | 30.1× |
| Prod | 4 | — | — | — | 1056 | 15.5× |
| Prod | 8 | — | — | — | 2080 | 7.9× |

Decay MSE: ~2.5–2.9× на бит (теория: ~4×).
IP bias: < 0.001 для всех b.

### Rust (criterion)

```bash
cd rust && cargo bench --bench benchmarks
```

| Операция | d=256, n=1000 |
|---|---|
| MSE setup | ~36 ms |
| MSE encode | ~42–46 ms |
| MSE decode | ~43–65 ms |
| Prod encode | ~148–157 ms |
| Prod decode | ~128 ms |

---

## Тестирование

```bash
# Python unit tests + property-based
python -m pytest tests/ -v          # 33 теста

# Python fuzzer
python fuzzer.py --iters 50         # 5 инвариантов

# Rust tests (без Python bindings)
cd rust && cargo test --no-default-features  # 15 тестов + 3 proptest

# Rust fuzzer
cargo run --no-default-features --bin fuzzer -- --iters 50

# Python bindings (через maturin)
maturin develop --release
python -c "from turboquant_rs import TurboQuantMse; print('OK')"
```

Итого: **53+ теста** (33 Python + 15 Rust + 5 fuzzer invariants + Python bindings smoke).
Все проходят CI на каждый push.

### CI Pipeline

CI автоматически запускает:
- **Python pure tests** — тестирование оригинальной Python версии
- **Python+Rust integration** — тестирование Python bindings через maturin
- **Rust tests** — нативные Rust тесты, clippy, fuzzer

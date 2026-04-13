"""TurboQuant — Near-Optimal Vector Quantization for LLM Inference.

This package provides both pure Python implementations and Rust bindings
for the TurboQuant vector quantization algorithm.
"""

# Import Rust bindings (the compiled .so/.pyd module)
try:
    # The compiled Rust extension is imported as the C extension
    import turboquant_rs.turboquant_rs as _rust
    TurboQuantMse = _rust.TurboQuantMse
    TurboQuantProd = _rust.TurboQuantProd
    QuantizedProd = _rust.QuantizedProd
    __has_rust__ = True
except (ImportError, AttributeError):
    __has_rust__ = False
    # Fallback to pure Python
    import sys
    import importlib
    _pure = importlib.import_module('turboquant')
    TurboQuantMse = _pure.TurboQuantMSE
    TurboQuantProd = _pure.TurboQuantProd
    QuantizedProd = None

__version__ = "0.1.0"

__all__ = ["TurboQuantMse", "TurboQuantProd", "QuantizedProd", "__has_rust__", "__version__"]


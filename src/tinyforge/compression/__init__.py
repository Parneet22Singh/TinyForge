"""Dependency-free compression experiments."""
from .quantization import quantize, dequantize, QuantizedArray
from .benchmark import benchmark, benchmark_quantization
__all__ = ["quantize", "dequantize", "QuantizedArray", "benchmark", "benchmark_quantization"]

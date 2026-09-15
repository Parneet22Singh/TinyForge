from .quantization import quantize, dequantize
import time

def benchmark(values, *, bits=(8, 16)) -> dict:
    source = list(values)
    if not source: return {}
    result = {}
    for width in bits:
        started = time.perf_counter()
        encoded = quantize(source, width)
        restored = dequantize(encoded)
        error = sum(abs(a-b) for a,b in zip(source, restored)) / len(source)
        compressed_bytes = (len(source) * width + 7) // 8 + 16
        original_bytes = len(source) * 4
        result[width] = {"mean_absolute_error": error,
                         "compression_ratio": original_bytes / compressed_bytes,
                         "original_bytes": original_bytes,
                         "compressed_bytes": compressed_bytes,
                         "latency_ms": round((time.perf_counter() - started) * 1000, 4)}
    return result

benchmark_quantization = benchmark

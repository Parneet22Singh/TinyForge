from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class QuantizedArray:
    values: tuple[int, ...]
    scale: float
    zero_point: int
    bits: int = 8

def quantize(values: Sequence[float], bits: int = 8) -> QuantizedArray:
    if bits not in (4, 8, 16): raise ValueError("bits must be 4, 8 or 16")
    if not values: return QuantizedArray((), 1.0, 0, bits)
    lo, hi = min(values), max(values)
    levels = (1 << bits) - 1
    scale = (hi - lo) / levels if hi != lo else 1.0
    zero = round(-lo / scale) if hi != lo else 0
    return QuantizedArray(tuple(max(0, min(levels, round(v / scale) + zero)) for v in values),
                          scale, zero, bits)

def dequantize(values: QuantizedArray) -> list[float]:
    return [(v - values.zero_point) * values.scale for v in values.values]

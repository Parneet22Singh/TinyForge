"""Deterministic model selection based on hardware constraints."""
from .hardware import HardwareInfo, detect_hardware
from .registry import ModelRegistry, ModelSpec

def select_model(registry: ModelRegistry, hardware: HardwareInfo | None = None,
                 *, task: str | None = None, max_memory_gb: float | None = None) -> ModelSpec:
    hw = hardware or detect_hardware()
    available_memory = hw.vram_gb if hw.gpu and hw.vram_gb > 0 else hw.memory_gb
    limit = max_memory_gb if max_memory_gb is not None else (available_memory if available_memory > 0 else None)
    candidates = [
        m for m in registry.list()
        if (task is None or task == m.task or task in m.supported_tasks)
        and (limit is None or not m.memory_gb or m.memory_gb <= limit)
    ]
    if not candidates:
        raise ValueError("no model satisfies the hardware constraints")
    return min(candidates, key=lambda m: (m.memory_gb or float("inf"), m.parameters, m.name))

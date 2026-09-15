from dataclasses import dataclass, asdict
import math
from .config import TrainingConfig

@dataclass(frozen=True)
class PilotResult:
    recommended_batch_size: int
    estimated_steps: int
    estimated_memory_gb: float
    backend: str = "simulation"
    def to_dict(self): return asdict(self)

def run_pilot(config: TrainingConfig, *, samples: int = 100, memory_gb: float = 1.0) -> PilotResult:
    if samples < 0: raise ValueError("samples must be non-negative")
    batch = max(1, min(config.batch_size, samples or config.batch_size))
    return PilotResult(batch, math.ceil(samples / batch) * config.epochs if samples else 0,
                       round(memory_gb * batch / max(config.batch_size, 1), 6))

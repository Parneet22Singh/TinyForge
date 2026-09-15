from dataclasses import dataclass, asdict
import random
from typing import Sequence
from .config import TrainingConfig

@dataclass(frozen=True)
class TrainResult:
    loss: float
    steps: int
    history: list[float]
    backend: str = "simulation"
    def to_dict(self): return asdict(self)

def train(data: Sequence[object] | None = None, config: TrainingConfig | None = None) -> TrainResult:
    config = config or TrainingConfig()
    n = len(data) if data is not None else 100
    steps = min(config.max_steps, n * config.epochs) if config.max_steps else n * config.epochs
    rng = random.Random(config.seed)
    loss = 1.0
    history = []
    for _ in range(max(0, steps)):
        loss = max(0.0, loss - config.learning_rate * (0.5 + rng.random() * 0.5))
        history.append(round(loss, 8))
    return TrainResult(round(loss, 8), steps, history)

from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class TrainingConfig:
    model: str = "tiny"
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-3
    seed: int = 0
    max_steps: int | None = None
    output_dir: str = "runs"
    sequence_length: int = 1024
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    quantization: str = "none"
    device: str = "auto"

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        if self.sequence_length < 8 or self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("sequence_length, lora_rank and lora_alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.quantization not in {"none", "4bit", "8bit"}:
            raise ValueError("quantization must be none, 4bit or 8bit")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda or cpu")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

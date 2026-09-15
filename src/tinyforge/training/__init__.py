"""Framework-independent training orchestration."""
from .config import TrainingConfig
from .pilot import PilotResult, run_pilot
from .trainer import TrainResult, train
from .evaluator import evaluate
from .lora import dependencies_available, train_lora
__all__ = ["TrainingConfig", "PilotResult", "run_pilot", "TrainResult", "train", "evaluate",
           "dependencies_available", "train_lora"]

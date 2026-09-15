from collections.abc import Sequence
from typing import Any

def evaluate(predictions: Sequence[Any], targets: Sequence[Any]) -> dict[str, float]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if not predictions: return {"accuracy": 0.0, "count": 0.0}
    correct = sum(p == t for p, t in zip(predictions, targets))
    return {"accuracy": correct / len(predictions), "count": float(len(predictions))}


def split_records(records: Sequence[Any], validation_fraction: float = 0.2) -> tuple[list[Any], list[Any]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    cut = max(1, min(len(records) - 1, round(len(records) * (1 - validation_fraction)))) if len(records) > 1 else len(records)
    return list(records[:cut]), list(records[cut:])


def regression_metrics(predictions: Sequence[float], targets: Sequence[float]) -> dict[str, float]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if not predictions:
        return {"mae": 0.0, "mse": 0.0, "count": 0.0}
    errors = [float(p) - float(t) for p, t in zip(predictions, targets)]
    return {"mae": sum(abs(error) for error in errors) / len(errors),
            "mse": sum(error * error for error in errors) / len(errors),
            "count": float(len(errors))}

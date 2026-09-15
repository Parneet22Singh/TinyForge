"""Small model registry that works without an ML framework."""
from dataclasses import dataclass, field
from typing import Any, Iterable

@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameters: int = 0
    context_length: int = 2048
    task: str = "text"
    memory_gb: float = 0.0
    supported_tasks: tuple[str, ...] = ("text",)
    default_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    metadata: dict[str, Any] = field(default_factory=dict)

class ModelRegistry:
    def __init__(self, models: Iterable[ModelSpec] | None = None):
        self._models = {m.name: m for m in (models or ())}

    def register(self, model: ModelSpec) -> ModelSpec:
        self._models[model.name] = model
        return model

    def get(self, name: str) -> ModelSpec:
        try:
            return self._models[name]
        except KeyError:
            raise KeyError(f"unknown model: {name}") from None

    def list(self, task: str | None = None) -> list[ModelSpec]:
        return [m for m in self._models.values() if task is None or m.task == task]

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __len__(self) -> int:
        return len(self._models)


def default_registry() -> ModelRegistry:
    """Return a small registry of common causal-LM families.

    Memory values are conservative estimates for inference/fine-tuning planning,
    not guarantees; actual requirements depend on sequence length and runtime.
    """
    return ModelRegistry([
        ModelSpec("smollm-135m", 135_000_000, 2048, "text", 1.5,
                  metadata={"hf_id": "HuggingFaceTB/SmolLM-135M"}),
        ModelSpec("qwen2.5-0.5b", 500_000_000, 32768, "text", 3.0,
                  metadata={"hf_id": "Qwen/Qwen2.5-0.5B"}),
        ModelSpec("llama3.2-1b", 1_000_000_000, 131072, "text", 5.0,
                  metadata={"hf_id": "meta-llama/Llama-3.2-1B"}),
        ModelSpec("mistral-7b", 7_000_000_000, 32768, "text", 14.0,
                  metadata={"hf_id": "mistralai/Mistral-7B-v0.3"}),
        ModelSpec("gpt2", 124_000_000, 1024, "text", 1.5,
                  default_target_modules=("c_attn", "c_proj"),
                  metadata={"hf_id": "gpt2"}),
    ])

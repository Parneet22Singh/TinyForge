from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .harvest.crawler import Crawler
from .compression.benchmark import benchmark
from .models.hardware import detect_hardware
from .models.registry import ModelRegistry, ModelSpec, default_registry
from .models.selector import select_model
from .optimization.search import grid_search
from .training.config import TrainingConfig
from .training.pilot import run_pilot
from .training.trainer import train
from .training.lora import train_lora


def _inspect(path: Path) -> int:
    stats_path = path / "dataset_stats.json" if path.is_dir() else path.with_name("dataset_stats.json")
    if not stats_path.exists():
        raise FileNotFoundError(f"missing dataset statistics: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    print("TinyForge dataset")
    print(f"  examples: {stats['examples']}")
    print(f"  approximate tokens: {stats['approximate_tokens']}")
    print(f"  average document length: {stats['average_document_length']}")
    print(f"  duplicate count: {stats['duplicate_count']}")
    print(f"  average quality: {stats['quality']['average']:.3f}")
    print(f"  languages: {stats['language_distribution']}")
    return 0


def _registry() -> ModelRegistry:
    return default_registry()


def _records(path: Path) -> list[dict[str, object]]:
    dataset = path / "dataset.jsonl" if path.is_dir() else path
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not found: {dataset}")
    return [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_run(output: Path, config: TrainingConfig, result: object, metrics: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "training.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    model_dir = output / "model"
    model_dir.mkdir(exist_ok=True)
    weights = list(getattr(result, "history", [])) or [0.0]
    (model_dir / "weights.json").write_text(json.dumps(weights) + "\n", encoding="utf-8")


def _hardware() -> int:
    info = detect_hardware()
    print("TinyForge hardware")
    print(f"  CPU cores: {info.cpu_count}")
    print(f"  RAM: {info.memory_gb:.2f} GB" if info.memory_gb else "  RAM: unavailable")
    print(f"  GPU: {info.gpu_name or 'none detected'}")
    print(f"  VRAM: {info.vram_gb:.2f} GB" if info.vram_gb else "  VRAM: unavailable")
    if info.physical_gpu and not info.gpu:
        print("  GPU runtime: detected, but PyTorch is CPU-only")
    print(f"  backend: {info.backend}")
    if info.cuda_version:
        print(f"  CUDA: {info.cuda_version}")
    elif info.driver_cuda_version:
        print(f"  NVIDIA driver CUDA: {info.driver_cuda_version}")
    if info.torch_version:
        print(f"  PyTorch: {info.torch_version}")
    print(f"  preferred device: {info.preferred_device}")
    print(f"  recommended mode: {'QLoRA / 4-bit' if info.gpu else 'CPU simulation'}")
    return 0


def _train(dataset: Path, output: Path, model_name: str | None, max_memory: float | None,
            real: bool = False, model_id: str | None = None,
            quantization: str = "none", device: str = "auto") -> int:
    records = _records(dataset)
    hardware = detect_hardware()
    if device == "cuda" and not hardware.gpu:
        raise RuntimeError("CUDA was requested but no CUDA-capable GPU is available")
    registry = _registry()
    model = registry.get(model_name) if model_name else select_model(registry, hardware, max_memory_gb=max_memory)
    if max_memory is not None and model.memory_gb > max_memory:
        raise ValueError(f"model {model.name} requires approximately {model.memory_gb:.1f} GB")
    config = TrainingConfig(model=model.name, max_steps=max(1, min(100, len(records) or 1)),
                            quantization=quantization, device=device)
    pilot = run_pilot(config, samples=len(records), memory_gb=model.memory_gb)
    started = time.perf_counter()
    if real:
        if not model_id:
            raise ValueError("--model-id is required with --real")
        result = train_lora(records, config, model_id, output, model.default_target_modules)
        elapsed = round(time.perf_counter() - started, 6)
        metrics = {
            "model": model_id, "training_time_seconds": elapsed,
            "training_loss": result["training_loss"], "backend": result["backend"],
            "pilot": pilot.to_dict(), "model_size_mb": None,
        }
        _write_run(output, config, type("Result", (), {"to_dict": lambda self: result, "history": []})(), metrics)
        print(f"Real LoRA training complete: {output}")
        return 0
    result = train(records, config)
    elapsed = round(time.perf_counter() - started, 6)
    metrics = {"model": model.name, "training_time_seconds": elapsed, "training_loss": result.loss,
               "validation_loss": round(result.loss * 1.05, 8), "pilot": pilot.to_dict(),
               "model_size_mb": round(model.parameters * 4 / 1_000_000, 2), "backend": result.backend}
    _write_run(output, config, result, metrics)
    print(f"Selected model: {model.name} ({model.memory_gb:.1f} GB estimated memory)")
    print(f"Pilot: {pilot.estimated_steps} steps; training loss: {result.loss:.6f}")
    print(f"Run written to {output}")
    return 0


def _shrink(model: Path, output: Path, bits: int) -> int:
    values_path = model / "weights.json" if model.is_dir() else model
    values = json.loads(values_path.read_text(encoding="utf-8"))
    if isinstance(values, dict):
        values = values.get("values", [])
    if not isinstance(values, list) or not values:
        raise ValueError("model must contain a non-empty JSON array of numeric weights")
    result = benchmark([float(value) for value in values], bits=(16, 8, 4) if bits == 0 else (bits,))
    output.mkdir(parents=True, exist_ok=True)
    (output / "compression.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def _benchmark(model: Path) -> int:
    result = model / "compression.json" if model.is_dir() else model
    if not result.exists():
        raise FileNotFoundError(f"compression results not found: {result}")
    print(result.read_text(encoding="utf-8"), end="")
    return 0


def _optimize(dataset: Path, max_memory: float) -> int:
    records = _records(dataset)
    space = {"model": [spec.name for spec in _registry().list()
                       if spec.memory_gb <= max_memory],
             "learning_rate": [1e-3, 2e-3], "epochs": [1, 2]}
    def objective(params: dict[str, object]) -> float:
        size = _registry().get(str(params["model"])).memory_gb
        return (len(records) + 1) / (size * 100) - float(params["learning_rate"]) * 10 + int(params["epochs"]) * 0.01
    if not space["model"]:
        raise ValueError(f"no registered model fits within {max_memory:.2f} GB")
    result = grid_search(space, objective)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyforge", description="Local-first web-to-dataset ML experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    harvest = subparsers.add_parser("harvest", help="crawl a permitted public web source")
    harvest.add_argument("url")
    harvest.add_argument("--output", "-o", type=Path, default=Path("dataset"))
    harvest.add_argument("--max-pages", type=int, default=20)
    harvest.add_argument("--delay", type=float, default=1.0)
    subparsers.add_parser("hardware", help="inspect local compute hardware")
    subparsers.add_parser("models", help="list registered model families")
    inspect = subparsers.add_parser("dataset", help="inspect a harvested dataset")
    dataset_commands = inspect.add_subparsers(dest="dataset_command", required=True)
    inspect_command = dataset_commands.add_parser("inspect")
    inspect_command.add_argument("path", type=Path)
    train_command = subparsers.add_parser("train", help="run a reproducible local training simulation")
    train_command.add_argument("dataset", type=Path)
    train_command.add_argument("--output", type=Path, default=None)
    train_command.add_argument("--model")
    train_command.add_argument("--max-memory", type=float)
    train_command.add_argument("--real", action="store_true", help="run real LoRA training")
    train_command.add_argument("--model-id", help="Hugging Face model ID/path for --real")
    train_command.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none")
    train_command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    shrink = subparsers.add_parser("shrink", help="benchmark quantized model representations")
    shrink.add_argument("model", type=Path)
    shrink.add_argument("--output", type=Path, default=Path("compressed"))
    shrink.add_argument("--bits", type=int, choices=(0, 4, 8, 16), default=0)
    benchmark_command = subparsers.add_parser("benchmark", help="benchmark stored compression results")
    benchmark_command.add_argument("model", type=Path)
    optimize = subparsers.add_parser("optimize", help="search a small constrained configuration space")
    optimize.add_argument("--dataset", type=Path, required=True)
    optimize.add_argument("--max-memory", type=float, default=4.0)
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "harvest":
        report = Crawler(args.output, max_pages=args.max_pages, delay=args.delay).run(args.url)
        print(f"Harvested {report['pages_extracted']} pages into {args.output}")
        print(f"Dataset examples: {report['dataset']['examples']}")
        print(f"Average quality: {report['dataset']['quality']['average']:.3f}")
        return 0
    if args.command == "dataset" and args.dataset_command == "inspect":
        return _inspect(args.path)
    if args.command == "hardware":
        return _hardware()
    if args.command == "models":
        for spec in _registry().list():
            print(f"{spec.name}\t{spec.metadata.get('hf_id', '-')}\t{spec.memory_gb:.1f} GB\t{spec.context_length} tokens")
        return 0
    if args.command == "train":
        output = args.output or Path("runs") / time.strftime("%Y%m%d_%H%M%S")
        return _train(args.dataset, output, args.model, args.max_memory, args.real, args.model_id, args.quantization, args.device)
    if args.command == "shrink":
        return _shrink(args.model, args.output, args.bits)
    if args.command == "benchmark":
        return _benchmark(args.model)
    if args.command == "optimize":
        return _optimize(args.dataset, args.max_memory)
    return 1


def main() -> int:
    args = build_parser().parse_args()
    try:
        return _dispatch(args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        print(f"tinyforge: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

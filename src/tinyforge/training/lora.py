"""Optional real LoRA/QLoRA integration.

The dependency-free simulator remains the default. This adapter only runs when
the user explicitly requests it and the Hugging Face stack is installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .config import TrainingConfig


def dependencies_available() -> bool:
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def train_lora(
    records: Sequence[dict[str, Any]],
    config: TrainingConfig,
    model_name_or_path: str,
    output_dir: Path,
    target_modules: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not dependencies_available():
        raise RuntimeError(
            "real LoRA training requires datasets, transformers, and peft; "
            "install the tinyforge[ml] extra or omit --real"
        )
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
        Trainer, TrainingArguments,
    )
    try:
        import torch
        from peft import prepare_model_for_kbit_training
    except ImportError:
        torch = None
        prepare_model_for_kbit_training = None

    texts = [str(record.get("text", "")) for record in records if str(record.get("text", "")).strip()]
    if not texts:
        raise ValueError("dataset contains no non-empty text examples")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = {}
    use_cuda = config.device == "cuda" or (config.device == "auto" and torch is not None and torch.cuda.is_available())
    if config.device == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but no CUDA-capable PyTorch device is available")
    if config.quantization == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            if torch is None:
                raise RuntimeError("PyTorch is required for 4-bit QLoRA")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            if not use_cuda:
                raise RuntimeError("4-bit QLoRA requires CUDA; use --quantization none on CPU")
            load_kwargs["device_map"] = "auto"
        except ImportError as error:
            raise RuntimeError("4-bit QLoRA requires bitsandbytes and transformers") from error
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    if use_cuda and "device_map" not in load_kwargs:
        model = model.to("cuda")
    if config.quantization == "4bit" and prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    target_modules = list(target_modules) if target_modules else (["q_proj", "v_proj"] if {"q_proj", "v_proj"} <= module_names else (
        ["c_attn", "c_proj"] if {"c_attn", "c_proj"} <= module_names else None
    ))
    if not target_modules:
        raise RuntimeError("could not infer LoRA target modules for this model; configure the model architecture explicitly")
    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=target_modules, task_type=TaskType.CAUSAL_LM,
    ))
    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(
        test_size=min(0.2, max(1 / len(dataset), 0.01)), seed=config.seed
    ) if len(dataset) > 1 else {"train": dataset, "test": dataset}
    tokenized = split["train"].map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=config.sequence_length),
        batched=True,
        remove_columns=["text"],
    )
    eval_tokenized = split["test"].map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=config.sequence_length),
        batched=True,
        remove_columns=["text"],
    )
    training_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir), num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size, learning_rate=config.learning_rate,
        gradient_accumulation_steps=1, logging_strategy="steps", logging_steps=1,
        save_strategy="epoch", report_to=[],
        fp16=use_cuda, use_cpu=not use_cuda,
    )
    import inspect
    argument_names = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in argument_names:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    if config.max_steps is not None:
        training_kwargs["max_steps"] = config.max_steps
    args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized, eval_dataset=eval_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    result = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    return {"backend": "lora", "training_loss": float(result.training_loss), "steps": int(result.global_step)}

# TinyForge

## What is TinyForge?

TinyForge is a local-first ML systems project for turning permitted public web documentation
into small, specialized models that can run within a user's hardware constraints.

The central idea is simple:

> Give TinyForge useful domain data and a hardware limit, then measure how small a useful
> specialized model can become.

Large general-purpose models are powerful, but they are often expensive to download, fine-tune,
store, and run. A focused documentation corpus may contain enough signal to build a smaller
model for a narrow task—especially when the experiment is designed around the available VRAM,
RAM, model size, latency, and training budget.

TinyForge is the experimental instrument for studying that trade-off. It connects:

```text
permitted web source
        ↓
harvest and content extraction
        ↓
clean, deduplicate, and score the dataset
        ↓
select a model that fits the hardware
        ↓
pilot and fine-tune with LoRA/QLoRA
        ↓
evaluate quality and resource use
        ↓
quantize, benchmark, and compare
        ↓
small specialized local model
```

The project is built around a practical research question:

> How much useful specialization can we squeeze out of a model when data, compute, memory,
> and model size are all constrained?

TinyForge emphasizes:

- Local execution instead of a hosted service
- Reproducible experiments instead of unsupported claims
- Dataset quality instead of blindly scraping everything
- Parameter-efficient fine-tuning instead of training from scratch
- Hardware-aware model and configuration selection
- Measured trade-offs between quality, size, memory, latency, and training cost

TinyForge is not:

- A generic chatbot
- A generic web scraper
- A cloud training platform
- A dashboard or SaaS product
- An autonomous browsing agent
- A general AutoML framework
- A CAPTCHA solver or anti-bot bypass tool

It is a CLI-first experimental pipeline for exploring efficient, specialized models on real
local hardware.

## Pipeline overview

```text
public URL -> crawl -> extract -> clean -> dataset -> select -> train -> shrink -> benchmark
```

## Requirements

### Base workflow

- Python 3.10 or newer
- Internet access for harvesting public pages
- Internet access only when downloading remote models or Python packages

The base workflow uses the Python standard library plus `psutil` for hardware inspection.
It works without PyTorch, Transformers, a GPU, or model downloads.

### Real LoRA/QLoRA workflow

- PyTorch
- Transformers
- Datasets
- PEFT
- Accelerate
- bitsandbytes for 4-bit QLoRA
- A compatible CPU or CUDA-capable NVIDIA GPU
- Enough RAM, VRAM, disk space, and time for the selected model

The repository does not contain model weights. Real training uses either a local checkpoint
or a model downloaded from the Hugging Face Hub.

## Installation

Clone the repository and enter its directory:

```powershell
git clone <repository-url>
cd tinyforge
```

### Recommended: create an isolated environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools and install the base project:

```powershell
python -m pip install --upgrade pip
python -m pip install .
```

For development and tests:

```powershell
python -m pip install ".[test]"
```

For real LoRA training:

```powershell
python -m pip install ".[ml]"
```

The ML extra installs PyTorch, Transformers, Datasets, PEFT, Accelerate, and
bitsandbytes. GPU-enabled PyTorch wheels are platform-specific; follow the official
PyTorch installation selector for the correct CUDA build before installing the TinyForge
ML extra if your default package index would install CPU-only PyTorch.

After installation, verify the active environment:

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
tinyforge hardware
```

If `nvidia-smi` reports a GPU but `torch.cuda.is_available()` is `False`, the NVIDIA driver
may be installed while the active Python environment still has CPU-only PyTorch. Install a
CUDA-enabled PyTorch build into the same environment used to run TinyForge.

## Device selection

TinyForge uses GPU-first behavior:

```text
CUDA-enabled PyTorch available -> CUDA
otherwise                      -> CPU
```

Use automatic selection by default:

```powershell
tinyforge train dataset --device auto
```

Force a device when needed:

```powershell
tinyforge train dataset --device cuda
tinyforge train dataset --device cpu
```

Forcing CUDA fails clearly when no usable CUDA runtime is available. A physical GPU alone
is not sufficient; PyTorch must be built with CUDA support and be able to initialize it.

## Complete workflow

### 1. Inspect hardware and models

```powershell
tinyforge hardware
tinyforge models
```

`hardware` reports CPU, RAM, GPU, VRAM, CUDA, PyTorch, and the preferred device.
`models` lists the built-in model families, approximate memory requirements, context lengths,
Hugging Face IDs, and architecture-aware LoRA metadata.

### 2. Harvest a permitted public source

```powershell
tinyforge harvest https://example.com/docs --output dataset
```

Useful options:

```powershell
tinyforge harvest https://example.com/docs `
  --output dataset `
  --max-pages 50 `
  --delay 1.5
```

The harvester:

- Follows same-origin links
- Checks `robots.txt` when available
- Uses conservative request delays
- Stores raw HTML and cleaned text
- Extracts titles, content, source metadata, and links
- Removes common scripts, navigation, menus, forms, and footer content
- Detects duplicate content
- Assigns deterministic quality scores

It does not bypass authentication, solve CAPTCHAs, spoof browser fingerprints, or defeat
access controls. Only harvest sources you are permitted to use.

Generated dataset layout:

```text
dataset/
├── raw/
├── cleaned/
├── pages.jsonl
├── dataset.jsonl
├── dataset_stats.json
└── report.json
```

### 3. Inspect the dataset

```powershell
tinyforge dataset inspect dataset
```

The inspection reports example count, approximate tokens, document lengths, duplicate count,
language distribution, and quality distribution.

### 4. Run the safe local training simulation

```powershell
tinyforge train dataset `
  --device auto `
  --output runs/latest
```

This path is deterministic and safe for testing the complete orchestration flow without
downloading a model or consuming GPU resources. It creates:

```text
runs/latest/
├── config.json
├── metrics.json
├── training.json
└── model/
    └── weights.json
```

### 5. Run real LoRA or QLoRA training

Use a local model checkpoint:

```powershell
tinyforge train dataset `
  --real `
  --model-id C:\models\your-causal-model `
  --device auto `
  --quantization none `
  --output runs/lora
```

Use a Hugging Face model ID:

```powershell
tinyforge train dataset `
  --real `
  --model-id <hugging-face-causal-model-id> `
  --device auto `
  --quantization 4bit `
  --output runs/qlora
```

The real path supports:

- Train/evaluation split
- LoRA adapter training
- Architecture-aware target modules
- Optional 4-bit NF4 QLoRA
- Double quantization
- Gradient checkpointing
- CUDA-first execution
- CPU LoRA fallback without 4-bit quantization
- Epoch evaluation
- Adapter and checkpoint export

4-bit QLoRA requires a usable CUDA runtime and bitsandbytes. On CPU-only systems use
`--quantization none`.

### 6. Shrink and benchmark a model artifact

For the lightweight numeric-weight benchmark:

```powershell
tinyforge shrink runs/latest/model --output compressed
tinyforge benchmark compressed
```

You can select one representation:

```powershell
tinyforge shrink runs/latest/model --bits 4 --output compressed-int4
```

The benchmark records:

- Original bytes
- Estimated compressed bytes
- Compression ratio
- Mean absolute error
- Quantization latency

### 7. Search constrained configurations

```powershell
tinyforge optimize `
  --dataset dataset `
  --max-memory 4
```

The search filters models that exceed the memory limit and evaluates a small reproducible
configuration space. Memory values are estimates, not guarantees; actual usage depends on
sequence length, batch size, runtime, quantization, and model architecture.

## Model families

The built-in registry includes representative presets for:

- SmolLM
- Qwen
- Llama
- Mistral
- GPT-2

List them with:

```powershell
tinyforge models
```

Presets provide planning metadata only. The actual checkpoint is loaded from `--model-id`.
Some models may require access approval, authentication, a license acceptance, or additional
runtime configuration on the model host.

## Testing

Run the complete test suite:

```powershell
python -m pytest -q
```

Run with coverage:

```powershell
python -m pytest --cov=tinyforge --cov-report=term-missing
```

The automated tests cover:

- CLI parsing
- Dataset extraction and cleaning
- Quality scoring
- Statistics
- Model registry and memory selection
- GPU/CPU preference logic
- Deterministic simulation training
- Pilot planning
- Compression
- Optimization
- Artifact generation

Real model downloads and long GPU training runs are not part of the default automated suite.
For a real-runtime smoke test, use a small local checkpoint and a tiny dataset, then verify
that `adapter/`, `metrics.json`, and `training.json` are created.

## Troubleshooting

### `tinyforge` command not found

Run it through the active interpreter:

```powershell
python -m tinyforge.cli --help
```

Or reinstall the package:

```powershell
python -m pip install .
```

### GPU appears in `nvidia-smi` but TinyForge selects CPU

Check the active Python environment:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.version.cuda` is `None`, install a CUDA-enabled PyTorch build into that same
environment. NVIDIA drivers and the CUDA toolkit alone do not make a CPU-only PyTorch wheel
GPU-capable.

### No model fits the memory limit

Try:

- A smaller registered model
- `--quantization 4bit` on CUDA
- A shorter sequence length
- A smaller batch size
- CPU execution
- A larger `--max-memory` value if the hardware permits it

### Real training fails while loading a model

Check:

- Network access or local checkpoint path
- Model ID spelling
- Model license/access requirements
- Available disk space
- Transformers and PyTorch compatibility
- CUDA and bitsandbytes compatibility for 4-bit training

## Safety and scope

TinyForge is intended for permitted public sources and local experimentation. It does not
implement authentication bypasses, CAPTCHA solving, anti-bot evasion, stealth fingerprinting,
distributed training, cloud deployment, accounts, billing, or a web dashboard.

## Project structure

```text
tinyforge/
├── src/tinyforge/cli.py
├── src/tinyforge/harvest/
├── src/tinyforge/dataset/
├── src/tinyforge/models/
├── src/tinyforge/training/
├── src/tinyforge/compression/
├── src/tinyforge/optimization/
├── tests/
├── pyproject.toml
├── pytest.ini
├── LICENSE
└── .gitignore
```

TinyForge is an experimental instrument for measuring the trade-offs between specialization,
quality, memory, model size, latency, and training cost.

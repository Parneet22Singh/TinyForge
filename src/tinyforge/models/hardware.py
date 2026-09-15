"""Portable hardware discovery (with a deterministic, dependency-free fallback)."""
from dataclasses import dataclass
import os
import platform
import ctypes
import sys
import subprocess

@dataclass(frozen=True)
class HardwareInfo:
    cpu_count: int
    memory_gb: float
    gpu: bool = False
    gpu_name: str | None = None
    backend: str = "cpu"
    vram_gb: float = 0.0
    cuda_version: str | None = None
    torch_version: str | None = None
    physical_gpu: bool = False
    driver_cuda_version: str | None = None

    @property
    def preferred_device(self) -> str:
        return "cuda" if self.gpu else "cpu"


def _system_memory_gb() -> float:
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong),
                        ("memory_load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong),
                        ("page_available", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_available", ctypes.c_ulonglong),
                        ("extended", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total / 2**30
    if hasattr(os, "sysconf"):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except (ImportError, AttributeError, OSError):
        pass
    return 0.0

def detect_hardware() -> HardwareInfo:
    count = os.cpu_count() or 1
    try:
        memory = _system_memory_gb()
    except (AttributeError, OSError, ValueError):
        memory = 0.0
    # Optional torch is intentionally only probed, never required.
    try:
        import torch  # type: ignore
        available = bool(torch.cuda.is_available())
        vram = 0.0
        if available:
            vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        physical_gpu, gpu_name, physical_vram, driver_cuda = _nvidia_smi()
        if not available and physical_gpu:
            return HardwareInfo(
                count, memory, False, gpu_name, "cpu", physical_vram,
                None, getattr(torch, "__version__", None), True, driver_cuda,
            )
        return HardwareInfo(
            count, memory, available,
            torch.cuda.get_device_name(0) if available else None,
            "cuda" if available else "cpu",
            vram, getattr(torch.version, "cuda", None), getattr(torch, "__version__", None),
            available, None,
        )
    except (ImportError, RuntimeError, AttributeError):
        pass
    physical_gpu, gpu_name, vram, driver_cuda = _nvidia_smi()
    return HardwareInfo(
        count, memory, False, gpu_name, "cpu", vram, None, None,
        physical_gpu, driver_cuda,
    )


def _nvidia_smi() -> tuple[bool, str | None, float, str | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True, timeout=5,
        )
        first = completed.stdout.strip().splitlines()[0]
        name, memory = [part.strip() for part in first.split(",", 1)]
        driver = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
        marker = "CUDA Version:"
        driver_cuda = driver.split(marker, 1)[1].split()[0] if marker in driver else None
        return True, name, float(memory) / 1024, driver_cuda
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError, OSError):
        return False, None, 0.0, None

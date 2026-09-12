"""GPU memory, wall-clock timing and parameter-count instrumentation.

Used identically by the LoRA and QLoRA runs so the numbers in
``outputs/<method>/resource_metrics.json`` are directly comparable:
the same definition of "peak memory", the same timer, the same counting rule.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

BYTES_PER_GIB = 1024**3
BYTES_PER_MIB = 1024**2


# --------------------------------------------------------------------------
# GPU probing
# --------------------------------------------------------------------------
def _torch():
    import torch

    return torch


def cuda_available() -> bool:
    try:
        return _torch().cuda.is_available()
    except Exception:
        return False


def gpu_info() -> dict:
    """Static description of the accelerator, or why measurement is impossible."""
    if not cuda_available():
        return {
            "available": False,
            "unavailable_reason": "no CUDA device visible to torch; "
            "GPU memory metrics cannot be measured on this host",
            "name": None,
            "total_vram_bytes": None,
            "count": 0,
        }
    torch = _torch()
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "unavailable_reason": None,
        "name": props.name,
        "index": index,
        "count": torch.cuda.device_count(),
        "total_vram_bytes": int(props.total_memory),
        "total_vram_gib": round(props.total_memory / BYTES_PER_GIB, 3),
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def reset_peak_memory() -> None:
    """Zero CUDA peak-memory counters. Call immediately before a measured phase."""
    if cuda_available():
        torch = _torch()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory() -> dict:
    """Peak CUDA memory since the last :func:`reset_peak_memory`."""
    if not cuda_available():
        return {
            "peak_memory_allocated_bytes": None,
            "peak_memory_reserved_bytes": None,
            "peak_memory_allocated_gib": None,
            "peak_memory_reserved_gib": None,
            "measured": False,
        }
    torch = _torch()
    torch.cuda.synchronize()
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    return {
        "peak_memory_allocated_bytes": allocated,
        "peak_memory_reserved_bytes": reserved,
        "peak_memory_allocated_gib": round(allocated / BYTES_PER_GIB, 4),
        "peak_memory_reserved_gib": round(reserved / BYTES_PER_GIB, 4),
        "measured": True,
    }


def synchronize() -> None:
    if cuda_available():
        _torch().cuda.synchronize()


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    seconds = float(seconds)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.2f}s"


class Stopwatch:
    """Monotonic wall-clock timer that also records human-readable timestamps.

    CUDA work is synchronized on entry and exit so the measured interval covers
    kernels that were still queued when the Python call returned.
    """

    def __init__(self, label: str = "phase"):
        self.label = label
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.duration_seconds: float | None = None
        self._start: float | None = None

    def __enter__(self) -> "Stopwatch":
        synchronize()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        synchronize()
        self.duration_seconds = time.perf_counter() - self._start
        self.ended_at = datetime.now(timezone.utc).isoformat()
        return False

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration_seconds, 4) if self.duration_seconds else None,
            "duration_human": format_duration(self.duration_seconds) if self.duration_seconds else None,
        }


# --------------------------------------------------------------------------
# Parameter counting
# --------------------------------------------------------------------------
def _logical_numel(param) -> int:
    """Number of *logical* parameters in a tensor, undoing 4-bit packing.

    bitsandbytes stores two NF4 values per uint8, so ``param.numel()`` on a
    quantized weight reports half the real parameter count. The original shape
    survives on ``param.quant_state``; using it keeps QLoRA's reported parameter
    COUNT equal to LoRA's, which is correct -- quantization changes the memory
    FOOTPRINT of those parameters, not how many of them there are.
    """
    quant_state = getattr(param, "quant_state", None)
    shape = getattr(quant_state, "shape", None) if quant_state is not None else None
    if shape is not None:
        count = 1
        for dim in shape:
            count *= int(dim)
        return count
    return int(param.numel())


def _is_quantized(param) -> bool:
    return type(param).__name__ in {"Params4bit", "Int8Params"}


def count_parameters(model) -> dict:
    """Total / trainable parameter counts plus an explicit memory footprint.

    Parameter COUNT and parameter MEMORY FOOTPRINT are reported separately and
    never conflated: a 4-bit base model has the same number of parameters as its
    16-bit counterpart but roughly a quarter of the storage.
    """
    total = trainable = 0
    total_storage_elements = 0
    quantized_params = 0
    parameter_bytes = 0
    quantized_modules = 0

    for param in model.parameters():
        logical = _logical_numel(param)
        total += logical
        total_storage_elements += int(param.numel())
        parameter_bytes += int(param.numel()) * int(param.element_size())
        if _is_quantized(param):
            quantized_params += logical
            quantized_modules += 1
        if param.requires_grad:
            trainable += logical

    buffer_bytes = sum(int(b.numel()) * int(b.element_size()) for b in model.buffers())

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_ratio": (trainable / total) if total else 0.0,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
        # Storage-level view, kept distinct from the logical counts above.
        "total_storage_elements": total_storage_elements,
        "quantized_parameters": quantized_params,
        "quantized_parameter_tensors": quantized_modules,
        "parameter_memory_bytes": parameter_bytes,
        "parameter_memory_gib": round(parameter_bytes / BYTES_PER_GIB, 4),
        "buffer_memory_bytes": buffer_bytes,
        "note": (
            "'total'/'trainable' are logical parameter counts; 4-bit packed weights "
            "are unpacked via quant_state.shape so they are not undercounted. "
            "'parameter_memory_bytes' is the actual storage footprint."
        ),
    }


# --------------------------------------------------------------------------
# Artifact assembly
# --------------------------------------------------------------------------
def count_adapter_parameters(model) -> int:
    """Logical parameter count of the LoRA adapter tensors in a model.

    Evaluation loads adapters frozen, so ``requires_grad`` is False everywhere
    and a plain trainable-parameter count would report 0 for a fine-tuned model.
    Counting the adapter tensors on the instantiated model recovers the number
    that *was* trained, without trusting the intended configuration.
    """
    return sum(_logical_numel(p) for name, p in model.named_parameters() if "lora_" in name)


def build_resource_metrics(
    method: str,
    gpu: dict,
    training_peak: dict,
    training_timer: Stopwatch,
    parameters: dict,
    steps: int | None,
    samples: int | None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Assemble the machine-readable resource artifact.

    ``steps_per_second`` / ``samples_per_second`` are derived, optional detail;
    ``duration_seconds`` remains the primary training-time metric.
    """
    duration = training_timer.duration_seconds
    metrics = {
        "method": method,
        "schema_version": 1,
        "gpu": {
            "name": gpu.get("name"),
            "available": gpu.get("available", False),
            "unavailable_reason": gpu.get("unavailable_reason"),
            "total_vram_bytes": gpu.get("total_vram_bytes"),
            "total_vram_gib": gpu.get("total_vram_gib"),
            "compute_capability": gpu.get("compute_capability"),
            "count": gpu.get("count", 0),
            "torch_version": gpu.get("torch_version"),
            "cuda_version": gpu.get("cuda_version"),
            **training_peak,
        },
        "training": {
            **training_timer.as_dict(),
            "steps": steps,
            "samples": samples,
            "steps_per_second": (steps / duration) if (steps and duration) else None,
            "samples_per_second": (samples / duration) if (samples and duration) else None,
        },
        "parameters": parameters,
    }
    if extra:
        metrics.update(extra)
    return metrics


def write_json(path: str, payload: Any) -> str:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

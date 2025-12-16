import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil


def ensure_directories(*dirs: str) -> None:
    """Ensure directories exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_gpu_memory_info() -> dict:
    """Get GPU memory information if available."""
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)

            return {
                "available": True,
                "device": torch.cuda.get_device_name(device),
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "free_gb": (total - reserved) / (1024**3),
            }
    except ImportError:
        pass

    return {"available": False}


def get_system_memory_info() -> dict:
    """Get system memory information."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent_used": mem.percent,
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def generate_experiment_id(prefix: str = "exp") -> str:
    """Generate a unique experiment ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{unique}"


def calculate_hash(data: str) -> str:
    """Calculate SHA256 hash of data."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def estimate_training_time(
    num_samples: int, batch_size: int, epochs: int, estimated_batch_time_ms: float = 100
) -> float:
    """Estimate training time in seconds."""
    batches_per_epoch = (num_samples + batch_size - 1) // batch_size
    total_batches = batches_per_epoch * epochs
    return (total_batches * estimated_batch_time_ms) / 1000

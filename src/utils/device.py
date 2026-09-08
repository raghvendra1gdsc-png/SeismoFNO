"""
device.py — Hardware Accelerator & Device Synchronization Utilities.

Provides a unified, thread-safe device manager and global accelerator lock.
On Apple Silicon (MPS), PyTorch Metal command queue submissions are not thread-safe
across concurrent threads in a multi-threaded web server (such as FastAPI/Uvicorn
thread pool workers). Attempting concurrent MPS command buffer encoding triggers:
    -[IOGPUMetalCommandBuffer validate]:213: failed assertion 'commit an already committed command buffer'

The GLOBAL_DEVICE_LOCK serializes all device tensor transfers, kernel launches,
and command buffer synchronizations across all serving adapters.
"""

import os
import threading
from typing import Optional
import torch

# Global reentrant lock to serialize all operations on accelerator devices (MPS/CUDA)
# Reentrant (RLock) so nested helper calls within the same thread do not deadlock.
GLOBAL_DEVICE_LOCK = threading.RLock()


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve compute device respecting SEISMOFNO_DEVICE or DEVICE env vars."""
    target = preferred or os.getenv("SEISMOFNO_DEVICE", os.getenv("DEVICE"))
    if target:
        try:
            return torch.device(target)
        except Exception:
            pass
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device: Optional[torch.device] = None) -> None:
    """Synchronize accelerator device execution queues for precise wall-clock timing."""
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            return

    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

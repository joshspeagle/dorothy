"""Small shared utilities."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a device specification to a torch.device.

    Args:
        device: "auto" (CUDA if available, else Apple MPS, else CPU), a device
            string such as "cpu" / "cuda" / "mps", or a torch.device.

    Returns:
        The resolved torch.device.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

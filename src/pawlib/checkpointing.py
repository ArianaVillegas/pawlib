from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch

StateDict = Dict[str, torch.Tensor]
ModelFactory = Callable[[], torch.nn.Module]


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """Persist a model's state dict and optional metadata to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, target)
    return target


def load_checkpoint(
    checkpoint_path: str | Path,
    model_factory: ModelFactory,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a checkpoint and instantiate a model using ``model_factory``."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint} not found.")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = model_factory()
    state_dict: StateDict = payload["state_dict"]
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()  # Set to eval mode by default
    metadata = payload.get("metadata", {})
    return model, metadata

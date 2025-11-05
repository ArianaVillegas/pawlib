"""Reusable PAW seismic waveform components.

This package is intentionally lightweight. Only import modules that do not
bring along the entire research stack so the library stays portable.
"""

from . import losses, architectures, metrics
from .checkpointing import load_checkpoint, save_checkpoint
from .paw import PAW
from .utils import print_metrics, print_subset_metrics
from .config import (
    PAWReferenceConfig,
    load_reference_config,
    default_reference_config,
)

__all__ = [
    "PAW",
    "losses",
    "architectures",
    "metrics",
    "load_checkpoint",
    "save_checkpoint",
    "print_metrics",
    "print_subset_metrics",
    "PAWReferenceConfig",
    "load_reference_config",
    "default_reference_config",
]

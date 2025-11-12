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

# Optional: SAC file utilities (requires obspy)
try:
    from . import sac_utils
    from .sac_utils import load_sac_waveform, load_obspy_example_sac
    _has_sac_utils = True
except ImportError:
    _has_sac_utils = False

# Optional: Preprocessing utilities (requires scipy and optionally obspy)
try:
    from . import preprocessing_utils
    from .preprocessing_utils import (
        detrend_waveform,
        filter_waveform,
        resample_waveform,
        normalize_waveform,
        preprocess_for_paw,
        extract_windows_from_prediction
    )
    _has_preprocessing_utils = True
except ImportError:
    _has_preprocessing_utils = False
    
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

if _has_sac_utils:
    __all__.append("sac_utils")

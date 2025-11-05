"""Configuration for PAW model architecture."""

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping
import json
import yaml


@dataclass(frozen=True)
class PAWReferenceConfig:
    """Configuration for PAW architecture."""
    in_channels: int = 1
    out_channels: int = 1
    nb_filters: tuple[int, ...] = (32, 64)
    kernel_size: tuple[int, ...] = (13, 13)
    n_cnn: int = 5
    n_lstm: int = 1
    n_transformer: int = 1
    drop_rate: float = 0.4
    lstm_units: int = 64
    num_heads: int = 2
    dropout_rate: float = 0.1
    symmetry: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "nb_filters": list(self.nb_filters),
            "kernel_size": list(self.kernel_size),
            "n_cnn": self.n_cnn,
            "n_lstm": self.n_lstm,
            "n_transformer": self.n_transformer,
            "drop_rate": self.drop_rate,
            "lstm_units": self.lstm_units,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "symmetry": self.symmetry,
        }

    def update(self, **overrides: Any) -> PAWReferenceConfig:
        return replace(self, **overrides)


def default_reference_config() -> PAWReferenceConfig:
    """Default configuration matching the baseline paw_copy settings."""
    return PAWReferenceConfig()


def load_reference_config(source: Any = None) -> PAWReferenceConfig:
    """Load PAW configuration from dict, file, or use defaults."""
    if source is None or isinstance(source, PAWReferenceConfig):
        return source or PAWReferenceConfig()
    
    if isinstance(source, Mapping):
        return default_reference_config().update(**dict(source))
    
    # Load from file
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")

    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")

    return default_reference_config().update(**data)

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from ..config import PAWReferenceConfig, load_reference_config


class SpatialDropout1D(nn.Module):
    """Spatial dropout implemented through Dropout2d for 1-D signals."""

    def __init__(self, dropout_rate: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)
        x = self.dropout(x)
        return x.squeeze(2)


class CNN(nn.Module):
    """Pre-activation CNN block used in the reference PAW model."""

    def __init__(self, in_filters: int, out_filters: int, kernel_size: int, drop_rate: float) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_filters)
        self.relu = nn.ReLU()
        self.dropout = SpatialDropout1D(drop_rate)
        self.conv = nn.Conv1d(in_filters, out_filters, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.conv(x)


class LSTMBlock(nn.Module):
    """Bidirectional LSTM block followed by channel reduction."""

    def __init__(self, nb_filters: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(nb_filters, nb_filters, bidirectional=True, batch_first=True)
        self.conv1d = nn.Conv1d(nb_filters * 2, nb_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(nb_filters)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = lstm_out.transpose(1, 2)
        out = self.conv1d(out)
        out = self.bn(out)
        out = self.relu(out)
        return out.transpose(1, 2)


class TransformerBlock(nn.Module):
    """Self-attention block with residual connection."""

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        return self.norm(x)


class PAWReference(nn.Module):
    """Reference PAW architecture mirroring src/model/paw_copy.py."""

    def __init__(self, config: PAWReferenceConfig | Mapping[str, object]) -> None:
        super().__init__()
        cfg = config if isinstance(config, PAWReferenceConfig) else load_reference_config(config)
        self.config = cfg

        self.symmetry = cfg.symmetry
        self.nb_filters = list(cfg.nb_filters)
        self.kernel_size = list(cfg.kernel_size)
        if len(self.nb_filters) != len(self.kernel_size):
            raise ValueError("nb_filters and kernel_size must have the same length.")

        self.n_cnn = cfg.n_cnn
        self.n_lstm = cfg.n_lstm
        self.n_transformer = cfg.n_transformer
        self.drop_rate = cfg.drop_rate
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels

        self.encoder = self._build_encoder()
        self.cnn_blocks = nn.ModuleList(
            CNN(self.nb_filters[-1], self.nb_filters[-1], self.kernel_size[-1], self.drop_rate)
            for _ in range(self.n_cnn)
        )
        self.lstm_blocks = nn.ModuleList(LSTMBlock(self.nb_filters[-1]) for _ in range(self.n_lstm))
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(self.nb_filters[-1], num_heads=4, dropout_rate=self.drop_rate)
            for _ in range(self.n_transformer)
        )
        self.decoder = self._build_decoder()
        self.conv_out = nn.Conv1d(
            self.nb_filters[0],
            self.out_channels,
            self.kernel_size[0],
            padding=self.kernel_size[0] // 2,
        )

    def _build_encoder(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_ch = self.in_channels
        for filters, kernel in zip(self.nb_filters, self.kernel_size):
            layers.extend(
                [
                    nn.Conv1d(in_ch, filters, kernel, padding=kernel // 2),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ]
            )
            in_ch = filters
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        for idx in reversed(range(1, len(self.nb_filters))):
            layers.extend(
                [
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(self.nb_filters[idx], self.nb_filters[idx - 1], self.kernel_size[idx], padding=self.kernel_size[idx] // 2),
                    nn.BatchNorm1d(self.nb_filters[idx - 1]),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Upsample(scale_factor=2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder(x)

        for cnn in self.cnn_blocks:
            x_enc = cnn(x_enc)
        cnn_out = x_enc

        x_seq = x_enc.transpose(1, 2)
        for lstm in self.lstm_blocks:
            x_seq = lstm(x_seq)
        lstm_out = x_seq

        for transformer in self.transformer_blocks:
            x_seq = transformer(x_seq)
        transformer_out = x_seq

        x_seq = lstm_out + transformer_out if self.symmetry else transformer_out
        x_combined = x_seq.transpose(1, 2) + cnn_out
        x_dec = self.decoder(x_combined)
        return self.conv_out(x_dec)  # Return logits, not probabilities!


def build_paw_copy(
    config: PAWReferenceConfig | Mapping[str, object] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PAWReference:
    """Factory for the reference PAW architecture."""
    cfg = load_reference_config(config or {})
    model = PAWReference(cfg)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    return model


__all__ = ["PAWReference", "build_paw_copy"]

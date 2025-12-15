"""
Streaming seismic data processing with real-time arrival detection.

This module provides tools for processing continuous seismic streams,
detecting phase arrivals, and extracting amplitude windows using PAW.
"""

from .core.buffer import CircularBuffer
from .core.stream import StreamSource, ObsPyStreamSource
from .core.pipeline import StreamProcessor
from .detectors.classical import STALTADetector

__all__ = [
    "CircularBuffer",
    "StreamSource", 
    "ObsPyStreamSource",
    "StreamProcessor",
    "STALTADetector"
]

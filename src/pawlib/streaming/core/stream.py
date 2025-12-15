"""
Stream sources for continuous seismic data.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple
import time

try:
    from obspy import read, Stream, Trace
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False


class StreamSource(ABC):
    """Abstract base class for seismic data streams."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        """Iterate over data chunks.
        
        Yields:
            (data, timestamp): Data chunk and timestamp of first sample
        """
        pass
    
    @abstractmethod
    def get_sampling_rate(self) -> float:
        """Get stream sampling rate."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close stream and cleanup resources."""
        pass


class ObsPyStreamSource(StreamSource):
    """Stream source from ObsPy Stream/Trace objects.
    
    Simulates real-time streaming by chunking existing data.
    Useful for testing and replay of historical data.
    """
    
    def __init__(
        self,
        stream_or_trace,
        chunk_duration: float = 1.0,
        realtime_simulation: bool = False,
        start_time: Optional[float] = None
    ):
        """Initialize ObsPy stream source.
        
        Args:
            stream_or_trace: ObsPy Stream or Trace object
            chunk_duration: Duration of each chunk in seconds
            realtime_simulation: If True, add delays to simulate real-time
            start_time: Override start time (for testing)
        """
        if not OBSPY_AVAILABLE:
            raise ImportError("ObsPy required for ObsPyStreamSource")
        
        if isinstance(stream_or_trace, Stream):
            if len(stream_or_trace) == 0:
                raise ValueError("Empty stream provided")
            self.trace = stream_or_trace[0]  # Use first trace
        else:
            self.trace = stream_or_trace
        
        self.chunk_duration = chunk_duration
        self.realtime_simulation = realtime_simulation
        self.sampling_rate = float(self.trace.stats.sampling_rate)
        
        # Calculate chunk size in samples
        self.chunk_samples = int(chunk_duration * self.sampling_rate)
        
        # Set start time
        if start_time is not None:
            self.start_time = start_time
        else:
            self.start_time = float(self.trace.stats.starttime.timestamp)
        
        # Stream state
        self.current_pos = 0
        self.is_closed = False
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        """Iterate over data chunks."""
        last_chunk_time = time.time()
        
        while not self.is_closed and self.current_pos < len(self.trace.data):
            # Extract chunk
            end_pos = min(self.current_pos + self.chunk_samples, len(self.trace.data))
            chunk_data = self.trace.data[self.current_pos:end_pos].astype(np.float32)
            
            # Calculate timestamp
            chunk_timestamp = self.start_time + self.current_pos / self.sampling_rate
            
            # Simulate real-time if requested
            if self.realtime_simulation:
                current_time = time.time()
                elapsed = current_time - last_chunk_time
                expected_duration = len(chunk_data) / self.sampling_rate
                
                if elapsed < expected_duration:
                    time.sleep(expected_duration - elapsed)
                
                last_chunk_time = time.time()
            
            yield chunk_data, chunk_timestamp
            
            self.current_pos = end_pos
    
    def get_sampling_rate(self) -> float:
        """Get stream sampling rate."""
        return self.sampling_rate
    
    def close(self) -> None:
        """Close stream."""
        self.is_closed = True
    
    def reset(self) -> None:
        """Reset stream to beginning."""
        self.current_pos = 0
        self.is_closed = False


class SyntheticStreamSource(StreamSource):
    """Synthetic stream source for testing.
    
    Generates continuous noise with optional synthetic events.
    """
    
    def __init__(
        self,
        sampling_rate: float = 40.0,
        chunk_duration: float = 1.0,
        noise_level: float = 1.0,
        events: Optional[list] = None
    ):
        """Initialize synthetic stream.
        
        Args:
            sampling_rate: Sampling rate in Hz
            chunk_duration: Chunk duration in seconds
            noise_level: Background noise amplitude
            events: List of (time, amplitude, duration) tuples for synthetic events
        """
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.noise_level = noise_level
        self.events = events or []
        
        self.chunk_samples = int(chunk_duration * sampling_rate)
        self.current_time = 0.0
        self.is_closed = False
        
        # Random state for reproducible noise
        self.rng = np.random.RandomState(42)
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        """Generate synthetic data chunks."""
        while not self.is_closed:
            # Generate noise
            chunk_data = self.rng.normal(0, self.noise_level, self.chunk_samples).astype(np.float32)
            
            # Add synthetic events
            for event_time, amplitude, duration in self.events:
                if (event_time >= self.current_time and 
                    event_time < self.current_time + self.chunk_duration):
                    
                    # Generate synthetic event (simple sine wave)
                    event_start_idx = int((event_time - self.current_time) * self.sampling_rate)
                    event_samples = int(duration * self.sampling_rate)
                    event_end_idx = min(event_start_idx + event_samples, self.chunk_samples)
                    
                    if event_end_idx > event_start_idx:
                        t = np.arange(event_end_idx - event_start_idx) / self.sampling_rate
                        event_signal = amplitude * np.sin(2 * np.pi * 5.0 * t)  # 5 Hz event
                        chunk_data[event_start_idx:event_end_idx] += event_signal
            
            yield chunk_data, self.current_time
            
            self.current_time += self.chunk_duration
            
            # Optional: stop after some duration for testing
            if self.current_time > 300.0:  # 5 minutes
                break
    
    def get_sampling_rate(self) -> float:
        """Get sampling rate."""
        return self.sampling_rate
    
    def close(self) -> None:
        """Close stream."""
        self.is_closed = True

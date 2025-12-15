"""
Circular buffer for efficient continuous seismic data storage.
"""

import numpy as np
import threading
from typing import Optional, Tuple
from collections import namedtuple

BufferInfo = namedtuple('BufferInfo', ['size', 'capacity', 'start_time', 'sampling_rate'])


class CircularBuffer:
    """Thread-safe circular buffer for continuous seismic data.
    
    Efficiently stores a rolling window of seismic samples with timestamp tracking.
    Optimized for real-time streaming applications.
    """
    
    def __init__(self, capacity_seconds: float, sampling_rate: float, n_channels: int = 1):
        """Initialize circular buffer.
        
        Args:
            capacity_seconds: Buffer duration in seconds
            sampling_rate: Sampling rate in Hz
            n_channels: Number of channels (default: 1)
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.capacity = int(capacity_seconds * sampling_rate)
        
        # Pre-allocate buffer array
        self.buffer = np.zeros((self.capacity, n_channels), dtype=np.float32)
        
        # Buffer state
        self.size = 0  # Current number of samples
        self.write_pos = 0  # Next write position
        self.start_time = None  # Timestamp of first sample
        
        # Thread safety
        self._lock = threading.RLock()
    
    def append(self, data: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Add new samples to buffer.
        
        Args:
            data: New samples, shape (n_samples,) or (n_samples, n_channels)
            timestamp: Timestamp of first sample (optional)
        """
        with self._lock:
            # Handle input shapes
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim == 2 and data.shape[1] != self.n_channels:
                if data.shape[0] == self.n_channels:
                    data = data.T
                else:
                    raise ValueError(f"Data shape {data.shape} incompatible with {self.n_channels} channels")
            
            n_samples = data.shape[0]
            
            # Set start time on first append
            if self.start_time is None and timestamp is not None:
                self.start_time = timestamp
            
            # Handle buffer overflow
            if n_samples >= self.capacity:
                # Replace entire buffer
                self.buffer[:] = data[-self.capacity:]
                self.size = self.capacity
                self.write_pos = 0
                if timestamp is not None:
                    self.start_time = timestamp + (n_samples - self.capacity) / self.sampling_rate
            else:
                # Append to buffer
                end_pos = self.write_pos + n_samples
                
                if end_pos <= self.capacity:
                    # No wraparound
                    self.buffer[self.write_pos:end_pos] = data
                else:
                    # Wraparound required
                    split_point = self.capacity - self.write_pos
                    self.buffer[self.write_pos:] = data[:split_point]
                    self.buffer[:end_pos - self.capacity] = data[split_point:]
                
                self.write_pos = end_pos % self.capacity
                self.size = min(self.size + n_samples, self.capacity)
                
                # Update start time if buffer is full
                if self.size == self.capacity and timestamp is not None:
                    overflow_samples = max(0, self.size + n_samples - self.capacity)
                    if overflow_samples > 0:
                        self.start_time += overflow_samples / self.sampling_rate
    
    def get_window(self, start_time: float, duration: float) -> Tuple[np.ndarray, float]:
        """Extract time window from buffer.
        
        Args:
            start_time: Window start time (absolute timestamp)
            duration: Window duration in seconds
            
        Returns:
            data: Window data, shape (n_samples, n_channels)
            actual_start_time: Actual start time of returned data
        """
        with self._lock:
            if self.size == 0 or self.start_time is None:
                return np.array([]).reshape(0, self.n_channels), start_time
            
            # Convert to sample indices
            buffer_end_time = self.start_time + self.size / self.sampling_rate
            
            # Check if requested window is available
            if start_time < self.start_time or start_time >= buffer_end_time:
                return np.array([]).reshape(0, self.n_channels), start_time
            
            # Calculate sample indices
            start_offset = start_time - self.start_time
            start_idx = int(start_offset * self.sampling_rate)
            n_samples = int(duration * self.sampling_rate)
            
            # Clamp to available data
            start_idx = max(0, min(start_idx, self.size - 1))
            end_idx = min(start_idx + n_samples, self.size)
            actual_n_samples = end_idx - start_idx
            
            if actual_n_samples <= 0:
                return np.array([]).reshape(0, self.n_channels), start_time
            
            # Extract data (handle circular buffer wraparound)
            data = np.zeros((actual_n_samples, self.n_channels), dtype=np.float32)
            
            if self.size < self.capacity:
                # Buffer not full, simple slice
                data[:] = self.buffer[start_idx:end_idx]
            else:
                # Buffer is full, handle wraparound
                read_start = (self.write_pos - self.size + start_idx) % self.capacity
                read_end = read_start + actual_n_samples
                
                if read_end <= self.capacity:
                    # No wraparound in read
                    data[:] = self.buffer[read_start:read_end]
                else:
                    # Wraparound in read
                    split_point = self.capacity - read_start
                    data[:split_point] = self.buffer[read_start:]
                    data[split_point:] = self.buffer[:read_end - self.capacity]
            
            actual_start_time = self.start_time + start_idx / self.sampling_rate
            return data, actual_start_time
    
    def get_latest(self, duration: float) -> Tuple[np.ndarray, float]:
        """Get most recent data from buffer.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            data: Latest data, shape (n_samples, n_channels)  
            start_time: Start time of returned data
        """
        with self._lock:
            if self.size == 0 or self.start_time is None:
                return np.array([]).reshape(0, self.n_channels), 0.0
            
            end_time = self.start_time + self.size / self.sampling_rate
            start_time = max(self.start_time, end_time - duration)
            
            return self.get_window(start_time, duration)
    
    def info(self) -> BufferInfo:
        """Get buffer information.
        
        Returns:
            BufferInfo with current state
        """
        with self._lock:
            return BufferInfo(
                size=self.size,
                capacity=self.capacity,
                start_time=self.start_time,
                sampling_rate=self.sampling_rate
            )
    
    def clear(self) -> None:
        """Clear buffer contents."""
        with self._lock:
            self.size = 0
            self.write_pos = 0
            self.start_time = None
            self.buffer.fill(0.0)

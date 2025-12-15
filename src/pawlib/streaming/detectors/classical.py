"""
Classical seismic arrival detection methods.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class Detection:
    """Seismic arrival detection result."""
    timestamp: float  # Arrival time (absolute)
    confidence: float  # Detection confidence (STA/LTA ratio)
    sta_value: float  # Short-term average
    lta_value: float  # Long-term average
    channel: int = 0  # Channel index


class STALTADetector:
    """Short-Term Average / Long-Term Average arrival detector.
    
    Efficient streaming implementation using running averages.
    Detects sudden amplitude increases characteristic of seismic arrivals.
    """
    
    def __init__(
        self,
        sta_window: float = 0.5,
        lta_window: float = 10.0,
        trigger_threshold: float = 3.0,
        detrigger_threshold: float = 1.5,
        min_event_spacing: float = 5.0,
        sampling_rate: float = 40.0
    ):
        """Initialize STA/LTA detector.
        
        Args:
            sta_window: Short-term window duration (seconds)
            lta_window: Long-term window duration (seconds)  
            trigger_threshold: STA/LTA ratio to trigger detection
            detrigger_threshold: STA/LTA ratio to end detection
            min_event_spacing: Minimum time between detections (seconds)
            sampling_rate: Data sampling rate (Hz)
        """
        self.sta_window = sta_window
        self.lta_window = lta_window
        self.trigger_threshold = trigger_threshold
        self.detrigger_threshold = detrigger_threshold
        self.min_event_spacing = min_event_spacing
        self.sampling_rate = sampling_rate
        
        # Convert to samples
        self.sta_samples = int(sta_window * sampling_rate)
        self.lta_samples = int(lta_window * sampling_rate)
        
        # State tracking
        self.sta_buffer = deque(maxlen=self.sta_samples)
        self.lta_buffer = deque(maxlen=self.lta_samples)
        self.sta_sum = 0.0
        self.lta_sum = 0.0
        
        self.is_triggered = False
        self.last_detection_time = -np.inf
        self.sample_count = 0
        self.start_time = None
    
    def reset(self) -> None:
        """Reset detector state."""
        self.sta_buffer.clear()
        self.lta_buffer.clear()
        self.sta_sum = 0.0
        self.lta_sum = 0.0
        self.is_triggered = False
        self.last_detection_time = -np.inf
        self.sample_count = 0
        self.start_time = None
    
    def process_chunk(self, data: np.ndarray, timestamp: Optional[float] = None) -> List[Detection]:
        """Process a chunk of data and return any detections.
        
        Args:
            data: Input data, shape (n_samples,) or (n_samples, n_channels)
            timestamp: Timestamp of first sample
            
        Returns:
            List of Detection objects
        """
        # Handle input shapes
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_channels = data.shape
        detections = []
        
        # Set start time on first chunk
        if self.start_time is None and timestamp is not None:
            self.start_time = timestamp
        
        # Process each sample
        for i in range(n_samples):
            current_time = (self.start_time or 0.0) + self.sample_count / self.sampling_rate
            
            # Process each channel (for now, just use first channel)
            sample = data[i, 0]
            
            # Compute characteristic function (squared amplitude)
            cf_value = sample ** 2
            
            # Update STA buffer
            if len(self.sta_buffer) == self.sta_samples:
                self.sta_sum -= self.sta_buffer[0]
            self.sta_buffer.append(cf_value)
            self.sta_sum += cf_value
            
            # Update LTA buffer  
            if len(self.lta_buffer) == self.lta_samples:
                self.lta_sum -= self.lta_buffer[0]
            self.lta_buffer.append(cf_value)
            self.lta_sum += cf_value
            
            # Compute STA/LTA ratio
            if len(self.sta_buffer) >= self.sta_samples and len(self.lta_buffer) >= self.lta_samples:
                sta_avg = self.sta_sum / len(self.sta_buffer)
                lta_avg = self.lta_sum / len(self.lta_buffer)
                
                if lta_avg > 0:
                    ratio = sta_avg / lta_avg
                else:
                    ratio = 0.0
                
                # Check for trigger
                if not self.is_triggered and ratio >= self.trigger_threshold:
                    # Check minimum event spacing
                    if current_time - self.last_detection_time >= self.min_event_spacing:
                        self.is_triggered = True
                        self.last_detection_time = current_time
                        
                        detection = Detection(
                            timestamp=current_time,
                            confidence=ratio,
                            sta_value=sta_avg,
                            lta_value=lta_avg,
                            channel=0
                        )
                        detections.append(detection)
                
                # Check for detrigger
                elif self.is_triggered and ratio <= self.detrigger_threshold:
                    self.is_triggered = False
            
            self.sample_count += 1
        
        return detections
    
    def get_current_ratio(self) -> float:
        """Get current STA/LTA ratio."""
        if len(self.sta_buffer) >= self.sta_samples and len(self.lta_buffer) >= self.lta_samples:
            sta_avg = self.sta_sum / len(self.sta_buffer)
            lta_avg = self.lta_sum / len(self.lta_buffer)
            return sta_avg / lta_avg if lta_avg > 0 else 0.0
        return 0.0
    
    def is_ready(self) -> bool:
        """Check if detector has enough data to operate."""
        return len(self.lta_buffer) >= self.lta_samples


class AICPicker:
    """Akaike Information Criterion picker for precise onset timing.
    
    Refines STA/LTA trigger times by finding the point of maximum
    change in signal characteristics.
    """
    
    def __init__(self, window_duration: float = 2.0, sampling_rate: float = 40.0):
        """Initialize AIC picker.
        
        Args:
            window_duration: Analysis window duration (seconds)
            sampling_rate: Data sampling rate (Hz)
        """
        self.window_duration = window_duration
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_duration * sampling_rate)
    
    def refine_pick(self, data: np.ndarray, trigger_time: float, data_start_time: float) -> float:
        """Refine arrival time using AIC criterion.
        
        Args:
            data: Waveform data, shape (n_samples,) or (n_samples, 1)
            trigger_time: Initial trigger time (absolute)
            data_start_time: Start time of data array
            
        Returns:
            Refined arrival time (absolute)
        """
        if data.ndim == 2:
            data = data[:, 0]
        
        # Convert trigger time to sample index
        trigger_offset = trigger_time - data_start_time
        trigger_idx = int(trigger_offset * self.sampling_rate)
        
        # Define analysis window around trigger
        half_window = self.window_samples // 2
        start_idx = max(0, trigger_idx - half_window)
        end_idx = min(len(data), trigger_idx + half_window)
        
        if end_idx - start_idx < 10:  # Need minimum samples
            return trigger_time
        
        window_data = data[start_idx:end_idx]
        aic_values = self._compute_aic(window_data)
        
        # Find minimum AIC (best pick)
        if len(aic_values) > 0:
            min_idx = np.argmin(aic_values)
            refined_idx = start_idx + min_idx
            refined_time = data_start_time + refined_idx / self.sampling_rate
            return refined_time
        
        return trigger_time
    
    def _compute_aic(self, data: np.ndarray) -> np.ndarray:
        """Compute AIC function for onset detection."""
        n = len(data)
        aic = np.zeros(n)
        
        for k in range(1, n - 1):
            # Split data at point k
            x1 = data[:k]
            x2 = data[k:]
            
            # Compute variances
            var1 = np.var(x1) if len(x1) > 1 else 0.0
            var2 = np.var(x2) if len(x2) > 1 else 0.0
            
            # AIC formula
            if var1 > 0 and var2 > 0:
                aic[k] = k * np.log(var1) + (n - k) * np.log(var2)
            else:
                aic[k] = np.inf
        
        return aic

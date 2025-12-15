"""
Main streaming processing pipeline.
"""

import numpy as np
from typing import List, Optional, Iterator, Callable
import threading
import time
from dataclasses import dataclass

from .buffer import CircularBuffer
from .stream import StreamSource
from ..detectors.classical import STALTADetector, Detection

try:
    from ...paw import PAW
    from ...preprocessing_utils import preprocess_for_paw, extract_windows_from_prediction
    PAW_AVAILABLE = True
except ImportError:
    PAW_AVAILABLE = False


@dataclass
class AmplitudeWindow:
    """Detected amplitude window result."""
    detection_time: float  # Original arrival detection time
    start_time: float      # Amplitude window start (absolute time)
    end_time: float        # Amplitude window end (absolute time)
    confidence: float      # Detection confidence
    paw_confidence: float  # PAW prediction confidence (max value in window)


class StreamProcessor:
    """Main streaming processor coordinating detection and PAW analysis.
    
    Processes continuous seismic streams, detects arrivals using STA/LTA,
    and extracts amplitude windows using PAW model.
    """
    
    def __init__(
        self,
        stream_source: StreamSource,
        detector: Optional[STALTADetector] = None,
        paw_model: Optional['PAW'] = None,
        buffer_duration: float = 60.0,
        window_duration: float = 5.0,
        window_padding: float = 0.5,
        preprocessing_params: Optional[dict] = None
    ):
        """Initialize stream processor.
        
        Args:
            stream_source: Source of streaming data
            detector: Arrival detector (default: STA/LTA with standard params)
            paw_model: PAW model for amplitude window detection
            buffer_duration: Buffer duration in seconds
            window_duration: PAW analysis window duration
            window_padding: Padding around analysis window
            preprocessing_params: Parameters for preprocessing pipeline
        """
        self.stream_source = stream_source
        self.sampling_rate = stream_source.get_sampling_rate()
        
        # Initialize detector
        if detector is None:
            self.detector = STALTADetector(sampling_rate=self.sampling_rate)
        else:
            self.detector = detector
        
        self.paw_model = paw_model
        self.window_duration = window_duration
        self.window_padding = window_padding
        
        # Preprocessing parameters
        self.preprocessing_params = preprocessing_params or {
            'detrend': True,
            'apply_filter': True,
            'freqmin': 1.0,
            'freqmax': 15.0,
            'target_freq': 40.0,
            'normalize': True
        }
        
        # Initialize buffer
        self.buffer = CircularBuffer(
            capacity_seconds=buffer_duration,
            sampling_rate=self.sampling_rate,
            n_channels=1
        )
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.results_queue = []
        self.results_lock = threading.Lock()
        
        # Detection queue for delayed PAW analysis
        self.pending_detections = []
        self.detection_delay = self.window_duration + self.window_padding  # Wait for future data
        
        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'detections_found': 0,
            'paw_analyses': 0,
            'processing_errors': 0
        }
    
    def start(self) -> None:
        """Start streaming processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
    
    def stop(self) -> None:
        """Stop streaming processing."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        self.stream_source.close()
    
    def get_results(self) -> List[AmplitudeWindow]:
        """Get all accumulated results and clear queue."""
        with self.results_lock:
            results = self.results_queue.copy()
            self.results_queue.clear()
            return results
    
    def get_latest_results(self, max_age: float = 60.0) -> List[AmplitudeWindow]:
        """Get recent results within max_age seconds."""
        current_time = time.time()
        with self.results_lock:
            recent_results = [
                r for r in self.results_queue 
                if current_time - r.detection_time <= max_age
            ]
            return recent_results
    
    def _processing_loop(self) -> None:
        """Main processing loop."""
        try:
            for chunk_data, chunk_timestamp in self.stream_source:
                if not self.is_running:
                    break
                
                # Add to buffer
                self.buffer.append(chunk_data, chunk_timestamp)
                
                # Process with detector
                detections = self.detector.process_chunk(chunk_data, chunk_timestamp)
                
                # Add new detections to pending queue
                for detection in detections:
                    self.pending_detections.append(detection)
                
                # Process pending detections that are old enough
                current_time = chunk_timestamp + len(chunk_data) / self.sampling_rate
                ready_detections = []
                remaining_detections = []
                
                for detection in self.pending_detections:
                    if current_time - detection.timestamp >= self.detection_delay:
                        ready_detections.append(detection)
                    else:
                        remaining_detections.append(detection)
                
                self.pending_detections = remaining_detections
                
                # Process ready detections with PAW
                for detection in ready_detections:
                    try:
                        amplitude_window = self._analyze_with_paw(detection)
                        if amplitude_window:
                            with self.results_lock:
                                self.results_queue.append(amplitude_window)
                            self.stats['paw_analyses'] += 1
                    except Exception as e:
                        print(f"PAW analysis error: {e}")
                        self.stats['processing_errors'] += 1
                
                self.stats['chunks_processed'] += 1
                self.stats['detections_found'] += len(detections)
                
        except Exception as e:
            print(f"Processing loop error: {e}")
            self.stats['processing_errors'] += 1
        finally:
            self.is_running = False
    
    def _analyze_with_paw(self, detection: Detection) -> Optional[AmplitudeWindow]:
        """Analyze detection with PAW model."""
        if not PAW_AVAILABLE or self.paw_model is None:
            return None
        
        # Check if buffer has enough data for analysis
        buffer_info = self.buffer.info()
        if buffer_info.size == 0 or buffer_info.start_time is None:
            return None
        
        buffer_end_time = buffer_info.start_time + buffer_info.size / self.sampling_rate
        total_duration = self.window_duration + 2 * self.window_padding
        
        # Check if we have enough buffer history for the requested window
        required_start_time = detection.timestamp - self.window_padding
        if (required_start_time < buffer_info.start_time or 
            detection.timestamp + self.window_duration + self.window_padding > buffer_end_time):
            # Not enough buffer data available
            return None
        
        # Extract window from buffer
        window_data, actual_start_time = self.buffer.get_window(
            required_start_time,
            total_duration
        )
        
        if window_data.shape[0] == 0:
            return None
        
        # Check if we have enough data (should be close to expected now)
        expected_samples = int(total_duration * self.sampling_rate)
        if window_data.shape[0] < expected_samples * 0.9:  # Stricter tolerance
            return None
        
        try:
            # Preprocess for PAW
            waveform = window_data.reshape(1, -1, 1)  # (1, T, 1)
            
            if PAW_AVAILABLE:
                waveform = preprocess_for_paw(
                    waveform,
                    self.sampling_rate,
                    **self.preprocessing_params
                )
            
            # Run PAW inference
            prediction = self.paw_model.predict(waveform)
            
            # Extract amplitude window
            if PAW_AVAILABLE:
                windows = extract_windows_from_prediction(prediction)
                
                if windows.shape[0] > 0 and windows[0, 1] > windows[0, 0]:
                    # Apply half-cycle cropping (this is missing from the predict method!)
                    # Convert waveform to tensor for half-cycle cropping
                    import torch
                    
                    # IMPORTANT: PAW adds 20 samples padding on each side, so we need to account for this
                    # The prediction coordinates are in the padded space, but _limit_to_half_cycle
                    # expects coordinates in the original signal space
                    
                    # Get the padded signal that PAW actually processed
                    padded_signal = torch.from_numpy(waveform[:, :, 0]).float()  # (N, T) - this includes padding
                    windows_tensor = torch.from_numpy(windows).float()
                    
                    # Apply the same half-cycle cropping as in evaluation
                    cropped_windows = self.paw_model._limit_to_half_cycle(padded_signal, windows_tensor)
                    windows = cropped_windows.numpy().astype(int)
                    
                    # Convert indices to absolute times
                    # Note: prediction is at 40 Hz after preprocessing
                    target_freq = self.preprocessing_params.get('target_freq', 40.0)
                    
                    start_idx = windows[0, 0]
                    end_idx = windows[0, 1]
                    
                    # Convert to seconds relative to window start
                    start_offset = start_idx / target_freq
                    end_offset = end_idx / target_freq
                    
                    # Convert to absolute times
                    # Account for the fact that we removed padding before PAW
                    abs_start_time = detection.timestamp + start_offset
                    abs_end_time = detection.timestamp + end_offset
                    
                    # Calculate PAW confidence (max prediction value in window)
                    paw_confidence = float(prediction[0, 0, start_idx:end_idx+1].max())
                    
                    return AmplitudeWindow(
                        detection_time=detection.timestamp,
                        start_time=abs_start_time,
                        end_time=abs_end_time,
                        confidence=detection.confidence,
                        paw_confidence=paw_confidence
                    )
            
        except Exception as e:
            print(f"PAW processing error: {e}")
            raise
        
        return None
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        stats = self.stats.copy()
        stats['is_running'] = self.is_running
        stats['buffer_info'] = self.buffer.info()._asdict()
        stats['detector_ready'] = self.detector.is_ready()
        stats['current_sta_lta'] = self.detector.get_current_ratio()
        return stats


# Convenience function for simple usage
def process_stream_simple(
    stream_source: StreamSource,
    duration: float = 60.0,
    paw_model_path: Optional[str] = None,
    **detector_params
) -> List[AmplitudeWindow]:
    """Simple function to process a stream for a fixed duration.
    
    Args:
        stream_source: Stream to process
        duration: Processing duration in seconds
        paw_model_path: Path to PAW model (optional)
        **detector_params: Parameters for STA/LTA detector
        
    Returns:
        List of detected amplitude windows
    """
    # Load PAW model if provided
    paw_model = None
    if paw_model_path and PAW_AVAILABLE:
        paw_model = PAW.from_pretrained(paw_model_path)
    
    # Create detector
    detector = STALTADetector(
        sampling_rate=stream_source.get_sampling_rate(),
        **detector_params
    )
    
    # Create processor
    processor = StreamProcessor(
        stream_source=stream_source,
        detector=detector,
        paw_model=paw_model
    )
    
    # Process for specified duration
    processor.start()
    time.sleep(duration)
    processor.stop()
    
    return processor.get_results()

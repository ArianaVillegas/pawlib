"""
Preprocessing utilities for seismic waveforms.
"""

import numpy as np
from scipy import signal as scipy_signal

try:
    from obspy import Trace
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False


def detrend_waveform(waveform: np.ndarray, detrend_type: str = 'linear') -> np.ndarray:
    """
    Remove trend from waveform.
    
    Args:
        waveform: Input waveform array (1, T, 1) or (T,)
        detrend_type: Type of detrending ('linear' or 'constant')
        
    Returns:
        Detrended waveform with same shape as input
    """
    original_shape = waveform.shape
    data = waveform.reshape(-1)
    
    detrended = scipy_signal.detrend(data, type=detrend_type)
    
    return detrended.reshape(original_shape).astype(np.float32)


def filter_waveform(
    waveform: np.ndarray,
    sampling_rate: float,
    freqmin: float,
    freqmax: float,
    corners: int = 4,
    zerophase: bool = True
) -> np.ndarray:
    """
    Apply bandpass filter to waveform using ObsPy.
    
    Args:
        waveform: Input waveform array (1, T, 1) or (T,)
        sampling_rate: Sampling rate in Hz
        freqmin: High-pass corner frequency in Hz
        freqmax: Low-pass corner frequency in Hz
        corners: Filter corners (default: 4)
        zerophase: Use zero-phase filter (default: True)
        
    Returns:
        Filtered waveform with same shape as input
    """
    if not OBSPY_AVAILABLE:
        raise ImportError("ObsPy is required for filtering. Install with: pip install obspy")
    
    original_shape = waveform.shape
    data = waveform.reshape(-1)
    
    # Create temporary ObsPy trace for filtering
    trace = Trace(data=data)
    trace.stats.sampling_rate = sampling_rate
    trace.filter('bandpass', freqmin=freqmin, freqmax=freqmax, 
                 corners=corners, zerophase=zerophase)
    
    return trace.data.reshape(original_shape).astype(np.float32)


def resample_waveform(
    waveform: np.ndarray,
    original_freq: float,
    target_freq: float
) -> np.ndarray:
    """
    Resample waveform to target frequency.
    
    Args:
        waveform: Input waveform array (1, T, 1) or (T,)
        original_freq: Original sampling frequency in Hz
        target_freq: Target sampling frequency in Hz
        
    Returns:
        Resampled waveform
    """
    if abs(original_freq - target_freq) < 0.01:
        return waveform  # No resampling needed
    
    original_shape = waveform.shape
    data = waveform.reshape(-1)
    
    # Calculate new number of samples
    duration = len(data) / original_freq
    num_samples = int(duration * target_freq)
    
    # Resample
    resampled = scipy_signal.resample(data, num_samples)
    
    # Determine output shape
    if len(original_shape) == 3:
        return resampled.reshape(1, -1, 1).astype(np.float32)
    else:
        return resampled.astype(np.float32)


def normalize_waveform(waveform: np.ndarray, method: str = 'max') -> np.ndarray:
    """
    Normalize waveform.
    
    Args:
        waveform: Input waveform array
        method: Normalization method:
            - 'max': Scale to [-1, 1] using max absolute value
            - 'zscore': Z-score normalization (mean=0, std=1)
            - 'minmax': Min-max scaling to [0, 1]
            
    Returns:
        Normalized waveform
    """
    data = waveform.copy()
    
    if method == 'max':
        # Scale to [-1, 1]
        max_abs = np.abs(data).max()
        if max_abs > 0:
            data = data / max_abs
    
    elif method == 'zscore':
        # Z-score normalization
        mean = data.mean()
        std = data.std()
        if std > 0:
            data = (data - mean) / std
        else:
            data = data - mean
    
    elif method == 'minmax':
        # Min-max to [0, 1]
        min_val = data.min()
        max_val = data.max()
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return data.astype(np.float32)


def extract_windows_from_prediction(
    prediction: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """Extract window boundaries from PAW prediction using peak-based approach.
    
    Finds the highest prediction peak and expands left/right until values drop below threshold.
    
    Args:
        prediction: Model prediction, shape (N, C, T) or (N, T)
        threshold: Threshold to stop expansion (default: 0.5)
        
    Returns:
        windows: Array of shape (N, 2) with [start_idx, end_idx] for each sample
    """
    # Handle different input shapes
    if prediction.ndim == 3:
        pred = prediction[:, 0, :]  # (N, T)
    else:
        pred = prediction
    
    batch_size = pred.shape[0]
    windows = np.zeros((batch_size, 2), dtype=np.int32)
    
    for i in range(batch_size):
        signal = pred[i]
        
        # Find the peak (highest value)
        peak_idx = np.argmax(signal)
        peak_value = signal[peak_idx]
        
        # Skip if peak is below threshold
        if peak_value < threshold:
            continue
        
        # Expand left from peak until below threshold
        start_idx = peak_idx
        while start_idx > 0 and signal[start_idx - 1] >= threshold:
            start_idx -= 1
        
        # Expand right from peak until below threshold  
        end_idx = peak_idx
        while end_idx < len(signal) - 1 and signal[end_idx + 1] >= threshold:
            end_idx += 1
            
        windows[i, 0] = start_idx
        windows[i, 1] = end_idx
    
    return windows


def preprocess_for_paw(
    waveform: np.ndarray,
    sampling_rate: float,
    detrend: bool = True,
    apply_filter: bool = True,
    freqmin: float = 1.0,
    freqmax: float = 15.0,
    target_freq: float = 40.0,
    normalize: bool = True,
    verbose: bool = False
) -> np.ndarray:
    """Complete preprocessing pipeline for PAW model.
    
    Args:
        waveform: Input waveform array
        sampling_rate: Original sampling rate in Hz
        detrend: Remove linear trend
        apply_filter: Apply bandpass filter
        freqmin: High-pass corner in Hz
        freqmax: Low-pass corner in Hz
        target_freq: Target resampling frequency
        normalize: Normalize to [-1, 1]
        verbose: Print steps
        
    Returns:
        Preprocessed waveform
    """
    data = waveform.copy()
    
    if verbose:
        print(" Preprocessing:")
    
    if detrend:
        data = detrend_waveform(data)
        verbose and print("   Detrended")
    
    if apply_filter:
        data = filter_waveform(data, sampling_rate, freqmin, freqmax)
        verbose and print(f"   Filtered ({freqmin}-{freqmax} Hz)")
    
    if abs(sampling_rate - target_freq) > 0.01:
        data = resample_waveform(data, sampling_rate, target_freq)
        verbose and print(f"   Resampled ({sampling_rate:.1f} {target_freq:.1f} Hz)")
    
    if normalize:
        data = normalize_waveform(data, method='max')
        verbose and print("   Normalized")
    
    return data


__all__ = [
    'detrend_waveform',
    'filter_waveform', 
    'resample_waveform',
    'normalize_waveform',
    'preprocess_for_paw',
    'extract_windows_from_prediction'
]

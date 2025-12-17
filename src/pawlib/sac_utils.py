"""
Utilities for loading and processing SAC files for PAW inference.
"""
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

try:
    from obspy import read
    from obspy.core.stream import Stream
    from obspy.core.util import get_example_file
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False


def get_onset_time_from_sac(trace) -> Optional[float]:
    """Extract onset/pick time from SAC header."""
    # Check if SAC header exists
    if not hasattr(trace.stats, 'sac'):
        return None
    
    sac = trace.stats.sac
    # Check t0 (manual picks) and 'a' (first arrival). -12345.0 is SAC undefined value
    for field in ['t0', 'a']:
        if hasattr(sac, field) and getattr(sac, field) != -12345.0:
            return getattr(sac, field)
    return None


def load_obspy_example_sac() -> Tuple[str, float]:
    """Load an example SAC file from ObsPy's test data."""
    if not OBSPY_AVAILABLE:
        raise ImportError(
            "ObsPy is required. Install with: pip install obspy"
        )
    
    # Get example SAC file from ObsPy
    sac_file = get_example_file('test.sac')
    
    # Read it to get onset time
    stream = read(sac_file)
    trace = stream[0]
    
    # Try to get onset from header, or use middle of trace
    duration = len(trace.data) / trace.stats.sampling_rate
    onset = get_onset_time_from_sac(trace) or min(10.0, duration / 2)
    print(f"Onset: {onset:.3f}s {'(from header)' if get_onset_time_from_sac(trace) else '(default)'}")
    
    return sac_file, onset


def load_sac_waveform(
    sac_path: str,
    onset_time: Optional[float] = None,
    window_duration: float = 5.0,
    padding: float = 0.5
) -> Tuple[np.ndarray, dict]:
    """Load SAC file and extract waveform window.
    
    This is a minimal loader that only:
    1. Reads the SAC file using ObsPy
    2. Extracts a window around the onset time
    3. Returns raw data + metadata
    
    For preprocessing (filtering, resampling, normalization), use the
    separate preprocessing functions or do it manually in your notebook.
    
    Args:
        sac_path: Path to SAC file
        onset_time: Onset time in seconds from start of trace
        window_duration: Duration of signal window in seconds (default: 5.0)
        padding: Padding duration in seconds on each side (default: 0.5)
        
    Returns:
        waveform: Raw waveform array of shape (1, T, 1) at original sampling rate
        metadata: Dictionary with trace metadata and original ObsPy trace
        
    Example:
        >>> # Load raw waveform
        >>> waveform, meta = load_sac_waveform('signal.sac', onset_time=10.5)
        >>> # Then preprocess in your notebook (see preprocessing functions)
    """
    if not OBSPY_AVAILABLE:
        raise ImportError(
            "ObsPy is required for SAC file loading. Install with: pip install obspy"
        )
    
    sac_path = Path(sac_path)
    if not sac_path.exists():
        raise FileNotFoundError(f"SAC file not found: {sac_path}")
    
    # Read SAC file
    stream = read(str(sac_path))
    trace = stream[0]  # Get first trace
    
    # If onset_time not provided, try to read from SAC header
    if onset_time is None:
        onset_time = get_onset_time_from_sac(trace)
        if onset_time is None:
            raise ValueError(
                "No onset time provided and none found in SAC header. "
                "Please provide onset_time parameter."
            )
    
    # Get original sampling rate
    original_freq = trace.stats.sampling_rate
    original_dt = 1.0 / original_freq
    
    # Calculate sample indices for the window
    total_duration = window_duration + 2 * padding
    start_time = onset_time - padding
    end_time = onset_time + window_duration + padding
    
    # Convert to sample indices
    start_sample = int(start_time * original_freq)
    end_sample = int(end_time * original_freq)
    
    # Extract window with zero-padding if needed
    pad_left = max(0, -start_sample)
    pad_right = max(0, end_sample - len(trace.data))
    start_sample = max(0, start_sample)
    end_sample = min(end_sample, len(trace.data))
    
    data_segment = trace.data[start_sample:end_sample]
    if pad_left > 0:
        data_segment = np.concatenate([np.zeros(pad_left), data_segment])
    if pad_right > 0:
        data_segment = np.concatenate([data_segment, np.zeros(pad_right)])
    
    # Return raw data without preprocessing
    waveform = data_segment.reshape(1, -1, 1).astype(np.float32)
    
    # Collect metadata
    metadata = {
        'station': trace.stats.station,
        'channel': trace.stats.channel,
        'network': trace.stats.network,
        'sampling_rate': original_freq,
        'onset_time': onset_time,
        'window_start': start_time,
        'window_end': end_time,
        'n_samples': waveform.shape[1],
        'trace': trace
    }
    
    return waveform, metadata


def batch_load_sac_files(
    sac_files: list,
    onset_times: Optional[list] = None,
    window_duration: float = 5.0,
    padding: float = 0.5
) -> Tuple[np.ndarray, list]:
    """Load multiple SAC files for batch inference.
    
    Args:
        sac_files: List of paths to SAC files
        onset_times: List of onset times (one per file). If None, read from SAC headers
        window_duration: Duration of signal window in seconds
        padding: Padding duration in seconds on each side
        
    Returns:
        waveforms: Batch of waveforms, shape (N, T, 1)
        metadata_list: List of metadata dicts
    """
    if onset_times and len(sac_files) != len(onset_times):
        raise ValueError(f"Files/onset count mismatch: {len(sac_files)} vs {len(onset_times)}")
    
    waveforms_list, metadata_list = [], []
    onset_iter = onset_times or [None] * len(sac_files)
    
    for sac_file, onset_time in zip(sac_files, onset_iter):
        try:
            wf, meta = load_sac_waveform(sac_file, onset_time, window_duration, padding)
            waveforms_list.append(wf)
            metadata_list.append(meta)
        except Exception as e:
            print(f"Warning: Failed to load {sac_file}: {e}")
            metadata_list.append({'error': str(e), 'file': str(sac_file)})
    
    if not waveforms_list:
        raise RuntimeError("No waveforms loaded successfully")
    
    return np.concatenate(waveforms_list, axis=0), metadata_list


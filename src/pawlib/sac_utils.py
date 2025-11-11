"""
Utilities for loading and processing SAC files for PAW inference.
"""
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

try:
    from obspy import read
    from obspy.core.stream import Stream
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False


def load_sac_waveform(
    sac_path: str,
    onset_time: float,
    target_freq: float = 40.0,
    window_duration: float = 5.0,
    padding: float = 0.5
) -> Tuple[np.ndarray, dict]:
    """Load and preprocess SAC file for PAW inference.
    
    This function:
    1. Reads the SAC file using ObsPy
    2. Extracts a window around the onset time
    3. Resamples to target frequency (40 Hz)
    4. Adds padding (0.5s on each side)
    5. Normalizes the waveform
    
    Args:
        sac_path: Path to SAC file
        onset_time: Onset time in seconds from start of trace
        target_freq: Target sampling frequency in Hz (default: 40)
        window_duration: Duration of signal window in seconds (default: 5.0)
        padding: Padding duration in seconds on each side (default: 0.5)
        
    Returns:
        waveform: Preprocessed waveform array of shape (1, T, 1) where T = (window_duration + 2*padding) * target_freq
        metadata: Dictionary with trace metadata
        
    Example:
        >>> waveform, meta = load_sac_waveform('signal.sac', onset_time=10.5)
        >>> # waveform shape: (1, 240, 1) for 6 seconds at 40 Hz
        >>> model = PAW.from_pretrained('hf://ArianaVillegas/pawlib-pretrained/paw_corrected.pt')
        >>> prediction = model.predict(waveform)
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
    
    # Extract window
    if start_sample < 0:
        # Pad with zeros at the beginning
        pad_samples = -start_sample
        data_segment = trace.data[:end_sample]
        data_segment = np.concatenate([np.zeros(pad_samples), data_segment])
        start_sample = 0
    elif end_sample > len(trace.data):
        # Pad with zeros at the end
        data_segment = trace.data[start_sample:]
        pad_samples = end_sample - len(trace.data)
        data_segment = np.concatenate([data_segment, np.zeros(pad_samples)])
    else:
        data_segment = trace.data[start_sample:end_sample]
    
    # Resample to target frequency if needed
    if abs(original_freq - target_freq) > 0.01:  # If frequencies differ
        from scipy import signal
        num_samples = int(total_duration * target_freq)
        data_resampled = signal.resample(data_segment, num_samples)
    else:
        data_resampled = data_segment
    
    # Normalize (z-score normalization)
    mean = data_resampled.mean()
    std = data_resampled.std()
    if std > 0:
        data_normalized = (data_resampled - mean) / std
    else:
        data_normalized = data_resampled - mean
    
    # Reshape to (1, T, 1) for model input
    waveform = data_normalized.reshape(1, -1, 1).astype(np.float32)
    
    # Collect metadata
    metadata = {
        'station': trace.stats.station,
        'channel': trace.stats.channel,
        'network': trace.stats.network,
        'location': trace.stats.location,
        'start_time': str(trace.stats.starttime),
        'original_sampling_rate': original_freq,
        'resampled_rate': target_freq,
        'onset_time': onset_time,
        'window_start': start_time,
        'window_end': end_time,
        'n_samples': waveform.shape[1]
    }
    
    return waveform, metadata


def batch_load_sac_files(
    sac_files: list,
    onset_times: list,
    target_freq: float = 40.0,
    window_duration: float = 5.0,
    padding: float = 0.5
) -> Tuple[np.ndarray, list]:
    """Load multiple SAC files for batch inference.
    
    Args:
        sac_files: List of paths to SAC files
        onset_times: List of onset times (one per file)
        target_freq: Target sampling frequency in Hz
        window_duration: Duration of signal window in seconds
        padding: Padding duration in seconds on each side
        
    Returns:
        waveforms: Batch of waveforms, shape (N, T, 1)
        metadata_list: List of metadata dicts
        
    Example:
        >>> files = ['event1.sac', 'event2.sac', 'event3.sac']
        >>> onsets = [10.5, 15.2, 8.7]
        >>> waveforms, metadata = batch_load_sac_files(files, onsets)
        >>> predictions = model.predict(waveforms)
    """
    if len(sac_files) != len(onset_times):
        raise ValueError(f"Number of files ({len(sac_files)}) must match number of onset times ({len(onset_times)})")
    
    waveforms_list = []
    metadata_list = []
    
    for sac_file, onset_time in zip(sac_files, onset_times):
        try:
            waveform, metadata = load_sac_waveform(
                sac_file, onset_time, target_freq, window_duration, padding
            )
            waveforms_list.append(waveform)
            metadata_list.append(metadata)
        except Exception as e:
            print(f"Warning: Failed to load {sac_file}: {e}")
            metadata_list.append({'error': str(e), 'file': str(sac_file)})
    
    if not waveforms_list:
        raise RuntimeError("No waveforms could be loaded successfully")
    
    # Stack into batch
    waveforms_batch = np.concatenate(waveforms_list, axis=0)
    
    return waveforms_batch, metadata_list


def extract_predicted_window(
    prediction: np.ndarray,
    metadata: dict,
    threshold: float = 0.5
) -> Tuple[float, float]:
    """Extract predicted arrival window from model output.
    
    Args:
        prediction: Model prediction mask, shape (T,) or (1, T)
        metadata: Metadata dict from load_sac_waveform
        threshold: Threshold for binary mask
        
    Returns:
        start_time: Predicted window start time (seconds from trace start)
        end_time: Predicted window end time (seconds from trace start)
        
    Example:
        >>> waveform, metadata = load_sac_waveform('signal.sac', onset_time=10.5)
        >>> prediction = model.predict(waveform)
        >>> start, end = extract_predicted_window(prediction[0], metadata)
        >>> print(f"Predicted arrival window: {start:.2f}s - {end:.2f}s")
    """
    # Flatten prediction if needed
    if prediction.ndim > 1:
        prediction = prediction.squeeze()
    
    # Apply threshold
    binary_mask = (prediction > threshold).astype(int)
    
    # Find first and last activated samples
    activated = np.where(binary_mask > 0)[0]
    
    if len(activated) == 0:
        # No detection
        return None, None
    
    start_sample = activated[0]
    end_sample = activated[-1]
    
    # Convert to time relative to original trace start
    dt = 1.0 / metadata['resampled_rate']
    window_start_offset = metadata['window_start']
    
    start_time = window_start_offset + start_sample * dt
    end_time = window_start_offset + end_sample * dt
    
    return start_time, end_time


def visualize_sac_prediction(
    sac_path: str,
    onset_time: float,
    prediction: np.ndarray,
    save_path: Optional[str] = None
):
    """Visualize SAC waveform with predicted arrival window.
    
    Args:
        sac_path: Path to SAC file
        onset_time: Onset time used for window extraction
        prediction: Model prediction mask
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Load waveform
    waveform, metadata = load_sac_waveform(sac_path, onset_time)
    
    # Extract predicted window
    pred_start, pred_end = extract_predicted_window(prediction, metadata)
    
    # Create time axis
    n_samples = waveform.shape[1]
    dt = 1.0 / metadata['resampled_rate']
    time_axis = np.arange(n_samples) * dt + metadata['window_start']
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Waveform
    ax1.plot(time_axis, waveform[0, :, 0], 'k-', linewidth=0.5)
    ax1.axvline(onset_time, color='r', linestyle='--', label='Onset time', alpha=0.7)
    if pred_start is not None:
        ax1.axvspan(pred_start, pred_end, alpha=0.3, color='green', label='Predicted window')
    ax1.set_ylabel('Amplitude (normalized)')
    ax1.set_title(f"{metadata['network']}.{metadata['station']}.{metadata['channel']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prediction mask
    pred_flat = prediction.squeeze()
    pred_time = np.arange(len(pred_flat)) * dt + metadata['window_start']
    ax2.plot(pred_time, pred_flat, 'b-', linewidth=1)
    ax2.axhline(0.5, color='r', linestyle='--', label='Threshold', alpha=0.5)
    ax2.fill_between(pred_time, 0, pred_flat, alpha=0.3)
    ax2.set_ylabel('Prediction')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

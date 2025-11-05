"""Data preprocessing utilities"""
import numpy as np
import h5py
import torch


def load_h5_data(file_path):
    """Load waveforms and labels from HDF5 file.
    
    Args:
        file_path: Path to HDF5 file with 'waveforms' and 'labels' datasets
        
    Returns:
        waveforms: numpy array of shape (N, T, C)
        labels: numpy array of shape (N, 2) with start/end indices
    """
    with h5py.File(file_path, 'r') as f:
        waveforms = f['waveforms'][:]
        labels = f['labels'][:]
    return waveforms, labels


def extract_windows_from_masks(masks, threshold=0.5, remove_padding_offset=False, padding_offset=20):
    """Extract window boundaries from binary masks.
    
    Args:
        masks: Binary masks of shape (N, 1, T) or (N, T)
        threshold: Threshold for binarization
        remove_padding_offset: If True, subtract padding offset from indices
        padding_offset: Padding offset in samples (default 20 = 0.5s)
    
    Returns:
        windows: numpy array of shape (N, 2) with start/end indices
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    if len(masks.shape) == 3:
        masks = masks.squeeze(1)
    
    windows = []
    for mask in masks:
        # Find the highest peak
        peak_idx = np.argmax(mask)
        peak_value = mask[peak_idx]
        
        # If peak is below threshold, no valid window
        if peak_value <= threshold:
            windows.append([0, 0])
            continue
        
        # Expand left from peak until value drops below threshold
        start = peak_idx
        while start > 0 and mask[start - 1] > threshold:
            start -= 1
        
        # Expand right from peak until value drops below threshold
        end = peak_idx
        while end < len(mask) - 1 and mask[end + 1] > threshold:
            end += 1
        
        # Optionally remove padding offset to get indices relative to original signal
        if remove_padding_offset:
            start = max(0, start - padding_offset)
            end = max(0, end - padding_offset)
        
        windows.append([start, end])
    
    return np.array(windows)


__all__ = [
    "load_h5_data",
    "extract_windows_from_masks",
]

#!/usr/bin/env python
"""
Simple prediction script for Docker usage with optional visualization.

Usage:
    # Basic prediction
    docker run --rm --gpus all \
      -v /path/to/sac/files:/data \
      pawlib:latest python examples/docker_predict.py /data/signal.sac 10.5
      
    # With figure output
    docker run --rm --gpus all \
      -v /path/to/sac/files:/data \
      -v /path/to/output:/output \
      pawlib:latest python examples/docker_predict.py /data/signal.sac 10.5 /output/result.png
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
from pawlib import (
    PAW,
    load_sac_waveform,
    preprocess_for_paw,
    extract_windows_from_prediction
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python docker_predict.py <sac_file> [onset_time] [output_figure.png]")
        print("Example: python docker_predict.py /data/signal.sac 10.5 /output/result.png")
        sys.exit(1)
    
    sac_file = sys.argv[1]
    onset_time = float(sys.argv[2]) if len(sys.argv) > 2 else None
    output_figure = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"Loading SAC file: {sac_file}")
    
    # Load SAC file
    waveform, meta = load_sac_waveform(
        sac_file,
        onset_time=onset_time,
        window_duration=5.0,
        padding=0.5
    )
    
    print(f"  Sampling rate: {meta['sampling_rate']} Hz")
    print(f"  Onset time: {meta['onset_time']:.3f}s")
    print(f"  Waveform shape: {waveform.shape}")
    
    # Preprocess
    print("\nPreprocessing...")
    waveform = preprocess_for_paw(
        waveform,
        meta['sampling_rate'],
        target_freq=40.0,
        verbose=True
    )
    
    # Add model padding
    waveform = np.pad(waveform, ((0, 0), (20, 20), (0, 0)), mode='constant')
    
    # Load pretrained model from HuggingFace
    print("\nLoading pretrained model from HuggingFace...")
    model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')
    
    # Run prediction
    print("\nRunning inference...")
    windows = model.predict_windows(waveform[:, 20:-20, :])
    prediction = model.predict(waveform[:, 20:-20, :], return_windows=False)
    
    # Convert to time
    window_start_time = windows[0, 0] / 40.0
    window_end_time = windows[0, 1] / 40.0
    window_duration = window_end_time - window_start_time
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Detected window (samples): [{windows[0, 0]}, {windows[0, 1]}]")
    print(f"Detected window (time):    [{window_start_time:.3f}s, {window_end_time:.3f}s]")
    print(f"Window duration:           {window_duration:.3f}s")
    print(f"Max prediction value:      {prediction.max():.3f}")
    print("="*50)
    
    # Create visualization if output figure path provided
    if output_figure:
        print(f"\nCreating visualization: {output_figure}")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # Time axis (centered at 0 for the 5s window)
        time = (np.arange(waveform.shape[1]) / 40.0) - 0.5  # Center at 0
        
        # Plot waveform with detected window
        ax1.plot(time, waveform[0, :, 0], 'k-', linewidth=0.5)
        ax1.axvspan(window_start_time - 0.5, window_end_time - 0.5, 
                    alpha=0.2, color='yellow', 
                    label=f'Amplitude Window ({window_duration:.2f}s)', zorder=0)
        ax1.axvline(0, color='r', linestyle='--', label='Window Center', alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Waveform: {meta["station"]}.{meta["channel"]} (onset={meta["onset_time"]:.1f}s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot prediction
        ax2.plot(time, prediction[0, 0, :], 'b-', linewidth=1.5, label='PAW Prediction')
        ax2.axvspan(window_start_time - 0.5, window_end_time - 0.5, 
                    alpha=0.2, color='yellow', zorder=0)
        ax2.axvline(0, color='r', linestyle='--', label='Window Center', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Probability')
        ax2.set_title('Phase Arrival Prediction')
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_figure, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_figure}")
        plt.close()


if __name__ == "__main__":
    main()

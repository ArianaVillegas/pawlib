#!/usr/bin/env python
"""
Batch prediction script for Docker usage.

Usage:
    docker run --rm --gpus all \
      -v /path/to/sac/files:/data \
      -v /path/to/output:/output \
      pawlib:latest python examples/docker_batch_predict.py /data /output/results.csv
"""

import sys
import csv
from pathlib import Path
import numpy as np
from pawlib import (
    PAW,
    load_sac_waveform,
    preprocess_for_paw,
    extract_windows_from_prediction
)


def main():
    if len(sys.argv) < 3:
        print("Usage: python docker_batch_predict.py <sac_directory> <output_csv>")
        print("Example: python docker_batch_predict.py /data /output/results.csv")
        sys.exit(1)
    
    sac_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])
    
    # Find all SAC files
    sac_files = sorted(sac_dir.glob("*.sac"))
    if not sac_files:
        print(f"No SAC files found in {sac_dir}")
        sys.exit(1)
    
    print(f"Found {len(sac_files)} SAC files")
    
    # Load pretrained model
    print("Loading pretrained model from HuggingFace...")
    model = PAW.from_pretrained('hf://suroRitch/pawlib-pretrained/paw_corrected.pt')
    
    # Process files
    results = []
    for i, sac_file in enumerate(sac_files, 1):
        print(f"\n[{i}/{len(sac_files)}] Processing: {sac_file.name}")
        
        try:
            # Load and preprocess
            waveform, meta = load_sac_waveform(
                str(sac_file),
                onset_time=None,  # Will try to read from SAC header
                window_duration=5.0,
                padding=0.5
            )
            
            waveform = preprocess_for_paw(
                waveform,
                meta['sampling_rate'],
                target_freq=40.0
            )
            
            # Add model padding
            waveform = np.pad(waveform, ((0, 0), (20, 20), (0, 0)), mode='constant')
            
            # Predict
            prediction = model.predict(waveform[:, 20:-20, :])
            windows = extract_windows_from_prediction(prediction)
            
            # Convert to time
            window_start = windows[0, 0] / 40.0
            window_end = windows[0, 1] / 40.0
            duration = window_end - window_start
            max_prob = prediction.max()
            
            results.append({
                'filename': sac_file.name,
                'station': meta['station'],
                'channel': meta['channel'],
                'onset_time': meta['onset_time'],
                'window_start': window_start,
                'window_end': window_end,
                'duration': duration,
                'max_probability': max_prob,
                'status': 'success'
            })
            
            print(f"  ✓ Detected window: [{window_start:.3f}s, {window_end:.3f}s] (duration: {duration:.3f}s)")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'filename': sac_file.name,
                'station': '',
                'channel': '',
                'onset_time': '',
                'window_start': '',
                'window_end': '',
                'duration': '',
                'max_probability': '',
                'status': f'error: {e}'
            })
    
    # Save results to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "="*60)
    print(f"BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Total files: {len(sac_files)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] != 'success')}")
    print(f"Results saved to: {output_csv}")
    print("="*60)


if __name__ == "__main__":
    main()

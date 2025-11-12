#!/usr/bin/env python
"""
Download sample SAC files for testing PAWlib.

Usage:
    python examples/download_sample_sac.py [output_directory]
"""

import sys
from pathlib import Path

try:
    from obspy import read
    from obspy.core.util import get_example_file
except ImportError:
    print("Error: ObsPy is required. Install with: pip install obspy")
    sys.exit(1)


def download_samples(output_dir="./data/sample_sac"):
    """Download real seismograph SAC files from ObsPy examples."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading real seismograph SAC files to: {output_path.absolute()}")
    print("="*60)
    
    # Real seismograph examples from ObsPy
    examples = [
        ('https://examples.obspy.org/COP.BHZ.DK.2009.050', 'COP_BHZ_DK.sac', 200.0),
        ('https://examples.obspy.org/bw.furt.__.ehz.d.2010.147.a.slist.gz', 'BW_FURT_EHZ.sac', 50.0),
        ('https://examples.obspy.org/loc_RJOB20050831023349.z', 'RJOB_vertical.sac', 10.0),
    ]
    
    downloaded = []
    for url, filename, onset_time in examples:
        try:
            stream = read(url)
            output_file = output_path / filename
            stream.write(str(output_file), format='SAC')
            
            trace = stream[0]
            print(f"\n✓ Downloaded: {filename}")
            print(f"  Station: {trace.stats.network}.{trace.stats.station}.{trace.stats.channel}")
            print(f"  Sampling rate: {trace.stats.sampling_rate} Hz")
            print(f"  Duration: {len(trace.data) / trace.stats.sampling_rate:.1f} seconds")
            print(f"  Suggested onset: {onset_time}s")
            
            downloaded.append((filename, onset_time))
            
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
    
    print("\n" + "="*60)
    print(f"Sample files saved to: {output_path.absolute()}")
    
    if downloaded:
        print("\nTest with these commands:")
        for fname, onset in downloaded[:2]:
            print(f"\n  # {fname}")
            print(f"  python examples/docker_predict.py {output_path.absolute()}/{fname} {onset}")
        
        print("\nOr with Podman:")
        fname, onset = downloaded[0]
        print(f"  podman run --rm --device nvidia.com/gpu=all \\")
        print(f"    -v {output_path.absolute()}:/data:Z \\")
        print(f"    pawlib:latest python examples/docker_predict.py /data/{fname} {onset}")
    
    print("="*60)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/sample_sac"
    download_samples(output_dir)

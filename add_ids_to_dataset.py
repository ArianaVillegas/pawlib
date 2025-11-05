#!/usr/bin/env python
"""Add ARID values to dataset.h5 to enable subset testing."""

import h5py
import numpy as np
from pathlib import Path

print("\n" + "=" * 70)
print("Adding ARIDs to dataset.h5")
print("=" * 70 + "\n")

# Paths
dataset_path = Path('../datasets/dataset.h5')
source_dir = Path('/media/mueenlab/extradrive1/ariana/data/filtered')

# Check if source directory exists
if not source_dir.exists():
    print(f"❌ Source directory not found: {source_dir}")
    print("   Cannot load ARIDs from source files.")
    print("   Please update source_dir path in this script.")
    exit(1)

# Load ARIDs from source .npy files (same order as main.py)
print("Loading ARIDs from source files...")

station_names_3c = ['BOSA', 'CPUP', 'DBIC', 'LBTB', 'LPAZ', 'PLCA', 'VNDA']
station_names_arr = ['ASAR', 'BRTR', 'CMAR', 'ILAR', 'KSRS', 'MKAR', 'PDAR', 'TXAR']
stas = station_names_3c + station_names_arr

ids_list = []
version = '_5'  # Update this to match your data version

for sta in stas:
    sta_lower = sta.lower()
    arid_file = source_dir / f'{sta_lower}_arids{version}.npy'
    
    if arid_file.exists():
        station_ids = np.load(arid_file, allow_pickle=True)
        ids_list.append(station_ids)
        print(f"  ✓ Loaded {len(station_ids)} ARIDs from {sta}")
    else:
        print(f"  ⚠️  File not found: {arid_file}")

if not ids_list:
    print("\n❌ No ARID files found!")
    print("   Check that:")
    print(f"   1. Source directory is correct: {source_dir}")
    print(f"   2. Version suffix is correct: {version}")
    print("   3. Files exist like: bosa_arids_5.npy")
    exit(1)

# Concatenate all ARIDs
all_ids = np.concatenate(ids_list)
print(f"\nTotal ARIDs loaded: {len(all_ids)}")

# Check dataset size
with h5py.File(dataset_path, 'r') as f:
    n_samples = len(f['waveforms'])
    print(f"Dataset size: {n_samples} samples")

if len(all_ids) != n_samples:
    print(f"\n❌ ERROR: ARID count ({len(all_ids)}) != dataset size ({n_samples})")
    print("   The dataset.h5 file may have been created differently.")
    print("   Cannot safely add IDs.")
    exit(1)

# Add IDs to dataset
print(f"\nAdding {len(all_ids)} ARIDs to {dataset_path}...")

with h5py.File(dataset_path, 'a') as f:
    if 'ids' in f:
        print("  ⚠️  'ids' field already exists, deleting old version...")
        del f['ids']
    
    f.create_dataset('ids', data=all_ids, compression='gzip')
    print("  ✓ Successfully added 'ids' dataset")

# Verify
with h5py.File(dataset_path, 'r') as f:
    stored_ids = f['ids'][:]
    print(f"\nVerification:")
    print(f"  - Stored {len(stored_ids)} IDs")
    print(f"  - Sample IDs: {stored_ids[:5]}")
    print(f"  - ID range: {stored_ids.min()} to {stored_ids.max()}")

print("\n" + "=" * 70)
print("✅ SUCCESS! IDs added to dataset.h5")
print("=" * 70)
print("\nYou can now run subset testing:")
print("  python pawlib/train_and_eval.py")
print()

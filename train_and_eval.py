#!/usr/bin/env python
"""PAWlib Training and Evaluation Script"""
import sys
sys.path.insert(0, '/home/ariana/AFRL/Seismology/pawlib')

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from pawlib import PAW
from pawlib.preprocessing import extract_windows_from_masks

# Setup paths
is_in_pawlib = Path.cwd().name == 'pawlib'
base = Path('..') if is_in_pawlib else Path('.')
dataset_path = base / 'datasets/dataset_with_idx.h5'
metadata_overlap = (base / '../data' if is_in_pawlib else base / '../data') / 'AFTAC_overlap_v5.csv'
metadata_eval = (base / '../data' if is_in_pawlib else base / '../data') / 'AFTAC_eval_v5.csv'
checkpoint_dir = 'checkpoints_fixed' if is_in_pawlib else 'pawlib/checkpoints_fixed'
output_dir = Path('outputs' if is_in_pawlib else 'pawlib/outputs')
output_dir.mkdir(exist_ok=True)

print(f"\n{'='*70}\nPAW Training & Evaluation\n{'='*70}")
print(f"Dataset: {dataset_path}\nMetadata overlap: {metadata_overlap}\nMetadata eval: {metadata_eval}\n")

# Load full dataset first
print("Loading dataset...")
with h5py.File(dataset_path, 'r') as f:
    waveforms, labels = f['waveforms'][:], f['labels'][:]
    ids = f.get('ids', f.get('arids', np.arange(len(waveforms))))[:]
    has_ids = 'ids' in f or 'arids' in f

if not has_ids:
    print("⚠️ WARNING: No ARIDs found. Training on all data.\n")

# Filter training data if we have metadata
train_waveforms, train_labels = waveforms, labels
if metadata_overlap.exists() and metadata_eval.exists() and has_ids:
    print("Filtering training data to: adjusted + only_eval (excluding accepted)...")
    
    # Load and filter metadata
    df_overlap = pd.read_csv(metadata_overlap)
    df_overlap = df_overlap[
        (df_overlap.AMPWIN_STTIME_eval <= df_overlap.ARRTIME_eval) &
        ((df_overlap.AMPTIME_eval + df_overlap.PER_eval/2) <= (df_overlap.ARRTIME_eval + 5)) &
        (df_overlap.PER_eval != 0) & (df_overlap.AMP_eval != 0) & 
        (df_overlap.PER_det != 0) & (df_overlap.AMP_det != 0)
    ]
    df_overlap.index = df_overlap['ARID']
    
    df_eval = pd.read_csv(metadata_eval)
    df_eval = df_eval[
        (df_eval.AMPWIN_STTIME <= df_eval.ARRTIME) &
        ((df_eval.AMPTIME + df_eval.PER/2) <= (df_eval.ARRTIME + 5)) &
        (df_eval.PER != 0) & (df_eval.AMP != 0)
    ]
    df_eval.index = df_eval['ARID']
    
    # Define subsets
    df_adj = df_overlap[
        (df_overlap.ARRTIME_eval != df_overlap.ARRTIME_det) | 
        (df_overlap.AMPTIME_eval != df_overlap.AMPTIME_det) |
        (df_overlap.PER_eval != df_overlap.PER_det)
    ]
    df_eval_only = df_eval.drop(df_overlap.index, errors='ignore')
    
    # Training set = adjusted + only_eval
    train_arids = np.concatenate([df_adj['ARID'].values, df_eval_only['ARID'].values])
    train_mask = np.isin(ids, train_arids)
    train_waveforms = waveforms[train_mask]
    train_labels = labels[train_mask]
    
    print(f"  Adjusted samples: {len(df_adj)}")
    print(f"  Only-eval samples: {len(df_eval_only)}")
    print(f"  Total training samples: {len(train_waveforms)}")
    print(f"  Excluded (accepted): {len(df_overlap) - len(df_adj)}\n")
else:
    print(f"Training on all {len(waveforms)} samples...\n")

model = PAW()

history = model.train(
    data=train_waveforms,
    labels=train_labels,
    epochs=200,
    batch_size=64,
    loss='dice',
    checkpoint_dir=checkpoint_dir,
    save_best=True,
    verbose=True,
    limit_batches=0.2
)

# Testing on full dataset
print(f"\n{'='*70}\nTesting on Full Dataset\n{'='*70}\n")

# Test on full dataset
results = model.test(waveforms, labels, batch_size=64)
model.print_results(results, title="Full Dataset (All 80K samples)")

# Subset testing
if metadata_overlap.exists() and metadata_eval.exists() and has_ids:
    print(f"\n{'='*70}\nSubset Testing\n{'='*70}\n")
    
    # Reload metadata (already filtered above during training)
    df_overlap = pd.read_csv(metadata_overlap)
    df_overlap = df_overlap[
        (df_overlap.AMPWIN_STTIME_eval <= df_overlap.ARRTIME_eval) &
        ((df_overlap.AMPTIME_eval + df_overlap.PER_eval/2) <= (df_overlap.ARRTIME_eval + 5)) &
        (df_overlap.PER_eval != 0) & (df_overlap.AMP_eval != 0) & 
        (df_overlap.PER_det != 0) & (df_overlap.AMP_det != 0)
    ]
    df_overlap.index = df_overlap['ARID']
    
    df_eval = pd.read_csv(metadata_eval)
    df_eval = df_eval[
        (df_eval.AMPWIN_STTIME <= df_eval.ARRTIME) &
        ((df_eval.AMPTIME + df_eval.PER/2) <= (df_eval.ARRTIME + 5)) &
        (df_eval.PER != 0) & (df_eval.AMP != 0)
    ]
    df_eval.index = df_eval['ARID']
    
    id_to_idx = {v: i for i, v in enumerate(ids)}
    print(f"Overlap: {len(df_overlap)} | Eval: {len(df_eval)} | Dataset: {len(ids)}\n")
    
    def test_subset(name, mask_or_ids):
        """Helper to test a subset and add to results."""
        indices = [id_to_idx[i] for i in mask_or_ids if i in id_to_idx]
        if indices:
            res = model.test(waveforms[indices], labels[indices], batch_size=64)
            print(f"{name:15s} ({len(indices):5d}): {res['window_accuracy']:.2%}")
            return res
        return None
    
    # Define subsets matching main.py exactly
    df_adj = df_overlap[
        (df_overlap.ARRTIME_eval != df_overlap.ARRTIME_det) | 
        (df_overlap.AMPTIME_eval != df_overlap.AMPTIME_det) |
        (df_overlap.PER_eval != df_overlap.PER_det)
    ]
    df_acc = df_overlap.drop(df_adj.index)
    df_eval_only = df_eval.drop(df_overlap.index, errors='ignore')
    df_eval_fil = df_eval.drop(df_adj.index, errors='ignore')
    
    # Define subsets (3c and array stations are subsets of eval_only)
    eval_arids = df_eval_fil['ARID'].values if len(df_eval_fil) > 0 else []
    eval_only_arids = df_eval_only['ARID'].values if len(df_eval_only) > 0 else []
    subsets = {
        'eval': eval_arids,
        'only_eval': eval_only_arids,  # Same as eval, kept for compatibility
        'adjusted': df_adj['ARID'].values,
        'accepted': df_acc['ARID'].values,
    }
    
    # Add station-based subsets if we have eval data
    if len(df_eval_only) > 0 and 'STA' in df_eval_only.columns:
        sta_3c = ['BOSA', 'CPUP', 'DBIC', 'LBTB', 'LPAZ', 'PLCA', 'VNDA']
        sta_arr = ['ASAR', 'BRTR', 'CMAR', 'ILAR', 'KSRS', 'MKAR', 'PDAR', 'TXAR']
        subsets['3c_sta'] = df_eval_only[df_eval_only['STA'].isin(sta_3c)]['ARID'].values
        subsets['arr_sta'] = df_eval_only[df_eval_only['STA'].isin(sta_arr)]['ARID'].values
    
    subset_results = {k: v for k, v in ((k, test_subset(k, v)) for k, v in subsets.items()) if v}
    
    if subset_results:
        print(f"\n{'='*70}\nSubset Comparison\n{'='*70}\n")
        model.print_results({'full_dataset': results, **subset_results}, title="All Subsets")
else:
    if not metadata_overlap.exists() or not metadata_eval.exists():
        print(f"\n⚠️  Metadata not found")
    elif not has_ids:
        print("\n⚠️  No ARIDs - subset testing skipped")

# Visualizations
print(f"\n{'='*70}\nGenerating Visualizations\n{'='*70}\n")

def visualize_predictions(waveforms, labels, model, output_dir, n_correct=5, n_incorrect=5):
    """Generate visualizations of predictions."""
    sample_size = min(1000, len(waveforms))
    idx = np.random.choice(len(waveforms), sample_size, replace=False)
    
    print(f"Predicting on {sample_size} samples...")
    predictions = model.predict(waveforms[idx])
    
    # Pad waveforms to match predictions (240 samples)
    pad = 20
    n, t, c = waveforms[idx].shape
    wf_padded = np.concatenate([np.zeros((n,pad,c)), waveforms[idx], np.zeros((n,pad,c))], axis=1)
    
    pred_win = extract_windows_from_masks(predictions, threshold=0.5)
    true_win = np.round(labels[idx] / 0.025).astype(int) + 20
    diffs = np.abs(pred_win - true_win).sum(axis=1)
    correct = np.where(diffs <= 4)[0]
    incorrect = np.where(diffs > 4)[0]
    
    print(f"Found {len(correct)} correct, {len(incorrect)} incorrect")
    
    def plot_examples(indices, name, color):
        if len(indices) == 0: return
        n_show = min(len(indices), n_correct if name=='correct' else n_incorrect)
        examples = np.random.choice(indices, n_show, replace=False)
        
        fig, axes = plt.subplots(n_show, 1, figsize=(12, 3*n_show))
        if n_show == 1: axes = [axes]
        
        for i, ex in enumerate(examples):
            wf, pred = wf_padded[ex].squeeze(), predictions[ex].squeeze()
            ts, te = true_win[ex]
            ps, pe = pred_win[ex]
            time = np.arange(len(wf)) * 0.025
            
            ax = axes[i]
            ax.plot(time, wf, 'k-', lw=0.8, alpha=0.7)
            ax.axvspan(ts*0.025, te*0.025, alpha=0.2, color='green')
            ax.axvspan(ps*0.025, pe*0.025, alpha=0.2, color=color)
            
            ax2 = ax.twinx()
            ax2.plot(time, pred, f'{color[0]}-', lw=1.5, alpha=0.6)
            ax2.set_ylim(-0.1, 1.1)
            
            symbol = '✓' if name=='correct' else '✗'
            ax.set_title(f"{symbol} {name.upper()} - Sample {idx[ex]} (diff={diffs[ex]:.0f})")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = output_dir / f'{name}_predictions.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {name}: {path}")
    
    plot_examples(correct, 'correct', 'blue')
    plot_examples(incorrect, 'incorrect', 'red')
    return len(correct), len(incorrect), sample_size

n_cor, n_inc, tot = visualize_predictions(waveforms, labels, model, output_dir)
print(f"\nAnalyzed {tot} samples: {n_cor} correct ({n_cor/tot*100:.1f}%), {n_inc} incorrect\n")

# Summary
print(f"\n{'='*70}\nSummary\n{'='*70}")
print(f"Epochs: {len(history['train_loss'])} | Loss: {history['train_loss'][-1]:.4f} → {min(history['val_loss']):.4f}")
print(f"Full Dataset: {results['window_accuracy']:.2%} accuracy, {results['dice_score']:.2%} dice")

if 'subset_results' in locals() and subset_results:
    best = max(subset_results.items(), key=lambda x: x[1]['window_accuracy'])
    worst = min(subset_results.items(), key=lambda x: x[1]['window_accuracy'])
    print(f"Best: {best[0]} ({best[1]['window_accuracy']:.2%}) | Worst: {worst[0]} ({worst[1]['window_accuracy']:.2%})")

model.save(str(Path(checkpoint_dir)/'paw_corrected.pt'), metadata={'padding':20, 'freq':0.025})
print(f"\nModel saved to: {Path(checkpoint_dir).absolute()}/paw_corrected.pt")
print(f"{'='*70}\n")

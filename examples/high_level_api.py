#!/usr/bin/env python
"""
Example: High-Level PAW API
============================
"""

import numpy as np
import sys
from pathlib import Path

# Add pawlib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pawlib import PAW


def example_basic_usage():
    """Basic training and testing example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create model with custom config
    model = PAW(config={
        'n_cnn': 5,
        'n_lstm': 1,
        'n_transformer': 1,
        'drop_rate': 0.4
    })
    
    print(f"Model created on {model.device}")
    print(f"Config: {model.config}")
    
    # Generate synthetic data for demo
    np.random.seed(42)
    n_samples = 1000
    seq_len = 200
    
    # Synthetic waveforms
    data = np.random.randn(n_samples, seq_len, 1).astype(np.float32)
    
    # Synthetic labels (start, end indices)
    labels = np.zeros((n_samples, 2))
    for i in range(n_samples):
        start = np.random.randint(40, 80)
        length = np.random.randint(30, 60)
        labels[i] = [start, min(start + length, seq_len-1)]
    
    print(f"\nData shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Train model
    print("\nTraining model...")
    history = model.train(
        data=data,
        labels=labels,
        epochs=5,
        batch_size=32,
        loss='dice',
        lr=1e-3,
        val_split=0.2,
        verbose=True
    )
    
    # Test model
    print("\nTesting model...")
    results = model.test(data[:100], labels[:100], batch_size=32)
    
    # Print results
    model.print_results(results, title="Test Results")
    
    return model


def example_h5_training():
    """Example with HDF5 file (if available)."""
    print("\n" + "=" * 60)
    print("Example 2: Training from HDF5 File")
    print("=" * 60)
    
    h5_path = "../datasets/dataset.h5"
    
    try:
        # Create model
        model = PAW()
        
        # Train directly from H5 file
        print(f"Training from {h5_path}...")
        model.train(
            data=h5_path,
            epochs=100,
            batch_size=64,
            loss='dice',
            checkpoint_dir='checkpoints_demo',
            verbose=True
        )
        
        # Test
        results = model.test(h5_path, batch_size=64)
        model.print_results(results)
        
        # Save model
        model.save("checkpoints_demo/paw_demo.pt")
        
        # Load model
        model2 = PAW.from_pretrained("checkpoints_demo/paw_demo.pt")
        
        return model
        
    except FileNotFoundError:
        print(f"H5 file not found at {h5_path}")
        print("Skipping this example.")
        return None


def example_different_losses():
    """Example showing different loss functions."""
    print("\n" + "=" * 60)
    print("Example 3: Different Loss Functions")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    data = np.random.randn(500, 200, 1).astype(np.float32)
    labels = np.array([[50, 100] for _ in range(500)])
    
    losses_to_try = ['dice', 'bce', 'bce+dice']
    
    for loss_name in losses_to_try:
        print(f"\nTraining with {loss_name} loss...")
        
        model = PAW()
        model.train(
            data=data,
            labels=labels,
            epochs=3,
            batch_size=32,
            loss=loss_name,
            verbose=False
        )
        
        results = model.test(data[:100], labels[:100])
        print(f"{loss_name:12s} - Accuracy: {results['window_accuracy']:.4f}, "
              f"Dice: {results['dice_score']:.4f}")


def example_prediction():
    """Example of making predictions."""
    print("\n" + "=" * 60)
    print("Example 4: Making Predictions")
    print("=" * 60)
    
    # Train a simple model
    np.random.seed(42)
    data = np.random.randn(500, 200, 1).astype(np.float32)
    labels = np.array([[50, 100] for _ in range(500)])
    
    model = PAW()
    model.train(data, labels, epochs=2, verbose=False)
    
    # Make predictions on new data
    new_data = np.random.randn(10, 200, 1).astype(np.float32)
    predictions = model.predict(new_data)
    
    print(f"Input shape: {new_data.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Binary predictions
    binary_preds = (predictions > 0.5).astype(int)
    print(f"Number of positive predictions: {binary_preds.sum()}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "    PAWLib High-Level API Examples    ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # Run examples
    example_basic_usage()
    example_h5_training()
    example_different_losses()
    example_prediction()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nThe high-level API provides:")
    print("  ✓ Simple model.train() and model.test()")
    print("  ✓ Multiple loss functions")
    print("  ✓ Automatic data loading from H5 files")
    print("  ✓ Comprehensive metrics")
    print("  ✓ Easy save/load")
    print("  ✓ Pretty printing")
    print("\nReady for production use! 🚀\n")


if __name__ == "__main__":
    main()

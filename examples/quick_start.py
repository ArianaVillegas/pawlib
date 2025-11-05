#!/usr/bin/env python
"""
Quick Start Guide for pawlib
=============================

This example demonstrates the basic usage of pawlib for seismic waveform analysis.
Run this from the project root: python -m pawlib.examples.quick_start
"""

import numpy as np
import sys
from pathlib import Path

# Add pawlib to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from pawlib import PAW


def example_1_basic_training():
    """Example 1: Train a model from scratch."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Training")
    print("=" * 60 + "\n")
    
    # Create model
    model = PAW()
    print(f"✓ Model created on {model.device}")
    
    # Generate synthetic data for demo (200 samples, 200 timesteps)
    np.random.seed(42)
    n_samples = 1000
    data = np.random.randn(n_samples, 200, 1).astype(np.float32)
    
    # Generate synthetic labels (start, end) in seconds
    labels = np.random.rand(n_samples, 2).astype(np.float32) * 4.0
    labels[:, 1] = labels[:, 0] + 0.5  # end = start + 0.5s
    
    print(f"✓ Created synthetic dataset: {data.shape}")
    print(f"  Labels range: [{labels.min():.2f}, {labels.max():.2f}] seconds\n")
    
    # Train model
    print("Training for 10 epochs...")
    history = model.train(
        data=data,
        labels=labels,
        epochs=10,
        batch_size=32,
        loss='dice',
        verbose=True
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")


def example_2_inference():
    """Example 2: Load model and run inference."""
    print("\n" + "=" * 60)
    print("Example 2: Model Inference")
    print("=" * 60 + "\n")
    
    model = PAW()
    
    # Create test data (10 samples)
    test_data = np.random.randn(10, 200, 1).astype(np.float32)
    
    # Run prediction
    predictions = model.predict(test_data)
    
    print(f"✓ Predictions generated: {predictions.shape}")
    print(f"  Value range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Mean confidence: {predictions.mean():.3f}")


def example_3_evaluation():
    """Example 3: Evaluate model on test data."""
    print("\n" + "=" * 60)
    print("Example 3: Model Evaluation")
    print("=" * 60 + "\n")
    
    model = PAW()
    
    # Generate test data
    n_test = 500
    test_data = np.random.randn(n_test, 200, 1).astype(np.float32)
    test_labels = np.random.rand(n_test, 2).astype(np.float32) * 4.0
    test_labels[:, 1] = test_labels[:, 0] + 0.5
    
    # Evaluate
    results = model.test(test_data, test_labels, batch_size=32)
    
    print("✓ Evaluation results:")
    print(f"  Window Accuracy: {results['window_accuracy']:.2%}")
    print(f"  Dice Score: {results['dice_score']:.2%}")
    print(f"  Amplitude RMSE: {results['amplitude_rmse']:.4f}")
    print(f"  Period RMSE: {results['period_rmse']:.4f}")


def example_4_save_load():
    """Example 4: Save and load models."""
    print("\n" + "=" * 60)
    print("Example 4: Save and Load Model")
    print("=" * 60 + "\n")
    
    # Train a small model
    model = PAW()
    data = np.random.randn(100, 200, 1).astype(np.float32)
    labels = np.random.rand(100, 2).astype(np.float32) * 4.0
    labels[:, 1] = labels[:, 0] + 0.5
    
    print("Training model...")
    model.train(data, labels, epochs=5, verbose=False)
    
    # Save model
    save_path = '/tmp/paw_model.pt'
    model.save(save_path, metadata={'version': '1.0', 'date': '2024'})
    print(f"✓ Model saved to {save_path}")
    
    # Load model
    loaded_model = PAW.load(save_path)
    print(f"✓ Model loaded successfully")
    
    # Verify predictions match
    test_data = np.random.randn(5, 200, 1).astype(np.float32)
    pred_original = model.predict(test_data)
    pred_loaded = loaded_model.predict(test_data)
    
    diff = np.abs(pred_original - pred_loaded).max()
    print(f"✓ Prediction difference: {diff:.10f} (should be ~0)")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PAWLIB QUICK START GUIDE")
    print("=" * 60)
    
    # Run examples
    example_1_basic_training()
    example_2_inference()
    example_3_evaluation()
    example_4_save_load()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60 + "\n")
    
    print("Next steps:")
    print("  • See high_level_api.py for advanced usage")
    print("  • Check ../train_and_eval.py for real dataset training")
    print("  • Read ../README.md for full documentation")


if __name__ == '__main__':
    main()

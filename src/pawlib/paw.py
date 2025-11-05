from pathlib import Path
from typing import Union, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from .architectures import build_paw_copy
from .config import PAWReferenceConfig
from .losses import DiceLoss, AmpPerLoss, AmpPerLossWdW, BCEDiceLoss
from .metrics import (
    WindowAccuracy,
    AmplitudeRMSE,
    PeriodRMSE,
    MagnitudeRMSE,
    WindowRMSE,
    DiceScore
)
from .preprocessing import load_h5_data, extract_windows_from_masks
from .checkpointing import save_checkpoint, load_checkpoint
from .utils import print_metrics, print_subset_metrics


class PAW:
    """High-level PAW model interface for seismic phase detection.
    
    Simple sklearn/HuggingFace-style API:
    
    Example:
        >>> model = PAW(config={'n_cnn': 5, 'n_lstm': 1})
        >>> model.train(data, labels, epochs=50, loss='dice')
        >>> results = model.test(test_data, test_labels)
        >>> model.print_results(results)
    """
    
    def __init__(self, config: Optional[Dict] = None, device: Optional[str] = None):
        """Initialize PAW model.
        
        Args:
            config: Model configuration dict (optional, uses defaults if None)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Build model
        if config is None:
            self.config = PAWReferenceConfig()
        elif isinstance(config, dict):
            self.config = PAWReferenceConfig(**config)
        else:
            self.config = config
        
        self.model = build_paw_copy(config=self.config, device=str(self.device))
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train(
        self,
        data: Union[str, np.ndarray, torch.Tensor, Dataset],
        labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        loss: str = 'dice',
        lr: float = 1e-3,
        val_split: float = 0.2,
        checkpoint_dir: Optional[str] = None,
        save_best: bool = True,
        verbose: bool = True,
        limit_batches: Optional[float] = None
    ) -> Dict:
        """Train the PAW model.
        
        Args:
            data: Training data. Can be:
                - Path to HDF5 file
                - numpy array of shape (N, T, C)
                - torch.Tensor
                - torch Dataset
            labels: Training labels (N, 2) with start/end indices.
                Required if data is not an HDF5 path or Dataset.
            epochs: Number of training epochs
            batch_size: Batch size
            loss: Loss function name ('dice', 'bce', 'amper', 'amperwdw', 'bce+dice')
            lr: Learning rate
            val_split: Validation split ratio
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            verbose: Print training progress
            limit_batches: Fraction of batches to use per epoch (0.2 = 20%, like main.py)
            
        Returns:
            dict with training history
        """
        if verbose:
            print(f"Training PAW model on {self.device}")
            batch_limit_str = f", Batch limit: {limit_batches*100:.0f}%" if limit_batches else ""
            print(f"Epochs: {epochs}, Batch size: {batch_size}, Loss: {loss}{batch_limit_str}")
        
        # Load/prepare data
        train_loader, val_loader = self._prepare_dataloaders(
            data, labels, batch_size, val_split
        )
        
        # Setup loss
        criterion = self._get_loss_function(loss)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Setup checkpointing
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self._train_epoch(train_loader, optimizer, criterion, limit_batches)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self._validate_epoch(val_loader, criterion, limit_batches)
            self.history['val_loss'].append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_dir:
                    save_path = checkpoint_dir / "paw_best.pt"
                    save_checkpoint(
                        self.model,
                        save_path,
                        metadata={'epoch': epoch, 'val_loss': val_loss}
                    )
                    if verbose:
                        print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        
        if verbose:
            print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
        
        return self.history
    
    def test(
        self,
        data: Union[str, np.ndarray, torch.Tensor, Dataset],
        labels: Optional[np.ndarray] = None,
        batch_size: int = 32
    ) -> Dict:
        """Test the PAW model and compute all metrics.
        
        Args:
            data: Test data (same formats as train())
            labels: Test labels (N, 2)
            batch_size: Batch size for evaluation
            
        Returns:
            dict with all metrics
        """
        # Prepare test loader
        test_loader = self._prepare_test_loader(data, labels, batch_size)
        
        # Compute predictions and metrics
        metrics = self._compute_metrics(test_loader)
        
        return metrics
    
    def test_on_subsets(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        ids: np.ndarray,
        metadata_df: Any,  # pandas DataFrame
        subsets: list,
        batch_size: int = 32
    ) -> Dict[str, Dict]:
        """Test model on multiple data subsets.
        
        Args:
            data: Full dataset (N, T, C)
            labels: Full labels (N, 2)
            ids: Sample IDs (N,)
            metadata_df: Pandas DataFrame with metadata
            subsets: List of subset names to evaluate
            batch_size: Batch size
            
        Returns:
            dict mapping subset_name -> metrics_dict
        """
        results = {}
        
        for subset_name in subsets:
            # Filter data for this subset
            subset_indices = self._get_subset_indices(
                ids, metadata_df, subset_name
            )
            
            if len(subset_indices) == 0:
                print(f"Warning: No samples found for subset '{subset_name}'")
                continue
            
            subset_data = data[subset_indices]
            subset_labels = labels[subset_indices]
            
            # Test on subset
            metrics = self.test(subset_data, subset_labels, batch_size)
            results[subset_name] = metrics
            
            print(f"Subset '{subset_name}': {len(subset_indices)} samples - "
                  f"Accuracy: {metrics.get('window_accuracy', 0):.4f}")
        
        return results
    
    def predict(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            data: Input waveforms (N, T, C) or (N, C, T)
            
        Returns:
            predictions: Binary masks (N, C, T)
        """
        self.model.eval()
        
        # Add padding to match training preprocessing
        if isinstance(data, np.ndarray):
            padding_samples = 20
            if data.ndim == 3:  # (N, T, C)
                n_samples, seq_len, n_channels = data.shape
                padding_left = np.zeros((n_samples, padding_samples, n_channels))
                padding_right = np.zeros((n_samples, padding_samples, n_channels))
                data = np.concatenate([padding_left, data, padding_right], axis=1)
        
        # Convert to tensor
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # Ensure correct shape (N, C, T)
        if data.dim() == 2:
            data = data.unsqueeze(1)
        if data.shape[1] > data.shape[2]:  # (N, T, C) -> (N, C, T)
            data = data.permute(0, 2, 1)
        
        data = data.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(data)
            predictions = torch.sigmoid(predictions)
            predictions = self._normalize_predictions(predictions)
        
        return predictions.cpu().numpy()
    
    def save(self, path: str, metadata: Optional[Dict] = None):
        """Save model to file.
        
        Args:
            path: Path to save model
            metadata: Optional metadata dict
        """
        save_checkpoint(self.model, path, metadata=metadata)
        print(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None):
        """Load pretrained model from file.
        
        Args:
            path: Path to model checkpoint
            device: Device to load model on
            
        Returns:
            PAW instance with loaded model
        """
        # Create instance
        instance = cls(device=device)
        
        # Load model
        def model_factory():
            return build_paw_copy(device='cpu')
        
        model, metadata = load_checkpoint(path, model_factory, device=instance.device)
        instance.model = model
        
        print(f"Model loaded from {path}")
        if metadata:
            print(f"Metadata: {metadata}")
        
        return instance
    
    def print_results(self, results: Dict, title: str = "Test Results"):
        """Pretty print results.
        
        Args:
            results: Results dict from test() or test_on_subsets()
            title: Title for output
        """
        if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            # Multiple subsets
            print_subset_metrics(results, title)
        else:
            # Single result
            print_metrics(results, title)
    
    # Private helper methods
    
    def _preprocess_data(self, data, labels):
        """Load and preprocess data: load from file, add padding, convert to tensors."""
        # Load from file if needed
        if isinstance(data, str):
            data, labels = load_h5_data(data)
        
        # Add padding (0.5s = 20 samples on each side)
        if isinstance(data, np.ndarray):
            n_samples, seq_len, n_channels = data.shape
            padding = np.zeros((n_samples, 20, n_channels))
            data = np.concatenate([padding, data, padding], axis=1)
        
        # Convert to tensors
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).float()
        
        return data, labels
    
    def _prepare_dataloaders(self, data, labels, batch_size, val_split):
        """Prepare train and validation dataloaders."""
        data, labels = self._preprocess_data(data, labels)
        
        # Create dataset
        dataset = TensorDataset(data, labels)
        
        # Split
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _prepare_test_loader(self, data, labels, batch_size):
        """Prepare test dataloader."""
        data, labels = self._preprocess_data(data, labels)
        
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return loader
    
    def _get_loss_function(self, loss_name: str):
        """Get loss function by name."""
        loss_name = loss_name.lower()
        
        if loss_name == 'dice':
            return DiceLoss()
        elif loss_name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_name == 'amper':
            return AmpPerLoss()
        elif loss_name == 'amperwdw':
            return AmpPerLossWdW()
        elif loss_name in ['bce+dice', 'bcedice']:
            return BCEDiceLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def _train_epoch(self, loader, optimizer, criterion, limit_batches=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = int(len(loader) * limit_batches) if limit_batches else len(loader)
        
        for i, (data, labels) in enumerate(loader):
            if i >= num_batches:
                break
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Ensure correct shapes
            if data.dim() == 3 and data.shape[2] < data.shape[1]:
                data = data.permute(0, 2, 1)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            
            # Create binary masks from labels
            masks = self._labels_to_masks(labels, data.shape[2])
            
            # Handle different loss signatures
            if criterion.__class__.__name__ in ['AmpPerLoss', 'AmpPerLossWdW', 'DiceLoss']:
                # These losses expect (x, y_pred, y_true) with shapes squeezed
                x_for_loss = data[:, 0:1, :]  # Take first channel
                outputs_squeezed = outputs.squeeze(1) if outputs.shape[1] == 1 else outputs
                masks_squeezed = masks.squeeze(1) if masks.shape[1] == 1 else masks
                loss = criterion(x_for_loss, outputs_squeezed, masks_squeezed)
            else:
                loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        actual_batches = min(num_batches, len(loader)) if limit_batches else len(loader)
        return total_loss / actual_batches
    
    def _validate_epoch(self, loader, criterion, limit_batches=None):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = int(len(loader) * limit_batches) if limit_batches else len(loader)
        
        with torch.no_grad():
            for i, (data, labels) in enumerate(loader):
                if i >= num_batches:
                    break
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                if data.dim() == 3 and data.shape[2] < data.shape[1]:
                    data = data.permute(0, 2, 1)
                
                outputs = self.model(data)
                
                # Create binary masks from labels
                masks = self._labels_to_masks(labels, data.shape[2])
                
                if criterion.__class__.__name__ in ['AmpPerLoss', 'AmpPerLossWdW', 'DiceLoss']:
                    x_for_loss = data[:, 0:1, :]
                    outputs_squeezed = outputs.squeeze(1) if outputs.shape[1] == 1 else outputs
                    masks_squeezed = masks.squeeze(1) if masks.shape[1] == 1 else masks
                    loss = criterion(x_for_loss, outputs_squeezed, masks_squeezed)
                else:
                    loss = criterion(outputs, masks)
                
                total_loss += loss.item()
        
        actual_batches = min(num_batches, len(loader)) if limit_batches else len(loader)
        return total_loss / actual_batches
    
    def _labels_to_masks(self, labels, seq_len, freq=0.025, padding_offset=20):
        """Convert start/end labels in SECONDS to binary masks.
        
        Args:
            labels: (N, 2) tensor with [start_time, end_time] in SECONDS
            seq_len: Length of sequence
            freq: Sampling frequency (default 0.025s = 40Hz)
            padding_offset: Offset in samples due to padding (default 20 = 0.5s padding)
        
        Returns:
            masks: (N, 1, seq_len) binary masks
        """
        batch_size = labels.shape[0]
        masks = torch.zeros(batch_size, 1, seq_len, device=labels.device)
        
        for i in range(batch_size):
            # Labels are in seconds, convert to sample indices
            start_sec = labels[i, 0].item()
            end_sec = labels[i, 1].item()
            
            # Convert seconds to indices: divide by freq (or multiply by 1/freq = 40)
            start_idx = int(round(start_sec / freq)) + padding_offset
            end_idx = int(round(end_sec / freq)) + padding_offset
            
            # Clip to valid range
            start_idx = max(0, min(start_idx, seq_len - 1))
            end_idx = max(0, min(end_idx, seq_len - 1))
            
            if start_idx <= end_idx:
                masks[i, 0, start_idx:end_idx+1] = 1.0
        
        return masks
    
    def _normalize_predictions(self, preds):
        """Normalize predictions to [0,1] per sample (matching main.py postprocessing).
        
        Args:
            preds: (N, C, T) predictions
            
        Returns:
            Normalized predictions
        """
        # Normalize each sample independently to [0, 1]
        min_vals = preds.min(dim=2, keepdim=True)[0]
        max_vals = preds.max(dim=2, keepdim=True)[0]
        preds_norm = (preds - min_vals) / (max_vals - min_vals + 1e-8)
        preds_norm = torch.nan_to_num(preds_norm, nan=0.0)
        preds_norm = torch.clip(preds_norm, min=0, max=1)
        return preds_norm
    
    def _limit_to_half_cycle(self, signals, windows):
        """Crop predicted windows to exact half-cycle (matching main.py postprocessing).
        
        Args:
            signals: (N, T) signal data
            windows: (N, 2) window boundaries [start, end]
            
        Returns:
            Cropped windows
        """
        batch_size = signals.shape[0]
        results = windows.clone()
        
        for i in range(batch_size):
            start, end = int(windows[i, 0]), int(windows[i, 1])
            # Respect padding bounds
            start = max(19, start)
            end = min(signals.shape[1] - 20, end)
            
            if end <= start:
                continue
                
            window_signal = signals[i, start:end+1]
            
            max_idx = torch.argmax(window_signal).item()
            min_idx = torch.argmin(window_signal).item()
            max_val = window_signal[max_idx].item()
            min_val = window_signal[min_idx].item()
            
            # Find extremum (max or min with larger absolute value)
            if abs(max_val) >= abs(min_val):
                extremum_idx = max_idx
                extremum_val = max_val
            else:
                extremum_idx = min_idx
                extremum_val = min_val
                
            abs_extremum_idx = start + extremum_idx
            
            # Choose half-cycle with larger amplitude
            option_a_amp = abs(extremum_val - window_signal[0].item())
            option_b_amp = abs(window_signal[-1].item() - extremum_val)
            
            if option_a_amp >= option_b_amp:
                results[i, 1] = min(abs_extremum_idx+1, end)
            else:
                results[i, 0] = max(abs_extremum_idx-1, start)
        
        return results
    
    def _compute_metrics(self, loader):
        """Compute all evaluation metrics."""
        self.model.eval()
        
        # Initialize metrics and move to device
        window_acc = WindowAccuracy().to(self.device)
        amp_rmse = AmplitudeRMSE(squared=False).to(self.device)
        per_rmse = PeriodRMSE(squared=False).to(self.device)
        mag_rmse = MagnitudeRMSE(squared=False).to(self.device)
        wdw_rmse = WindowRMSE(squared=False).to(self.device)
        dice_score = DiceScore().to(self.device)
        
        all_preds = []
        all_targets = []
        all_data = []
        
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                if data.dim() == 3 and data.shape[2] < data.shape[1]:
                    data = data.permute(0, 2, 1)
                
                outputs = self.model(data)
                preds = torch.sigmoid(outputs)
                preds = self._normalize_predictions(preds)
                
                # Convert labels to masks
                masks = self._labels_to_masks(labels, data.shape[2])
                
                # Extract windows from predictions and targets
                pred_windows = extract_windows_from_masks(preds)
                target_windows = extract_windows_from_masks(masks)
                
                pred_windows = torch.from_numpy(pred_windows).to(self.device)
                target_windows = torch.from_numpy(target_windows).to(self.device)
                
                # Apply half-cycle cropping
                signal_data = data[:, 0, :] 
                pred_windows = self._limit_to_half_cycle(signal_data, pred_windows)
                
                # Update metrics
                window_acc.update(pred_windows, target_windows)
                amp_rmse.update(data, pred_windows, target_windows)
                per_rmse.update(pred_windows, target_windows)
                mag_rmse.update(data, pred_windows, target_windows)
                wdw_rmse.update(preds, masks)
                dice_score.update(preds, masks)
                
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
                all_data.append(data.cpu())
        
        # Compute final metrics
        results = {
            'window_accuracy': window_acc.compute().item(),
            'amplitude_rmse': amp_rmse.compute().item(),
            'period_rmse': per_rmse.compute().item(),
            'magnitude_rmse': mag_rmse.compute().item(),
            'window_rmse': wdw_rmse.compute().item(),
            'dice_score': dice_score.compute().item()
        }
        
        return results
    
    def _get_subset_indices(self, ids, metadata_df, subset_name):
        """Get indices for a named subset from metadata."""
        # This is a simplified version - adapt based on your metadata structure
        subset_name_lower = subset_name.lower()
        
        if subset_name_lower == 'eval':
            mask = metadata_df['PER_eval'] > 0
        elif subset_name_lower == 'adjusted':
            mask = metadata_df['PER_eval'] > 0
        elif subset_name_lower == 'accepted':
            mask = metadata_df['PER_eval'] > 0
        elif subset_name_lower == '3c_sta':
            sta_3c = ['BOSA', 'CPUP', 'DBIC', 'LBTB', 'LPAZ', 'PLCA', 'VNDA']
            mask = metadata_df['STA'].isin(sta_3c)
        elif subset_name_lower == 'arr_sta':
            sta_arr = ['ASAR', 'BRTR', 'CMAR', 'ILAR', 'KSRS', 'MKAR', 'PDAR', 'TXAR']
            mask = metadata_df['STA'].isin(sta_arr)
        else:
            # Custom subset - assume it's a column name
            if subset_name in metadata_df.columns:
                mask = metadata_df[subset_name] > 0
            else:
                return []
        
        subset_ids = metadata_df[mask]['ID'].values
        indices = [i for i, id in enumerate(ids) if id in subset_ids]
        
        return indices


__all__ = ["PAW"]

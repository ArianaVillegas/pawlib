"""Metrics for seismic phase detection - copied from working src/metrics/"""
import torch
from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredError


class AmplitudeRMSE(Metric):
    def __init__(self, squared, num_outputs=1):
        super().__init__()
        self.rmse = MeanSquaredError(squared=squared)

    def _get_amplitude_item(self, x, wdw):
        wdw = wdw.cpu().numpy().astype(int)
        if wdw[0] >= wdw[1]:
            return 0
        x_wdw = x[wdw[0]:wdw[1]]
        amp = (x_wdw.max() - x_wdw.min())/2
        return amp

    def _get_amplitude(self, x, wdw):
        amp = []
        for xi, wi in zip(x, wdw):
            x_amp = self._get_amplitude_item(xi.squeeze(), wi)
            amp.append(x_amp)
        amp = torch.tensor(amp, device=x.device, dtype=torch.float32)
        return amp

    def update(self, x, pred_wdw, target_wdw):
        assert pred_wdw.shape == target_wdw.shape
        pred_amp = self._get_amplitude(x, pred_wdw)
        pred_amp = torch.where(pred_amp > 0, pred_amp, torch.zeros_like(pred_amp))
        target_amp = self._get_amplitude(x, target_wdw)
        self.rmse(pred_amp, target_amp)

    def compute(self):
        return self.rmse.compute()


class PeriodRMSE(Metric):
    def __init__(self, squared, num_outputs=1):
        super().__init__()
        self.rmse = MeanSquaredError(squared=squared)

    def update(self, pred_wdw, target_wdw, freq=0.025):
        assert pred_wdw.shape == target_wdw.shape
        pred_per = pred_wdw[:, 1] - pred_wdw[:, 0]
        pred_per = torch.where(pred_per > 0, pred_per, 0) * freq * 2
        target_per = (target_wdw[:, 1] - target_wdw[:, 0]) * freq * 2
        self.rmse(pred_per, target_per)

    def compute(self):
        return self.rmse.compute()


class MagnitudeRMSE(Metric):
    def __init__(self, squared, num_outputs=1):
        super().__init__()
        self.rmse = MeanSquaredError(squared=squared)

    def _get_amplitude_item(self, x, wdw):
        wdw = wdw.cpu().numpy().astype(int)
        if wdw[0] >= wdw[1]:
            return 0
        x_wdw = x[wdw[0]:wdw[1]]
        amp = (x_wdw.max() - x_wdw.min())/2
        return amp

    def _get_amplitude(self, x, wdw):
        amp = []
        for xi, wi in zip(x, wdw):
            x_amp = self._get_amplitude_item(xi.squeeze(), wi)
            amp.append(x_amp)
        amp = torch.tensor(amp, device=x.device, dtype=torch.float32)
        return amp

    def update(self, x, pred_wdw, target_wdw, freq=0.025):
        assert pred_wdw.shape == target_wdw.shape
        pred_per = pred_wdw[:, 1] - pred_wdw[:, 0]
        pred_per = torch.where(pred_per > 0, pred_per, torch.zeros_like(pred_per)) * freq * 2
        target_per = (target_wdw[:, 1] - target_wdw[:, 0]) * freq * 2

        pred_amp = self._get_amplitude(x, pred_wdw)
        pred_amp = torch.where(pred_amp > 0, pred_amp, torch.zeros_like(pred_amp))
        target_amp = self._get_amplitude(x, target_wdw)

        pred_mag = torch.where((pred_amp == 0) | (pred_per == 0), torch.zeros_like(pred_amp), torch.log(pred_amp) - torch.log(pred_per))
        target_mag = torch.where((target_amp == 0) | (target_per == 0), torch.zeros_like(target_amp), torch.log(target_amp) - torch.log(target_per))

        self.rmse(pred_mag, target_mag)

    def compute(self):
        return self.rmse.compute()


class WindowAccuracy(Metric):
    def __init__(self, th=0.5, tol=4, relative=False):
        super().__init__()
        self.th = th
        self.tol = tol
        self.relative = relative
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds_wdw, target_wdw):
        assert preds_wdw.shape == target_wdw.shape

        diff = torch.abs(preds_wdw - target_wdw)
        diff = torch.sum(diff, dim=1)
        if self.relative:
            diff /= (target_wdw[1] - target_wdw[0])
        count = torch.sum(diff <= self.tol)

        self.correct += count
        self.total += target_wdw.shape[0]

    def compute(self):
        return self.correct.float() / self.total


# Also add DiceScore and WindowRMSE for completeness
class DiceScore(Metric):
    """Dice coefficient for binary segmentation."""
    def __init__(self, smooth=1.0, threshold=0.5):
        super().__init__()
        self.smooth = smooth
        self.threshold = threshold
        self.add_state("dice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred, target):
        """
        Args:
            pred: Predicted masks (N, C, T) - probabilities or binary
            target: Target masks (N, C, T) - binary
        """
        # Binarize predictions if needed
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
        
        pred_binary = (pred > self.threshold).float()
        
        # Compute Dice per sample
        for p, t in zip(pred_binary, target):
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            self.dice_sum += dice
            self.total += 1

    def compute(self):
        return self.dice_sum / self.total


class WindowRMSE(Metric):
    """RMSE of window predictions (pixel-wise on binary masks)."""
    def __init__(self, squared=False):
        super().__init__()
        self.squared = squared
        self.rmse = MeanSquaredError(squared=squared)

    def update(self, pred, target):
        """
        Args:
            pred: Predicted masks (N, C, T)
            target: Target masks (N, C, T)
        """
        self.rmse(pred.flatten(), target.flatten())

    def compute(self):
        return self.rmse.compute()


__all__ = [
    "WindowAccuracy",
    "AmplitudeRMSE", 
    "PeriodRMSE",
    "MagnitudeRMSE",
    "WindowRMSE",
    "DiceScore"
]

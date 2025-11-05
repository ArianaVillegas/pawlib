"""Loss functions for seismic phase detection - copied from working src/losses.py"""
import numpy as np
import torch
import torch.nn as nn


# Helper functions (inlined from src/data_utils.py for self-containment)
def get_per_amp(yp, th=0.5):
    """Extract window boundaries from prediction."""
    yp = yp.flatten()
    yp_s = yp.argmax()
    yp_st = yp[yp_s]
    while yp_st > th:
        if yp_s > 0:
            yp_st = yp[yp_s]
            yp_s -= 1
        else:
            yp_st = th
    yp_s += 1

    yp_e = yp.argmax()
    yp_et = yp[yp_e]
    while yp_et > th:
        if yp_e < len(yp):
            yp_et = yp[yp_e]
            yp_e += 1
        else:
            yp_et = th
    yp_e -= 1

    return yp_s, yp_e


def get_pred_tensor(x_true, y_true, y_pred, freq=0.025):
    """Extract amplitude and period tensors from predictions."""
    amp_true, per_true = [], []
    amp_pred, per_pred = [], []
    for xt, yt, yp in zip(x_true, y_true, y_pred):
        xt = xt.squeeze()
        yt_nonzero = np.nonzero(yt).squeeze()
        if len(yt_nonzero) > 0:
            yt_s = yt_nonzero.min()
            yt_e = yt_nonzero.max()
            yt_per = yt_e - yt_s
            per_true.append(yt_per * freq)

            yp_s, yp_e = get_per_amp(yp)
            yp_per = yp_e - yp_s
            per_pred.append(yp_per * freq)

            mpeaks_t = np.array(xt[yt_s:yt_e].cpu(), dtype=np.float64)
            mpeaks_t = np.abs(np.max(mpeaks_t) - np.min(mpeaks_t)) / 2
            amp_true.append(mpeaks_t)
            if yp_s < yp_e:
                mpeaks_p = np.array(xt[yp_s:yp_e].cpu(), dtype=np.float64)
                mpeaks_p = np.abs(np.max(mpeaks_p) - np.min(mpeaks_p)) / 2
                amp_pred.append(mpeaks_p)
            else:
                amp_pred.append(0)
    per_true = torch.FloatTensor(per_true).requires_grad_(True)
    per_pred = torch.FloatTensor(per_pred).requires_grad_(True)
    amp_true = torch.FloatTensor(amp_true).requires_grad_(True)
    amp_pred = torch.FloatTensor(amp_pred).requires_grad_(True)
    return per_true, per_pred, amp_true, amp_pred


class BCEDiceLoss(nn.Module):
    """Composite BCE + Dice loss tailored for PAW segmentation masks.
    
    Supports both logits (from_logits=True) and probabilities (from_logits=False).
    """

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 0.25,
        smooth: float = 1.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.from_logits = from_logits
        if from_logits:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs: Model outputs (logits if from_logits=True, probabilities otherwise)
            targets: Ground truth binary masks
        """
        bce_loss = self.bce(outputs, targets)
        
        # Get probabilities
        if self.from_logits:
            probs = torch.sigmoid(outputs)
        else:
            probs = outputs
        
        # Compute Dice loss
        intersection = (probs * targets).sum(dim=(1, 2))
        denom = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, lambda_p=1, lambda_t=0.5, lambda_d=0.25):
        super().__init__()
        self.smooth = smooth
        self.lambda_p = lambda_p
        self.lambda_t = lambda_t
        self.lambda_d = lambda_d
        self.loss_ae = nn.BCEWithLogitsLoss()
        self.loss_per = nn.MSELoss()
        self.loss_amp = nn.MSELoss()

    def temporal_consistency_loss(self, y_pred):
        return torch.mean(torch.abs(y_pred[:, 1:] - y_pred[:, :-1]))

    def forward(self, x, y_pred, y_true):
        bce_loss = self.loss_ae(y_pred, y_true)
        y_pred = torch.sigmoid(y_pred)
        intersection = (y_pred * y_true).sum(dim=1)
        union = y_pred.sum(dim=1) + y_true.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return bce_loss * self.lambda_p + dice_loss * self.lambda_d + self.lambda_t * self.temporal_consistency_loss(y_pred)


class AmpPerLoss(nn.Module):
    def __init__(self, bce_weight=1.0, amplitude_weight=0.5, smoothness_weight=0.3):
        super(AmpPerLoss, self).__init__()
        self.bce_weight = bce_weight
        self.amplitude_weight = amplitude_weight
        self.smoothness_weight = smoothness_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, signals, predictions, targets):
        if len(predictions.shape) == 2:
            predictions = predictions.unsqueeze(1)
        else:
            predictions = predictions
        if len(targets.shape) == 2:
            targets = targets.unsqueeze(1)
        else:
            targets = targets
        pred_sigmoid = torch.sigmoid(predictions)
        bce_loss = self.bce(predictions, targets)
        batch_size = signals.shape[0]
        amp_loss = 0
        for i in range(batch_size):
            signal = signals[i, 0]
            pred = pred_sigmoid[i]
            target = targets[i]
            pred_indices = torch.where(pred > 0.5)[0]
            target_indices = torch.where(target > 0.5)[0]
            if len(target_indices) > 0 and len(pred_indices) > 0:
                true_min_idx = torch.min(target_indices)
                true_max_idx = torch.max(target_indices)
                true_window = signal[true_min_idx:true_max_idx + 1]
                pred_min_idx = torch.min(pred_indices)
                pred_max_idx = torch.max(pred_indices)
                if pred_min_idx >= len(signal) or pred_max_idx >= len(signal):
                    continue
                pred_window = signal[pred_min_idx:pred_max_idx + 1]
                if len(true_window) == 0 or len(pred_window) == 0:
                    continue
                true_amp = torch.max(true_window) - torch.min(true_window)
                pred_amp = torch.max(pred_window) - torch.min(pred_window)
                if true_amp > 1e-6:
                    amp_loss += torch.abs(true_amp - pred_amp) / (true_amp + 1e-6)
                else:
                    amp_loss += torch.abs(true_amp - pred_amp)
        amp_loss = amp_loss / max(batch_size, 1)
        diff = torch.diff(pred_sigmoid, dim=1)
        smoothness_loss = 0 if diff.shape[1] == 0 else torch.mean(torch.abs(diff))
        total_loss = (
            self.bce_weight * bce_loss
            + self.amplitude_weight * amp_loss
            + self.smoothness_weight * smoothness_loss
        )
        return total_loss


class AmpPerLossWdW(nn.Module):
    def __init__(self, weight=0.1, ratio=1, inv_weight=0.01, freq=0.025):
        super().__init__()
        self.loss_amp = nn.MSELoss()
        self.loss_per = nn.MSELoss()
        self.loss_wdw = nn.MSELoss()
        self.loss_ae = nn.BCELoss()
        self.weight = weight
        self.ratio = ratio
        self.inv_weight = inv_weight
        self.freq = freq

    def forward(self, x_true, y_pred, y_true):
        per_pred, per_true = [], []
        amp_pred, amp_true = [], []
        sin_pred, sin_true = [], []
        for xt, yt, yp in zip(x_true, y_true, y_pred):
            xt = xt.squeeze()
            yt_nonzero = np.nonzero(yt).squeeze()
            if len(yt_nonzero) > 0:
                yt_s = yt_nonzero.min()
                yt_e = yt_nonzero.max()
                yt_per = yt_e - yt_s
                per_true.append(yt_per * self.freq)
                yp_s, yp_e = get_per_amp(yp)
                yp_per = max(yp_e - yp_s, 0)
                per_pred.append(yp_per * self.freq)
                mpeaks_t = np.array(xt[yt_s:yt_e], dtype=np.float64)
                mpeaks_t = np.abs(np.max(mpeaks_t) - np.min(mpeaks_t))
                amp_true.append(mpeaks_t)
                if yp_s < yp_e:
                    mpeaks_p = np.array(xt[yp_s:yp_e], dtype=np.float64)
                    mpeaks_p = np.abs(np.max(mpeaks_p) - np.min(mpeaks_p))
                    amp_pred.append(mpeaks_p)
                    frequency = 1 / (yp_per * self.freq * 2)
                    sinewave = np.sin(2 * np.pi * frequency * np.arange(0, 10, self.freq))
                    start_sine = int(yp_per * 2 * 0.25) + 1
                    sinewave = sinewave[start_sine:start_sine + yp_per]
                    if sinewave.shape != xt[yp_s:yp_e].shape:
                        sin_pred.append(1)
                    corr_coef = np.corrcoef(xt[yp_s:yp_e], sinewave)
                    corr_coef = np.abs(corr_coef[0][1])
                    sin_pred.append(corr_coef)
                else:
                    amp_pred.append(0)
                    sin_pred.append(1)
                sin_true.append(0)
        per_true = torch.FloatTensor(per_true).requires_grad_(True)
        per_pred = torch.FloatTensor(per_pred).requires_grad_(True)
        amp_true = torch.FloatTensor(amp_true).requires_grad_(True)
        amp_pred = torch.FloatTensor(amp_pred).requires_grad_(True)
        sin_true = torch.FloatTensor(sin_true).requires_grad_(True)
        sin_pred = torch.FloatTensor(sin_pred).requires_grad_(True)
        per_loss = self.loss_per(per_pred, per_true) * self.weight / 5
        amp_loss = self.loss_amp(amp_pred, amp_true) * self.weight
        wdw_loss = self.loss_wdw(sin_pred, sin_true) * self.weight * 0.5
        loss = torch.norm(torch.stack([per_loss, amp_loss]), p=2)
        ae_loss = self.loss_ae(y_pred, y_true)
        return ae_loss + wdw_loss


__all__ = ["BCEDiceLoss", "DiceLoss", "AmpPerLoss", "AmpPerLossWdW"]

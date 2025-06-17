import torch
import torch.nn.functional as F

def masked_mse(pred, target, mask):
    """
    pred, target: [B, H, W] or [B, C, H, W]
    mask: [H, W], [1, 1, H, W], or [B, H, W]
    """
    mask = torch.as_tensor(mask, device=pred.device, dtype=pred.dtype)
    # Squeeze mask to remove extra dimensions
    while mask.ndim > pred.ndim:
        mask = mask.squeeze(0)
    # Now expand mask to match pred
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(pred)
    diff = (pred - target) ** 2
    masked_diff = diff[mask > 0]
    return masked_diff.mean()

def masked_mae(pred, target, mask):
    """
    pred, target: [B, H, W] or [B, C, H, W]
    mask: [H, W], [1, 1, H, W], or [B, H, W]
    """
    mask = torch.as_tensor(mask, device=pred.device, dtype=pred.dtype)
    # Squeeze mask to remove extra dimensions
    while mask.ndim > pred.ndim:
        mask = mask.squeeze(0)
    # Now expand mask to match pred
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(pred)
    diff = torch.abs(pred - target)
    masked_diff = diff[mask > 0]
    return masked_diff.mean()

def masked_rmse(pred, target, mask):
    return torch.sqrt(masked_mse(pred, target, mask))

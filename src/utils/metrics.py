import math
import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    """
    pred, target: Tensors of shape (N, C, H, W) or (C, H, W)
    """
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(max_val / math.sqrt(mse))

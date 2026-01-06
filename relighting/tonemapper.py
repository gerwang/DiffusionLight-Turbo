import numpy as np
import torch
import torch.nn as nn

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha, tonemapped_img


class TonemapHDRTorch(nn.Module):
    """
        Tonemap HDR image globally in torch.
        input : torch.Tensor image (H, W, C) or any shape
        output : torch.Tensor image (same shape), alpha, tonemapped_img
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        super().__init__()
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping

    def forward(self, image, clip=True, alpha=None, gamma=True):
        if gamma:
            power = torch.clamp(image, min=0.0) ** (1.0 / self.gamma)
        else:
            power = image
        non_zero = power > 0
        if torch.any(non_zero):
            values = power[non_zero]
        else:
            values = power.reshape(-1)
        if values.numel() == 0:
            r_percentile = torch.tensor(0.0, device=power.device, dtype=power.dtype)
        else:
            r_percentile = torch.quantile(values, self.percentile / 100.0)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        elif not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=power.device, dtype=power.dtype)
        tonemapped_img = power * alpha
        if clip:
            tonemapped_img_clip = torch.clamp(tonemapped_img, 0.0, 1.0)
        else:
            tonemapped_img_clip = tonemapped_img
        return tonemapped_img_clip, alpha, tonemapped_img

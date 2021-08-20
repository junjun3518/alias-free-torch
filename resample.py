import torch
import torch.nn as nn
import torch.nn.functional as F
from .filter import LowPassFilter1d, LowPassFilter2d


class UpSample1d(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       kernel_size=int(6 * ratio // 2) * 2)

    def forward(self, x):
        shape = list(x.shape)
        new_shape = shape[:-1] + [shape[-1] * self.ratio]
        xx = torch.zeros(new_shape, device=x.device)
        xx[..., ::self.ratio] = x
        xx = self.ratio * xx
        x = self.lowpass(xx.view(new_shape))
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=int(6 * ratio // 2) * 2)

    def forward(self, x):
        xx = self.lowpass(x)
        return xx


class UpSample2d(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassFilter2d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       kernel_size=int(6 * ratio // 2) * 2)

    def forward(self, x):
        shape = list(x.shape)
        new_shape = shape[:-2] + [shape[-2] * self.ratio
                                  ] + [shape[-1] * self.ratio]

        xx = torch.zeros(new_shape, device=x.device)
        #shape + [self.ratio**2], device=x.device)
        xx[..., ::self.ratio, ::self.ratio] = x
        xx = self.ratio**2 * xx
        x = self.lowpass(xx.view(new_shape))
        return x


class DownSample2d(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassFilter2d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=int(6 * ratio // 2) * 2)

    def forward(self, x):
        xx = self.lowpass(x)
        return xx


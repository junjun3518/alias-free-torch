import torch
import torch.nn as nn
import torch.nn.functional as F
from .filter import LowPassFilter1d, LowPassFilter2d
from .filter import kaiser_sinc_filter1d, kaiser_jinc_filter2d


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, even=True):
        super().__init__()
        self.ratio = ratio
        self.even = even
        kernel_size = int(6 * ratio // 2) * 2 + int(not (even))
        self.stride = ratio
        self.pad = kernel_size // 2 - ratio // 2
        self.filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                           half_width=0.6 / ratio,
                                           kernel_size=kernel_size)

    def forward(self, x):
        x = self.ratio * F.conv_transpose1d(
            x, self.filter, stride=self.stride, padding=self.pad)
        if not self.even:
            x = x[..., :-1]
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
    def __init__(self, ratio=2, even=True):
        super().__init__()
        self.ratio = ratio
        self.even = even
        kernel_size = int(6 * ratio // 2) * 2 + int(not (even))
        self.stride = ratio
        self.pad = kernel_size // 2 - ratio // 2
        self.filter = kaiser_jinc_filter2d(cutoff=0.5 / ratio,
                                           half_width=0.6 / ratio,
                                           kernel_size=kernel_size)

    def forward(self, x):
        print(x.shape)
        x = self.ratio**2 * F.conv_transpose2d(
            x, self.filter, stride=self.stride, padding=self.pad)
        if not self.even:
            x = x[..., :-1, :-1]
        print(x.shape)
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

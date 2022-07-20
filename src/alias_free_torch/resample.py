import torch
import torch.nn as nn
import torch.nn.functional as F
from .filter import LowPassFilter1d, LowPassFilter2d
from .filter import kaiser_sinc_filter1d, kaiser_jinc_filter2d


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size = None, even=True):
        super().__init__()
        self.ratio = ratio
        self.even = even
        self.kernel_size = int(6 * ratio // 2) * 2 + int(not (even)) if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                    half_width=0.6 / ratio,
                                    kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    def forward(self, x):
        shape = list(x.shape)
        new_shape = shape[:-1] + [-1]
        x = x.view(-1, 1, shape[-1])
        x = F.pad(x, (self.pad, self.pad), mode = 'reflect')
        x = self.ratio * F.conv_transpose1d(
            x, self.filter, stride=self.stride)
        out_pad_1 = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        out_pad_2 = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        x = x[..., out_pad_1:-out_pad_2]
        return x.reshape(new_shape)


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
        x = F.pad(x, (self.pad, self.pad), mode = 'reflect')
        x = self.ratio**2 * F.conv_transpose2d(
            x, self.filter, stride=self.stride)
        if not self.even:
            x = x[..., :-1, :-1]
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

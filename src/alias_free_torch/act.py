import torch
import torch.nn as nn
import torch.nn.functional as F
from .resample import UpSample1d, DownSample1d
from .resample import UpSample2d, DownSample2d


class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio)
        self.downsample = DownSample1d(down_ratio)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


class Activation2d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample2d(up_ratio)
        self.downsample = DownSample2d(down_ratio)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x

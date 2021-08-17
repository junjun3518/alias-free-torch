import torch
import torch.nn as nn
import torch.nn.functional as F
from filter import LowPassFilter
from resample import UpSample, DownSample

class Activation(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act =activation
        self.upsample = UpSample(up_ratio)
        self.downsample = DownSample(down_ratio)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc
    # https://adefossez.github.io/julius/julius/core.html
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(x == 0,
                           torch.tensor(1., device=x.device, dtype=x.dtype),
                           torch.sin(math.pi * x) / math.pi / x)


# This code is adopted from adefossez's julius.lowpass.LowPassFilters
# https://adefossez.github.io/julius/julius/lowpass.html
class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 pad: bool = True,
                 kernel_size=12
                 ):  # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.stride = stride
        self.pad = pad
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.half_size = kernel_size // 2
        self.stride = stride

        #For kaiser window
        delta_f = 4 * half_width
        A = 2.285 * (self.half_size - 1) * math.pi * delta_f + 7.95
        if A > 50.:
            beta = 0.1102 * (A - 8.7)
        elif A >= 21.:
            beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
        else:
            beta = 0.
        window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
        #ratio = 0.5/cutroff
        if self.even:
            time = (torch.arange(-self.half_size, self.half_size) + 0.5)
        else:
            time = torch.arange(self.kernel_size) - self.half_size
        if cutoff == 0:
            filter_ = torch.zeros_like(time)
        else:
            filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
            # Normalize filter to have sum = 1, otherwise we will have a small leakage
            # of the constant component in the input signal.
            filter_ /= filter_.sum()
            filter = filter_.view(1, 1, self.kernel_size)
        self.register_buffer("filter", filter)

    #input [B,T] or [B,C,T]
    def forward(self, x):
        shape = list(x.shape)
        new_shape = shape[:-1] + [-1]
        x = x.view(-1, 1, shape[-1])
        if self.pad:
            x = F.pad(x, (self.half_size, self.half_size),
                      mode='constant',
                      value=0)  # empirically, it is better than replicate
            #mode='replicate')
        if self.even:
            out = F.conv1d(x, self.filter, stride=self.stride)[..., :-1]
        else:
            out = F.conv1d(x, self.filter, stride=self.stride)
        return out.reshape(new_shape)


class LowPassFilter2d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 pad: bool = True,
                 kernel_size=12):  # kernel_size should be even number
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.stride = stride
        self.pad = pad
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.half_size = kernel_size // 2
        self.stride = stride

        #For kaiser window
        delta_f = 4 * half_width
        A = 2.285 * (self.half_size - 1) * math.pi * delta_f + 7.95
        if A > 50.:
            beta = 0.1102 * (A - 8.7)
        elif A >= 21.:
            beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
        else:
            beta = 0.

        #rotation equivariant grid
        if self.even:
            time = torch.stack(torch.meshgrid(
                torch.arange(-self.half_size, self.half_size) + 0.5,
                torch.arange(-self.half_size, self.half_size) + 0.5),
                               dim=-1)
        else:
            time = torch.stack(torch.meshgrid(
                torch.arange(self.kernel_size) - self.half_size,
                torch.arange(self.kernel_size) - self.half_size),
                               dim=-1)

        time = torch.norm(time, dim=-1)
        #rotation equivariant window
        window = torch.i0(
            beta * torch.sqrt(1 -
                              (time / self.half_size / 2**0.5)**2)) / torch.i0(
                                  torch.tensor([beta]))
        #ratio = 0.5/cutroff
        #using sinc instead jinc
        if cutoff == 0:
            filter_ = torch.zeros_like(time)
        else:
            filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
            # Normalize filter to have sum = 1, otherwise we will have a small leakage
            # of the constant component in the input signal.
            filter_ /= filter_.sum()
            filter = filter_.view(1, 1, self.kernel_size, self.kernel_size)
        self.register_buffer("filter", filter)

    #input [B,C,W,H] or [B,W,H] or [W,H]
    def forward(self, x):
        shape = list(x.shape)
        x = x.view(-1, 1, shape[-2], shape[-1])
        if self.pad:
            x = F.pad(
                x, (self.half_size, self.half_size, self.half_size,
                    self.half_size),
                mode='constant',
                value=0)  # empirically, it is better than replicate or reflect
            #mode='replicate')
        if self.even:
            out = F.conv2d(x, self.filter, stride=self.stride)[..., :-1, :-1]
        else:
            out = F.conv2d(x, self.filter, stride=self.stride)

        new_shape = shape[:-2] + list(out.shape)[-2:]
        return out.reshape(new_shape)

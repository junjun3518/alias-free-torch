import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# This code is adopted from adefossez's julius.lowpass.LowPassFilters
# https://adefossez.github.io/julius/julius/lowpass.html
class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 pad: bool = True,
                 kernel_size=12):  #kernel_size should be even number
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")


#        if cutoff > 0.5:
#            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.stride = stride
        self.pad = pad
        self.kernel_size = kernel_size
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
        time = (torch.arange(-self.half_size, self.half_size) + 0.5) 
        if cutoff == 0:
            filter_ = torch.zeros_like(time)
        else:
            filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
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
            x = F.pad(
                x,
                (self.half_size, self.half_size),
                #mode='constant', value=0) if you want
                mode='replicate')
        out = F.conv1d(x, self.filter, stride=self.stride)[..., :-1]
        return out.reshape(new_shape)


class LowPassFilter2d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 pad: bool = True,
                 kernel_size=12):  #kernel_size should be even number
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")


#        if cutoff > 0.5:
#            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.stride = stride
        self.pad = pad
        self.kernel_size = kernel_size
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
        time = (torch.stack(torch.meshgrid(
            torch.arange(-self.half_size, self.half_size) + 0.5,
            torch.arange(-self.half_size, self.half_size) + 0.5),
                            dim=-1))  
        time = torch.norm(time, dim=-1)
        #rotation equivariant window
        window = torch.i0(
            beta * torch.sqrt(1 -
                              (time / self.half_size / 2**0.5)**2)) / torch.i0(
                                  torch.tensor([beta]))
        #ratio = 0.5/cutroff
        time = time / (0.5 / cutoff)
        #using sinc instead jinc
        if cutoff == 0:
            filter_ = torch.zeros_like(time)
        else:
            filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
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
            x = F.pad(x, (self.half_size, self.half_size, self.half_size,
                          self.half_size),
                      mode='reflect')
                      #value=0)

        out = F.conv2d(x, self.filter, stride=self.stride)[..., :-1, :-1]
        new_shape = shape[:-2] + list(out.shape)[-2:]
        return out.reshape(new_shape)

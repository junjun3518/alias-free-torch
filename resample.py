import torch
import torch.nn as nn
import torch.nn.functional as F
from filter import LowPassFilter

class UpSample(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassFilter(cutoff=0.5/ratio, half_width=0.6/ratio, kernel_size=int(6*ratio//2)*2)


    def forward(self, x):
        shape = list(x.shape)
        new_shape = shape[:-1]+[shape[-1]*self.ratio]

        ##Faster for cpu
        #total_elements = 1
        #for s in new_shape:
        #    total_elements *=s
        #xx = torch.zeros(total_elements)
        #xx[0::self.ratio] = x.view(-1)
        ##
        ##Faster for gpu
        xx = x.repeat_interleave(self.ratio)
        for i in range(self.ratio-1):
            xx[1+i::self.ratio]=0.
        xx = self.ratio*xx
        x= self.lowpass(xx.view(new_shape))
        return x

class DownSample(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassFilter(cutoff=0.5/ratio, half_width=0.6/ratio, 
                                     stride= ratio,kernel_size=12)

    def forward(self, x):
        xx = self.lowpass(x)
        return xx


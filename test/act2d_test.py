import torch
import matplotlib.pyplot as plt
from src.alias_free_torch.act import Activation2d
from src.alias_free_torch.resample import UpSample2d, DownSample2d
from src.alias_free_torch.filter import LowPassFilter2d
import math
continuous_ratio = 16
ratio = 2
size = 256
center = [0, 1.5 + 10 / 5.]
t = (torch.stack(torch.meshgrid(
    (torch.arange(-size, size) - center[0]) / size,
    (torch.arange(-size, size) - center[1]) / size),
                 dim=-1)) * 10.1  * math.pi

t = torch.sin(torch.cos(torch.norm(t, dim=-1, p=2))) #+ 0.01 * torch.cos(        (torch.arange(-size, size) - center[1]) / 1.5 * math.pi).unsqueeze(0)
act = Activation2d(torch.nn.Sigmoid(), ratio, ratio)
act_t = act(15 * t.unsqueeze(0)).squeeze(0)
upsample = UpSample2d(ratio=ratio * ratio)
downsample = DownSample2d(ratio=ratio)
low = LowPassFilter2d(0.5 / ratio,
                      0.6 / ratio,
                      kernel_size=int(ratio // 2) * 12)
upsample_disc = UpSample2d(ratio=continuous_ratio)
downsample_disc = DownSample2d(ratio=continuous_ratio)
plt.figure(figsize=(9, 6))
plt.subplot(2, 3, 1)
plt.pcolor(t.numpy(), vmin=-1., vmax=2.)
plt.gca().axis('off')
plt.tight_layout()
plt.subplot(2, 3, 2)
plt.pcolor(torch.sigmoid(15 * t).numpy(), vmin=-1.2, vmax=1.2)
plt.gca().axis('off')
plt.tight_layout()
plt.subplot(2, 3, 3)
plt.pcolor(low(
    (torch.sigmoid(15 * t.unsqueeze(0)).squeeze(0))).numpy(),
           vmin=-1.2,
           vmax=1.2)
plt.gca().axis('off')
plt.tight_layout()

plt.subplot(2, 3, 4)
plt.pcolor((t[::ratio, ::ratio].unsqueeze(0)).squeeze(0).numpy(),
           vmin=-1.,
           vmax=2.)
plt.gca().axis('off')
plt.tight_layout()
plt.subplot(2, 3, 5)
plt.pcolor(torch.sigmoid(
    15 * (t[::ratio, ::ratio].unsqueeze(0))).squeeze(0).numpy(),
           vmin=-1.2,
           vmax=1.2)
plt.gca().axis('off')
plt.tight_layout()
plt.subplot(2, 3, 6)
plt.pcolor((downsample(
    torch.sigmoid(15 * upsample(
        (t[::ratio, ::ratio].unsqueeze(0)))))).squeeze(0).numpy(),
           vmin=-1.2,
           vmax=1.2)
plt.gca().axis('off')
plt.tight_layout()
plt.show()

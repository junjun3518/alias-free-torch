import torch
import matplotlib.pyplot as plt
from src.alias_free_torch.resample import DownSample2d

ratio = 2
size = 80
t = (torch.stack(torch.meshgrid(
    torch.arange(-size, size) + 0.5,
    torch.arange(-size, size) + 0.5),
                 dim=-1)) / size * 2 * 3.141592
t = torch.norm(t, dim=-1, p=1)
tt = (torch.stack(torch.meshgrid(
    torch.arange(-size // ratio, size // ratio) + 0.5,
    torch.arange(-size // ratio, size // ratio) + 0.5),
                  dim=-1)) / size * ratio * 2 * 3.141592
tt = torch.norm(tt, dim=-1, p=1)

orig_sin = torch.sin(t) + torch.sin(t * 2)
real_down_sin = torch.sin(tt) + torch.sin(tt * 2)
downsample = DownSample2d(ratio)
down_sin = downsample(orig_sin.unsqueeze(0)).squeeze(0)
print(orig_sin.shape, down_sin.shape, real_down_sin.shape)

plt.figure(figsize=(7, 9))
plt.suptitle(f'downsample /{ratio}')
plt.subplot(4, 1, 1)
plt.gca().set_title('original')
plt.pcolor(orig_sin.numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.subplot(4, 1, 2)
plt.gca().set_title('real down')
plt.pcolor(real_down_sin.numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.subplot(4, 1, 3)
plt.gca().set_title('downsampled')
plt.pcolor(down_sin.numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.subplot(4, 1, 4)
plt.gca().set_title('Error')
plt.pcolor((real_down_sin - down_sin).numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.show()

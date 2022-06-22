import torch
import matplotlib.pyplot as plt
from src.alias_free_torch.resample import UpSample2d
from src.alias_free_torch.filter import LowPassFilter2d

ratio = 8
size = 40
t = (torch.stack(torch.meshgrid(
    torch.arange(-size, size) + 0.5,
    torch.arange(-size, size) + 0.5),
                 dim=-1)) / size * 2 * 3.141592
t = torch.norm(t, dim=-1, p=1)
tt = (torch.stack(torch.meshgrid(
    torch.arange(-ratio * size, ratio * size) + 0.5,
    torch.arange(-ratio * size, ratio * size) + 0.5),
                  dim=-1)) / size / ratio * 2 * 3.141592
tt = torch.norm(tt, dim=-1, p=1)


orig_sin = torch.sin(t) + torch.sin(2*t)
real_up_sin = torch.sin(tt) + torch.sin(2*tt)
upsample = UpSample2d(ratio)
#print(upsample(orig_sin.view(1,1,*orig_sin.shape)).shape)
up_sin = upsample(orig_sin.view(1,1,*orig_sin.shape)).view(2*size*ratio,2*size*ratio)

plt.figure(figsize=(7, 9))
plt.suptitle(f'upsample x{ratio}')
plt.subplot(4, 1, 1)
plt.gca().set_title('original')
plt.pcolor(orig_sin.numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.subplot(4, 1, 2)
plt.gca().set_title('real up')
plt.pcolor(real_up_sin.numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.subplot(4, 1, 3)
plt.gca().set_title('upsampled')
plt.pcolor(up_sin.numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.subplot(4, 1, 4)
plt.gca().set_title('Error')
plt.pcolor((real_up_sin - up_sin).numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.show()

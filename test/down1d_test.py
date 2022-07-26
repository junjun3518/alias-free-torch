import torch
import matplotlib.pyplot as plt
from src.alias_free_torch.resample import DownSample1d

ratio = 10
t = torch.arange(100) / 100. * 3.141592
tt = torch.arange(100 // ratio) / (100. / ratio) * 3.141592
t, tt = t.view(1,1,-1), tt.view(1,1,-1)
orig_sin = torch.sin(t) + torch.sin(t * 2)
real_down_sin = torch.sin(tt) + torch.sin(tt * 2)
downsample = DownSample1d(ratio)
down_sin = downsample(orig_sin)

plt.figure(figsize=(7, 5))
plt.suptitle(f'downsample /{ratio}')
plt.subplot(4, 1, 1)
plt.gca().set_title('original')
plt.plot(t.view(-1).numpy(), orig_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(4, 1, 2)
plt.gca().set_title('real down')
plt.plot(tt.view(-1).numpy(), real_down_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(4, 1, 3)
plt.gca().set_title('downsampled')
plt.plot(tt.view(-1).numpy(), down_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(4, 1, 4)
plt.gca().set_title('Error')
plt.plot(tt.view(-1).numpy(), (real_down_sin - down_sin).view(-1).numpy())
plt.tight_layout()
plt.show()

import torch
import matplotlib.pyplot as plt
from ..filter import LowPassFilter2d

ratio = 2

size = 20
t = (torch.stack(torch.meshgrid(
    torch.arange(-size, size) + 0.5,
    torch.arange(-size, size) + 0.5),
                 dim=-1)) / size * 2 * 3.141592
t = torch.norm(t, dim=-1, p=1)
low = LowPassFilter2d(cutoff=1 / ratio, half_width=0.6 / ratio)

orig_sin = torch.sin(t) + torch.cos(t * 2)
#orig_sin = torch.randn(400)
filter_sin = low(orig_sin)
plt.figure(figsize=(7, 10))
plt.suptitle(f'filter cutoff{1/ratio}')
plt.subplot(3, 1, 1)
plt.gca().set_title('original')
plt.pcolor(orig_sin.numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.subplot(3, 1, 2)
plt.gca().set_title('filtered')
plt.pcolor(filter_sin.numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.subplot(3, 1, 3)
plt.gca().set_title('Difference, original - filtered')
plt.pcolor((filter_sin - orig_sin).numpy(), vmin=-2.5, vmax=2.5)
plt.tight_layout()
plt.show()

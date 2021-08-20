import torch
import matplotlib.pyplot as plt
from ..resample import UpSample1d
from ..filter import LowPassFilter1d

ratio = 2
t = torch.arange(100) / 10. * 3.141592
tt = torch.arange(100 * ratio) / (10. * ratio) * 3.141592
#low = LowPassFilter1d(cutoff = 0.5/ratio/ratio,
#                    half_width = 0.6/ratio/ratio)

orig_sin = torch.sin(t) + torch.sin(t * 2)
real_up_sin = torch.sin(tt) + torch.sin(tt * 2)
upsample = UpSample1d(ratio)
print(upsample.lowpass.filter)
up_sin = (upsample(orig_sin))
#up_sin = low(upsample(orig_sin))

plt.figure(figsize=(7, 5))
plt.suptitle(f'upsample x{ratio}')
plt.subplot(4, 1, 1)
plt.gca().set_title('original')
plt.plot(t, orig_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(4, 1, 2)
plt.gca().set_title('real up')
plt.plot(tt, real_up_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(4, 1, 3)
plt.gca().set_title('upsampled')
plt.plot(tt, up_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(4, 1, 4)
plt.gca().set_title('Error')
plt.plot(tt, (real_up_sin - up_sin).view(-1).numpy())
plt.tight_layout()
plt.show()

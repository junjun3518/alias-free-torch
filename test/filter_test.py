import torch
import matplotlib.pyplot as plt
from filter import LowPassFilter

ratio = 2
t = torch.arange(400)/40.*3.141592 
low = LowPassFilter(cutoff = 1/ratio,
                    half_width = 0.6/ratio)

#orig_sin = torch.sin(t) +torch.sin(t*10)
orig_sin = torch.randn(400)
filter_sin = low(orig_sin)

plt.figure(figsize = (7,5))
plt.suptitle(f'filter cutoff{1/ratio}')
plt.subplot(3,1,1)
plt.gca().set_title('original')
plt.plot(t,orig_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(3,1,2)
plt.gca().set_title('filtered')
plt.plot(t,filter_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(3,1,3)
plt.gca().set_title('Difference, original - filtered')
plt.plot(t,(filter_sin - orig_sin).view(-1).numpy())
plt.tight_layout()
plt.show()

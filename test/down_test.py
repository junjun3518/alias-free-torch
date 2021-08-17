import torch
import matplotlib.pyplot as plt
from resample import DownSample

ratio = 10
t = torch.arange(1000)/500.*3.141592 
tt = torch.arange(1000//ratio)/(500./ratio)*3.141592

orig_sin = torch.sin(t) +torch.sin(t*2)
real_down_sin = torch.sin(tt)+torch.sin(tt*2)
downsample = DownSample(ratio)
down_sin = downsample(orig_sin)

plt.figure(figsize = (7,5))
plt.suptitle(f'downsample /{ratio}')
plt.subplot(4,1,1)
plt.gca().set_title('original')
plt.plot(t,orig_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(4,1,2)
plt.gca().set_title('real down')
plt.plot(tt,real_down_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(4,1,3)
plt.gca().set_title('downsampled')
plt.plot(tt,down_sin.view(-1).numpy())
plt.tight_layout()
plt.subplot(4,1,4)
plt.gca().set_title('Error')
plt.plot(tt,(real_down_sin - down_sin).view(-1).numpy())
plt.tight_layout()
plt.show()

# Alias-Free-Torch

Simple torch module implementation of [Alias-Free GAN](https://nvlabs.github.io/alias-free-gan/).

This repository including
- Alias-Free GAN style lowpass sinc filter @[filter.py](/filter.py)

- Alias-Free GAN style up/downsample @[resample.py](/resample.py)

- Alias-Free activation @[act.py](/act.py)

- and test codes @[./test](/test)

**Note: Since this repository is unofficial, filter and upsample could be different with official implementation.**
**Note: 2d lowpass filter is applying sinc instead of jinc (first order Bessel function of the first kind) in paper**

## TODO
- [x] 2d sinc filter
- [x] 2d resample
- [x] devide 1d and 2d modules
- [ ] pip packaging
- [ ] demo like [official page](https://nvlabs.github.io/alias-free-gan/)
- [ ] 3d or n dim?

## Test results 1d
| Filter sine              | Filter noise                  |
| --------------------- | --------------------------- |
| ![filtersin](asset/filtersin.png) | ![filternoise](asset/filternoise.png) |

| upsample              | downsample                  |
| --------------------- | --------------------------- |
| ![up2](asset/up2.png) | ![down10](asset/down10.png) |
| ![up256](asset/up256.png) | ![down100](asset/down100.png) |

## Test results 2d
| Filter L1 norm sine   | Filter noise                  |
| --------------------- | --------------------------- |
| ![filter2dsin](asset/filter2dsin.png) | ![filter2dnoise](asset/filter2dnoise.png) |

| upsample              | downsample                  |
| --------------------- | --------------------------- |
| ![up2d2](asset/up2d2.png) | ![downsample2d2](asset/downsample2d2.png) |
| ![up2d8](asset/up2d8.png) | ![downsample2d4](asset/downsample2d4.png) |



## References

- [Alias-Free GAN](https://nvlabs-fi-cdn.nvidia.com/alias-free-gan/alias-free-gan-paper.pdf)
- [adefossez](https://github.com/adefossez)/[julius](https://github.com/adefossez/julius)
- A. V. Oppenheim and R. W. Schafer. *Discrete-Time Signal Processing*. Pearson, International Edition, 3rd edition, 2010

## Acknowledgement

This work is done at [MINDsLab Inc](https://github.com/mindslab-ai).

Thanks to teammates at MINDsLab Inc.
- [Hyoungkyu Song](https://github.com/deepkyu)
- [Kang-wook Kim](https://github.com/wookladin)
- [Dongho Choi](https://github.com/dhchoi99)
- [Sang Hoon Woo](https://github.com/tonyswoo)


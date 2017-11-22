# time-lagged autoencoder

A toolbox for dimension reduction of time series data with a [time-lagged autoencoder](https://arxiv.org/abs/1710.11239)-type deep neural network.

## Development system:
| package | version | | channel |
|:---|:---|:---|:---|
| python | 3.6.1 | 2 | |
| conda | 4.3.29 | py36_0 | conda-forge |
| numpy | 1.13.3 | py36_blas_openblas_200 [blas_openblas] | conda-forge |
| pytorch | 0.2.0 | py36_4cu75 | soumith |
| pyemma | 2.4 | np113py36_1 | conda-forge |

## Installation
Make sure to install pytorch via conda, instructions on http://pytorch.org, before you install the present module with

```bash
python setup.py test
python setup.py install
```

## Methods
This package implements
- principal component analysis (PCA),
- time-lagged independent component analysis (TICA),
- time-lagged canonical correlation analysis (via TICA),
- kinetic maps (via TICA), and
- an autoencoder-type neural network (AE) trained in a time-lagged manner.


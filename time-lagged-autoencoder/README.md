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

To run the included benchmarks, you also need to install the packages [pyemma](https://github.com/markovmodel/pyemma) and [mdshare](https://github.com/markovmodel/mdshare).

## Methods
This package implements
- principal component analysis (PCA),
- time-lagged independent component analysis (TICA),
- time-lagged canonical correlation analysis (via TICA),
- kinetic maps (via TICA), and
- an autoencoder-type neural network (AE) trained in a time-lagged manner.

## Example
Assume that ``data`` is a single ``numpy.ndarray(shape=[n_frames, n_features])`` object, ``n_frames`` refers to the number of timesteps in the trajectory, and ``n_features`` refers to the number of features extracted from the original molecular dyamics (MD) data.

### Step 0 - imports
```python
import tae
from torch.utils.data import DataLoader
```

### Step 1 - create dataset objects
Choose a lag time ``lag`` and run
```python
data_0 = tae.utils.create_dataset(data, lag=0)
data_lag = tae.utils.create_dataset(data, lag=lag)
```
You now have a normal an a time-lagged view on the original data.

### Step 2 - split into training and test sets
Choose the fraction ``f_active`` of samples to go into the training set and run
```python
data_train, data_test = tae.utils.random_split(data_lag, f_active=f_active)
```
You now have splitted the time-lagged view into two distinct sets.

### Step 3 - create data loaders
Choose a batch size ``batch_size`` and run
```python
loader_0 = DataLoader(data_0, batch_size=batch_size)
loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(data_test, batch_size=batch_size)
```
You now have loaders for all three sets to handle the minibatch processing.

### Step 4 - train the time-lagged autoencoder
Choose a latent dimensionality ``lat_dim`` and run
```python
model = tae.AE(n_features, lat_dim, hid_size=[100])
train_error, test_error = model.fit(loader_train, n_epochs=100, test_loader=loader_test)
```
You now have a trained model and you can visualize the training and test performance using the returned errors. The choice ``n_epochs=100`` requests 100 runs over the training set during the training stage and ``hid_size=[100]`` requests one hidden layer of size 100 between the input and encoded layers, as well as between encoded and output layers.

### Step 5 - obtain the dimension-reduced data
```python
data_lat = model.transform(loader_0).numpy()
```
You now have a transformed trajectory as a ``numpy.ndarray(shape=[n_frames, lat_dim])`` object.

## Citation
```
@article{time-lagged-autoencoder-arxiv,
    Archiveprefix = {arXiv},
    Author = {{Wehmeyer}, C. and {No{\'e}}, F.},
    Eprint = {1710.11239},
    Journal = {ArXiv e-prints},
    Month = oct,
    Primaryclass = {stat.ML},
    Title = {{Time-lagged autoencoders: Deep learning of slow collective variables for molecular kinetics}},
    Year = 2017}
```

# time-lagged autoencoder

A toolbox for dimension reduction of time series data with a [time-lagged autoencoder](https://arxiv.org/abs/1710.11239)-type deep neural network.

## Methods
This package implements
- principal component analysis (PCA),
- time-lagged independent component analysis (TICA),
- time-lagged canonical correlation analysis (via TICA),
- kinetic maps (via TICA), and
- an autoencoder-type neural network (AE) trained in a time-lagged manner.

## Example
Assume that `data` is a single `numpy.ndarray(shape=[n_frames, n_features])` object or list thereof, `n_frames` refers to the number of timesteps in the/each trajectory, and `n_features` refers to the number of features extracted from the original molecular dyamics (MD) data. Now choose a target dimensionality `dim` and a transformation lag time `lag`, and run:

```python
import tae

# run PCA
pca_transformed_data, pca_train_loss, pca_val_loss = tae.pca(data, dim=dim)

# run TICA
tica_transformed_data, tica_train_loss, tica_val_loss = tae.tica(data, dim=dim, lag=lag)

# run AE
ae_transformed_data, ae_train_loss, ae_val_loss = tae.ae(data, dim=dim, lag=lag)

# run VAE
vae_transformed_data, vae_train_loss, vae_val_loss = tae.vae(data, dim=dim, lag=lag)

# run AE on a GPU
ae_transformed_data, ae_train_loss, ae_val_loss = tae.ae(data, dim=dim, lag=lag, cuda=True)
```

In this example, we get `*_val_loss=None` because we are training on the full data set. To exclude a randomly chosen fraction `fval` of the data from the training, add the parameter `validation_split=fval` to the function calls, e.g.:

```python
ae_transformed_data, ae_train_loss, ae_val_loss = tae.ae(
    data, dim=dim, lag=lag, validation_split=fval, cuda=True)
```

## Citation
```
@article{time-lagged-autoencoder,
	Author = {Christoph Wehmeyer and Frank No{\'{e}}},
	Doi = {10.1063/1.5011399},
	Journal = {J. Chem. Phys.},
	Month = {jun},
	Number = {24},
	Pages = {241703},
	Publisher = {{AIP} Publishing},
	Title = {Time-lagged autoencoders: Deep learning of slow collective variables for molecular kinetics},
	Volume = {148},
	Year = {2018}}
```

## Development system
This project was developed using the following python environment:

| package | version | | channel |
|:---|:---|:---|:---|
| python | 3.6.1 | 2 | |
| conda | 4.3.29 | py36_0 | conda-forge |
| numpy | 1.13.3 | py36_blas_openblas_200 [blas_openblas] | conda-forge |
| pytorch | 0.2.0 | py36_4cu75 | soumith |
| pyemma | 2.4 | np113py36_1 | conda-forge |

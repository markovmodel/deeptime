# deeptime
Deep learning meets molecular dynamics.

## Contents

- **tae**: a toolbox for dimension reduction of time series data with a [time-lagged autoencoder](https://aip.scitation.org/doi/full/10.1063/1.5011399)-type deep neural network.
- **vampnet**: Variational Approach for Markov Processes networks, see https://www.nature.com/articles/s41467-017-02388-1

## Dependencies

`deeptime.vampnet` requires [Tensorflow 1.4-1.6](https://www.tensorflow.org) to be used.
Please install either tensorflow or tensorflow-gpu. Installation instructions: https://www.tensorflow.org/install/.

**IMPORTANT**: we're currently investigating an issue with Tensorflow 1.7 which causes the eigenvalue decomposition to fail. This issue doesn't present itself on TF 1.4-1.6, so until this is resolved please use one of these older releases instead.

`deeptime.tae` requires PyTorch >= 0.4.0. Please refer to http://pytorch.org for detailed installation instructions.

To run the included benchmarks and example notebooks, you also need to install the packages [pyemma](https://github.com/markovmodel/pyemma) and [mdshare](https://github.com/markovmodel/mdshare):

```bash
conda install mdshare pyemma -c conda-forge
```

or

```bash
pip install mdshare pyemma
```

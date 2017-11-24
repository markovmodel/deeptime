# VAMPnet
Variational Approach for Markov Processes networks.


## What is it?
VAMPnet is an open source Python package for the implementation of the VAMPnet method for dynamical systems analysis (described in https://arxiv.org/abs/1710.06012). It includes losses functions, metrics, basic estimators for Koopman operators and the most important validation tools for Koopman models.

VAMPnet can be used from Jupyter (former IPython, recommended), or by
writing Python scripts.


## Citation
If you use VAMPnet in scientific work, please cite:

    Mardt, A., Pasquali, L., Wu, H., & Noé, F. (2017).
    VAMPnets: Deep learning of molecular kinetics.
    arXiv preprint arXiv:1710.06012.


## Installation
First clone the repository, then with pip:

```bash
pip install .
```

## Notes
This package requires [Tensorflow 1.4](https://www.tensorflow.org) to be used.

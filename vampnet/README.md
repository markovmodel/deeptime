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
This package requires [Tensorflow 1.4](https://www.tensorflow.org) to be used.
Please install either tensorflow or tensorflow-gpu. Installation instructions:

https://www.tensorflow.org/install/

To install this package, first clone the repository:

git clone https://github.com/markovmodel/deeptime.git

Then with pip:

```bash
python setup.py install
```

The examples are jupyter notebooks, so the jupyter package is needed to run them:

http://jupyter.readthedocs.io/en/latest/install.html

as well as keras:

https://keras.io/#installation

This is not needed if you'd like to use the package only.


If you want to run the alanine dipeptide example, you'll also need to install the mdshare package (necessary for the download of the trajectory files):

git clone https://github.com/markovmodel/mdshare.git
pip install ./mdshare


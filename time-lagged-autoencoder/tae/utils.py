#   This file is part of the markovmodel/deeptime repository.
#   Copyright (C) 2017 Computational Molecular Biology Group,
#   Freie Universitaet Berlin (GER)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Tools to handle datasets, transformations, and statistics.
'''

import numpy as _np
import torch as _torch
from torch import nn as _nn
from torch.utils.data import Dataset as _Dataset
from torch.utils.data import TensorDataset as _TensorDataset
from torch.utils.data import ConcatDataset as _ConcatDataset
from torch.utils.data import DataLoader as _DataLoader
from torch.autograd import Variable as _Variable

__all__ = [
    'LaggedDataset',
    'MaskedDataset',
    'create_dataset',
    'stride_split',
    'random_split',
    'get_mean',
    'get_covariance',
    'get_sqrt_inverse',
    'whiten_data',
    'cca',
    'Transform']

################################################################################
#
#   DATASETS
#
################################################################################

class LaggedDataset(_Dataset):
    '''Dataset for wrapping time-lagged data from a single stored time series.

    Each sample will contain the data_tensor at index t and the (not explicitly
    stored) target_tensor via data_tensor at index t+lag. We need this for
    training the time-lagged autoencoder and TICA.

    Arguments:
        data_tensor (Tensor): contains time series data
        lag (int): specifies the lag in time steps
    '''
    def __init__(self, data_tensor, lag=1):
        assert data_tensor.size(0) > lag, 'you need more samples than lag'
        assert lag >= 0, 'you need a non-negative lagtime'
        self.data_tensor = data_tensor
        self.lag = lag
    def __getitem__(self, index):
        return self.data_tensor[index], self.data_tensor[index + self.lag]
    def __len__(self):
        return self.data_tensor.size(0) - self.lag

class MaskedDataset(_Dataset):
    '''Dataset for wrapping a specified subset of another dataset.

    This helps to separate a dataset into two or more subsets, e.g., for
    training and testing.

    Arguments:
        data_tensor (Tensor): contains time series data
        active (sequence of int): indices of the active elements
    '''
    def __init__(self, dataset, active):
        assert len(dataset) >= len(active), \
            'you cannot have less total samples than active'
        assert _np.all(0 <= active) and _np.all(active < len(dataset)), \
            'you must use only valid indices'
        assert len(active) == len(_np.unique(active)), \
            'you must use every active index only once'
        self.dataset = dataset
        self.active = active
    def __getitem__(self, index):
        return self.dataset[self.active[index]]
    def __len__(self):
        return len(self.active)

def ensure_traj_format(data, dtype=_np.float32):
    data = _np.asarray(data, dtype=dtype)
    if data.ndim == 2:
        return data
    elif data.ndim == 1:
        return data.reshape(-1, 1)
    else:
        raise ValueError('data has incomplatible ndim: ' + str(data.ndim))

def create_dataset(data, lag=0, dtype=_np.float32):
    if isinstance(data, _np.ndarray):
        return LaggedDataset(
            _torch.from_numpy(ensure_traj_format(data, dtype=dtype)),
            lag=lag)
    elif isinstance(data, (list, tuple)):
        return _ConcatDataset([LaggedDataset(
            _torch.from_numpy(ensure_traj_format(d, dtype=dtype)),
            lag=lag) for d in data])
    else:
        raise ValueError(
            'use a single or a list of numpy.ndarrays of dim 1 or 2')

def stride_split(dataset, stride=2, offset=0):
    '''Split one dataset into two parts based on a stride.

    This helps to separate a dataset into two or more subsets, e.g., for
    training and testing. Every <stride>th element starting from <offset>
    goes into the first MaskedDataset, everything else into the second.

    Arguments:
        dataset (Dataset): contains the data you want to split
        stride (int): specify the size of the stride
        offset (int): specify where to start counting
    '''
    assert 0 < stride < len(dataset), \
        'use a positive stride smaller than the length of the dataset'
    assert 0 <= offset < stride, \
        'use a non-negative offset smaller than the stride'
    active = _np.arange(offset, len(dataset), stride)
    complement = _np.setdiff1d(
        _np.arange(len(dataset)), active, assume_unique=True)
    return MaskedDataset(dataset, active), MaskedDataset(dataset, complement)

def random_split(dataset, active=None, n_active=None, f_active=None):
    '''Split one dataset into two parts based on a random selction.

    This helps to separate a dataset into two or more subsets, e.g., for
    training and testing. Specify the active set either by giving the frame
    indices, the number of active frames or the fraction of active frames.

    Arguments:
        dataset (Dataset): contains the data you want to split
        active (iterable of int): specify the active frames
        n_active (int): number of active frames
        f_active (int): fraction of active frames
    '''
    if active is None:
        if n_active is None:
            if f_active is None:
                raise ValueError(
                    'specify either active, n_active or f_active')
            else:
                assert 0 < f_active < 1, \
                    'f_active must be 0 < f_active < 1'
            n_active = int(_np.floor(0.5 + f_active * len(dataset)))
        else:
            assert 0 < n_active < len(dataset), \
                'n_active must be 0 < n_active < len(dataset)'
            if f_active is not None:
                raise ValueError(
                    'do not specify f_active if n_active is given')
        active = _np.random.choice(len(dataset), size=n_active, replace=False)
    else:
        assert len(active) == len(_np.unique(active)), \
            'you must use every active index only once'
        assert _np.all(0 <= active < len(dataset)), \
            'you must use only valid indices'
        if f_active is not None:
            raise ValueError(
                'do not specify f_active if active is given')
        if n_active is not None:
            raise ValueError(
                'do not specify n_active if active is given')
    complement = _np.setdiff1d(
        _np.arange(len(dataset)), active, assume_unique=True)
    return MaskedDataset(dataset, active), MaskedDataset(dataset, complement)

################################################################################
#
#   STATISTICS
#
################################################################################

def get_mean(loader):
    '''Compute the mean value via minibatch summation using a loader.

    Arguments:
        loader (DataLoader): contains the data you want to analyze
    '''
    x_mean, y_mean = None, None
    for x, y in loader:
        try:
            x_mean.add_(x.sum(dim=0))
        except AttributeError:
            x_mean = x.sum(dim=0)
        try:
            y_mean.add_(y.sum(dim=0))
        except AttributeError:
            y_mean = y.sum(dim=0)
    x_mean.div_(float(len(loader.dataset)))
    y_mean.div_(float(len(loader.dataset)))
    return x_mean, y_mean

def get_covariance(loader, x_mean, y_mean):
    '''Compute the instantaneous and time-lagged covariance matrices via
    minibatch summation using a loader.

    Arguments:
        loader (DataLoader): contains the data you want to analyze
        x_mean (Tensor): mean value for the data_tensor
        y_mean (Tensor): mean value for the target_tensor
    '''
    cxx = _torch.zeros(len(x_mean), len(x_mean))
    cxy = _torch.zeros(len(x_mean), len(y_mean))
    cyy = _torch.zeros(len(y_mean), len(y_mean))
    for x, y in loader:
        x.sub_(x_mean[None, :])
        y.sub_(y_mean[None, :])
        cxx.add_(_torch.mm(x.t(), x))
        cxy.add_(_torch.mm(x.t(), y))
        cyy.add_(_torch.mm(y.t(), y))
    cxx.div_(float(len(loader.dataset)))
    cxy.div_(float(len(loader.dataset)))
    cyy.div_(float(len(loader.dataset)))
    return cxx, cxy, cyy

################################################################################
#
#   WHITENING
#
################################################################################

def get_sqrt_inverse(matrix, bias=1.0e-5):
    '''Compute the sqrt-inverse of the supplied symmetric/real matrix.

    We need this step for whitening and TICA.

    Arguments:
        matrix (Tensor): contains the matrix you want to transform
        bias (float): assures numerical stability
    '''
    e, v = _torch.symeig(matrix, eigenvectors=True)
    d = _torch.diag(1.0 / _torch.sqrt(_torch.abs(e) + bias))
    return _torch.mm(_torch.mm(v, d), v.t())

def whiten_data(data_tensor, batch_size=100):
    '''Whiten a Tensor in the PCA basis.

    Arguments:
        data_tensor (Tensor): contains the data you want to whiten
        batch_size (int): specify a batch size for the whitening process
    '''
    loader = _DataLoader(
        LaggedDataset(data_tensor, lag=0), batch_size=batch_size)
    x_mean, y_mean = get_mean(loader)
    cxx, cxy, cyy = get_covariance(loader, x_mean, y_mean)
    ixx = get_sqrt_inverse(cxx)
    whitened_data = []
    for x, _ in loader:
        x.sub_(x_mean[None, :])
        whitened_data.append(x.mm(ixx))
    return _torch.cat(whitened_data)

################################################################################
#
#   CCA
#
################################################################################

def cca(data_tensor_x, data_tensor_y, batch_size=100):
    '''Perform canonical correlation analysis for two data tensors.

    Arguments:
        data_tensor_x (Tensor): contains the first data tensor
        data_tensor_y (Tensor): contains the second data tensor
        batch_size (int): specify a batch size for the CCA calculation
    '''
    loader = _DataLoader(
        _TensorDataset(data_tensor_x, data_tensor_y),
        batch_size=batch_size)
    x_mean, y_mean = get_mean(loader)
    cxx, cxy, cyy = get_covariance(loader, x_mean, y_mean)
    ixx = get_sqrt_inverse(cxx)
    iyy = get_sqrt_inverse(cyy)
    return _torch.svd(_torch.mm(_torch.mm(ixx, cxy), iyy))

################################################################################
#
#   TRANSFORMER
#
################################################################################

class BaseTransform(object):
    def __init__(self, mean=None, covariance=None):
        if mean is not None:
            self.sub = mean
        if covariance is not None:
            self.mul = get_sqrt_inverse(covariance)
    def __call__(self, x, variable=False, **kwargs):
        try:
            x.sub_(self.sub[None, :])
        except AttributeError:
            pass
        try:
            x = x.mm(self.mul)
        except AttributeError:
            pass
        if variable:
            return _Variable(x, **kwargs)
        return x

class Transform(object):
    '''Apply whitening/centering transformations within a minibatch.

    As we do not want to preprocess and, thus, duplicate large datasets,
    we do the necessary whitening and centering operations on the fly while
    iterating over the datasets.

    Arguments:
        x_mean (Tensor): contains the mean of the data tensor
        x_covariance (Tensor): contains the covariance of the data tensor
        y_mean (Tensor): contains the mean of the target tensor
        y_covariance (Tensor): contains the covariance of the target tensor
    '''
    def __init__(
        self, x_mean=None, x_covariance=None, y_mean=None, y_covariance=None):
        self.x = BaseTransform(mean=x_mean, covariance=x_covariance)
        self.y = BaseTransform(mean=y_mean, covariance=y_covariance)
    def __call__(self, x, y, variable=False, train=False):
        return self.x(
                x, variable=variable,
                volatile=not train, requires_grad=train), \
            self.y(
                y, variable=variable,
                volatile=not train, requires_grad=False)



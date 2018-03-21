#   This file is part of the markovmodel/deeptime repository.
#   Copyright (C) 2017, 2018 Computational Molecular Biology Group,
#   Freie Universitaet Berlin (GER)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import torch
from torch.utils.data import DataLoader
from ..utils import LaggedDataset
from ..utils import MaskedDataset
from ..utils import ensure_traj_format
from ..utils import create_dataset
from ..utils import stride_split
from ..utils import random_split
from ..utils import random_block_split
from ..utils import get_mean
from ..utils import get_covariance
from ..utils import get_sqrt_inverse
from ..utils import whiten_data
from ..utils import cca
from ..utils import BaseTransform
from ..utils import Transform

################################################################################
#
#   DATASETS
#
################################################################################

def test_lagged_dataset_at_default_lag():
    data = np.arange(
        800 + np.random.randint(200)).reshape(-1, 1).astype(np.float32)
    dataset = LaggedDataset(torch.Tensor(data), lag=0)
    for x, y in dataset:
        assert x[0] == y[0]

def test_lagged_dataset_at_lag0():
    data = np.arange(
        800 + np.random.randint(200)).reshape(-1, 1).astype(np.float32)
    dataset = LaggedDataset(torch.Tensor(data), lag=0)
    for x, y in dataset:
        assert x[0] == y[0]

def test_lagged_dataset_at_random_lag():
    data = np.arange(
        800 + np.random.randint(200)).reshape(-1, 1).astype(np.float32)
    lag = 1 + np.random.randint(50)
    dataset = LaggedDataset(torch.Tensor(data), lag)
    for x, y in dataset:
        assert x[0] + lag == y[0]

def test_masked_dataset():
    data = np.arange(
        800 + np.random.randint(200)).reshape(-1, 1).astype(np.float32)
    active = np.random.choice(data[:, 0], size=100, replace=False)
    dataset = MaskedDataset(LaggedDataset(torch.Tensor(data), lag=0), active)
    assert len(dataset) == len(active)
    for (x, y), z in zip(dataset, active):
        assert x[0] == y[0] == z

def test_ensure_traj_format_1d():
    raw_data = np.arange(800 + np.random.randint(200))
    data = ensure_traj_format(raw_data)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert data.ndim == 2
    np.testing.assert_array_equal(data.shape, [len(raw_data), 1])
    np.testing.assert_allclose(raw_data.astype(np.float32), data[:, 0])

def test_ensure_traj_format_2d():
    raw_data = np.arange(800 + np.random.randint(200)).reshape(-1, 1)
    data = ensure_traj_format(raw_data)
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert data.ndim == 2
    np.testing.assert_array_equal(data.shape, raw_data.shape)
    np.testing.assert_allclose(raw_data.astype(np.float32), data)

def test_create_dataset_single_file_1d():
    data = np.arange(
        800 + np.random.randint(200))
    lag = np.random.randint(50)
    dataset = create_dataset(data, lag, dtype=np.float32)
    for x, y in dataset:
        assert x[0] + lag == y[0]

def test_create_dataset_single_file_2d():
    data = np.arange(
        800 + np.random.randint(200)).reshape(-1, 1)
    lag = np.random.randint(50)
    dataset = create_dataset(data, lag, dtype=np.float32)
    for x, y in dataset:
        assert x[0] + lag == y[0]

def test_create_dataset_multiple_files_1d():
    data = [np.arange(800 + np.random.randint(200)) for _ in range(3)]
    lag = np.random.randint(50)
    dataset = create_dataset(data, lag, dtype=np.float32)
    for x, y in dataset:
        assert x[0] + lag == y[0]

def test_create_dataset_multiple_files_2d():
    data = [np.arange(
            800 + np.random.randint(200)).reshape(-1, 1) for _ in range(3)]
    lag = np.random.randint(50)
    dataset = create_dataset(data, lag, dtype=np.float32)
    for x, y in dataset:
        assert x[0] + lag == y[0]

def test_stride_split():
    data = np.arange(
        800 + np.random.randint(200)).reshape(-1, 1).astype(np.float32)
    lag = 1 + np.random.randint(50)
    dataset = LaggedDataset(torch.Tensor(data), lag)
    stride = 1 + np.random.randint(10)
    offset = np.random.randint(stride)
    dataset_a, dataset_b = stride_split(dataset, stride=stride, offset=offset)
    assert len(dataset) == len(dataset_a) + len(dataset_b)
    for x, y in dataset_a:
        assert x[0] + lag == y[0]
    for x, y in dataset_b:
        assert x[0] + lag == y[0]

def test_random_split():
    data = np.arange(
        800 + np.random.randint(200)).reshape(-1, 1).astype(np.float32)
    lag = 1 + np.random.randint(50)
    dataset = LaggedDataset(torch.Tensor(data), lag)
    dataset_a, dataset_b = random_split(dataset, f_active=0.5)
    assert len(dataset) == len(dataset_a) + len(dataset_b)
    for x, y in dataset_a:
        assert x[0] + lag == y[0]
    for x, y in dataset_b:
        assert x[0] + lag == y[0]

def test_random_block_split():
    data = np.arange(
        800 + np.random.randint(200)).reshape(-1, 1).astype(np.float32)
    lag = 1 + np.random.randint(50)
    dataset = LaggedDataset(torch.Tensor(data), lag)
    dataset_a, dataset_b = random_block_split(dataset, lag, f_active=0.5)
    assert len(dataset) == len(dataset_a) + len(dataset_b)
    for x, y in dataset_a:
        assert x[0] + lag == y[0]
    for x, y in dataset_b:
        assert x[0] + lag == y[0]

################################################################################
#
#   STATISTICS
#
################################################################################

def test_get_mean_via_normal_distribution_parameters():
    data = torch.randn(10000, 1)
    dataset = LaggedDataset(data, lag=0)
    x, y = get_mean(
        DataLoader(
            dataset, batch_size=np.random.randint(low=10, high=100)))
    np.testing.assert_allclose(x.numpy(), 0.0, atol=0.05)
    np.testing.assert_allclose(y.numpy(), 0.0, atol=0.05)

def test_get_mean_via_distribution_symmetry():
    data = torch.rand(5000, 1)
    data = torch.cat([data, -data])
    dataset = LaggedDataset(data, lag=0)
    x, y = get_mean(
        DataLoader(
            dataset, batch_size=np.random.randint(low=10, high=100)))
    np.testing.assert_allclose(x.numpy(), 0.0, atol=0.0001)
    np.testing.assert_allclose(y.numpy(), 0.0, atol=0.0001)

def test_get_mean_vs_numpy():
    data = torch.randn(10000, 1)
    dataset = LaggedDataset(data, lag=0)
    x, y = get_mean(
        DataLoader(
            dataset, batch_size=np.random.randint(low=10, high=100)))
    numpy_result = np.mean(data.numpy())
    np.testing.assert_allclose(x.numpy(), numpy_result, atol=0.0001)
    np.testing.assert_allclose(y.numpy(), numpy_result, atol=0.0001)

def test_get_covariance_via_normal_distribution_parameters():
    data = torch.randn(10000, 1)
    dataset = LaggedDataset(data, lag=0)
    xx, xy, yy = get_covariance(
        DataLoader(
            dataset, batch_size=np.random.randint(low=10, high=100)),
        torch.Tensor([0]), torch.Tensor([0]))
    np.testing.assert_allclose(xx.numpy(), 1.0, atol=0.1)
    np.testing.assert_allclose(xy.numpy(), 1.0, atol=0.1)
    np.testing.assert_allclose(yy.numpy(), 1.0, atol=0.1)

def test_get_covariance_vs_numpy():
    data = torch.randn(10000, 1)
    dataset = LaggedDataset(data, lag=0)
    xx, xy, yy = get_covariance(
        DataLoader(
            dataset, batch_size=np.random.randint(low=10, high=100)),
        torch.Tensor([0]), torch.Tensor([0]))
    numpy_result = np.var(data.numpy(), ddof=1)
    np.testing.assert_allclose(xx.numpy(), numpy_result, atol=0.0005)
    np.testing.assert_allclose(xy.numpy(), numpy_result, atol=0.0005)
    np.testing.assert_allclose(yy.numpy(), numpy_result, atol=0.0005)

################################################################################
#
#   WHITENING
#
################################################################################

def test_get_sqrt_inverse():
    dim = 2 + np.random.randint(5)
    x = torch.rand(500, dim)
    x = torch.mm(x.t(), x)
    y = get_sqrt_inverse(x)
    y = torch.mm(y, y)
    np.testing.assert_allclose(
        x.mm(y).numpy(),
        np.diag([1.0] * dim).astype(np.float32),
        atol=0.0001)

def test_whiten_data():
    dim = 1 + np.random.randint(5)
    x = whiten_data(torch.rand(500, dim))
    np.testing.assert_allclose(
        x.numpy().mean(axis=0),
        0.0,
        atol=0.01)
    np.testing.assert_allclose(
        torch.mm(x.t(), x).div_(float(x.size()[0])).numpy(),
        np.diag([1.0] * dim).astype(np.float32),
        atol=0.01)

################################################################################
#
#   CCA
#
################################################################################

def test_cca():
    s = np.arange(1000)
    x = torch.from_numpy(
        np.vstack((s, np.random.randn(s.shape[0]))).T.astype(np.float32))
    y = torch.from_numpy(
        np.vstack((np.random.randn(s.shape[0]), s)).T.astype(np.float32))
    u, s, v = cca(x, y, batch_size=100)
    np.testing.assert_allclose(s.numpy(), [1.0, 0.0], atol=0.2)
    p = u.mm(torch.diag(s).mm(v))
    np.testing.assert_allclose(
        np.abs(p.numpy()), [[0.0, 1.0], [0.0, 0.0]], atol=0.2)

################################################################################
#
#   TRANSFORMER
#
################################################################################

def test_base_transform():
    dim = 2 + np.random.randint(5)
    mean = 10.0 * (torch.rand(dim) - 0.5)
    sigma = torch.rand(dim, dim)
    sigma.add_(sigma.t())
    data = torch.randn(500, dim).mm(sigma) + mean[None, :]
    loader = DataLoader(LaggedDataset(data, lag=0), batch_size=64)
    x_mean, y_mean = get_mean(loader)
    cxx, cxy, cyy = get_covariance(loader, x_mean, y_mean)
    transformer = BaseTransform(mean=x_mean, covariance=cxx)
    transformed_data = []
    for x, _ in loader:
        transformed_data.append(transformer(x))
    y = torch.cat(transformed_data)
    np.testing.assert_allclose(
        y.numpy().mean(axis=0),
        0.0,
        atol=0.01)
    np.testing.assert_allclose(
        torch.mm(y.t(), y).div_(float(y.size()[0])).numpy(),
        np.diag([1.0] * dim).astype(np.float32),
        atol=0.2)

def test_transform():
    dim = 2 + np.random.randint(5)
    mean = 10.0 * (torch.rand(dim) - 0.5)
    sigma = torch.rand(dim, dim)
    sigma.add_(sigma.t())
    data = torch.randn(500, dim).mm(sigma) + mean[None, :]
    loader = DataLoader(LaggedDataset(data, lag=0), batch_size=64)
    x_mean, y_mean = get_mean(loader)
    cxx, cxy, cyy = get_covariance(loader, x_mean, y_mean)
    transformer = Transform(
        x_mean=x_mean, x_covariance=cxx,
        y_mean=x_mean, y_covariance=cyy)
    x_, y_ = [], []
    for x, y in loader:
        x, y = transformer(x, y)
        x_.append(x)
        y_.append(y)
    x = torch.cat(x_)
    y = torch.cat(y_)
    np.testing.assert_allclose(
        x.numpy().mean(axis=0),
        0.0,
        atol=0.1)
    np.testing.assert_allclose(
        torch.mm(x.t(), x).div_(float(x.size()[0])).numpy(),
        np.diag([1.0] * dim).astype(np.float32),
        atol=0.1)
    np.testing.assert_allclose(
        y.numpy().mean(axis=0),
        0.0,
        atol=0.1)
    np.testing.assert_allclose(
        torch.mm(y.t(), y).div_(float(y.size()[0])).numpy(),
        np.diag([1.0] * dim).astype(np.float32),
        atol=0.1)

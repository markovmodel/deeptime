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
Automatized benchmarks.
'''

import numpy as _np
from .utils import create_dataset as _create_dataset
from .utils import stride_split as _stride_split
from .utils import whiten_data as _whiten_data
from .utils import cca as _cca
from .models import PCA as _PCA
from .models import TICA as _TICA
from .models import AE as _AE
from torch import from_numpy as _from_numpy
from torch import nn as _nn
from torch.utils.data import DataLoader as _DataLoader

try:
    import pyemma
except ImportError:
    print('running benchmarks requires the pyemma package')

__all__ = [
    'Organizer',
    'run_sqrt_model_benchmark',
    'run_swissroll_model_benchmark']

################################################################################
#
#   BENCHMARK DEFAULT DISCRETIZERS
#
################################################################################

def discretize_1d_model(data):
    x = _np.linspace(data.min(), data.max(), 101)
    x = 0.5 * (x[:-1] + x[1:]).reshape(-1, 1)
    return pyemma.coordinates.assign_to_centers(data.numpy(), centers=x)

def discretize_2d_model(data):
    return pyemma.coordinates.cluster_regspace(
        data.numpy(), dmin=0.2, max_centers=400).dtrajs

################################################################################
#
#   BENCHMARK RAW DATA HANDLERS
#
################################################################################

class ToyModelWrapper(object):
    def __init__(self, model, length, discretizer):
        self.model = model
        self.length = length
        self.discretizer = discretizer
    def __call__(self, **kwargs):
        traj, dtraj = self.model(self.length)
        wdtraj = _whiten_data(
            _from_numpy(dtraj.reshape(-1, 1).astype(_np.float32)))
        return traj, wdtraj, dtraj

class MDDataWrapper(object):
    def __init__(
        self, pdb_file, files, features, length, n_trajs, discretizer):
        featurizer = pyemma.coordinates.featurizer(pdb_file)
        if 'hap' in features:
            featurizer.add_selection(featurizer.select_Heavy())
        self.discretizer = discretizer
    def __call__(self, **kwargs):
        raise NotImplementedError()

################################################################################
#
#   BENCHMARK RUNNER
#
################################################################################

class Result(object):
    def __init__(self, **data):
        self.data = dict(**data)
    def get(self, key):
        try:
            return self.data[key]
        except KeyError:
            return None
    @property
    def key(self):
        if self.lag is not None:
            return '%s_%d' % (self.method, self.lag)
        return self.method
    @property
    def method(self):
        return self.get('method')
    @property
    def lag(self):
        return self.get('lag')
    @property
    def dim(self):
        return self.get('dim')
    @property
    def train_error(self):
        try:
            return self.get('train_error')[-1]
        except TypeError:
            return self.get('train_error')
    @property
    def test_error(self):
        try:
            return self.get('test_error')[-1]
        except TypeError:
            return self.get('test_error')
    @property
    def cca(self):
        return self.get('cca')
    @property
    def its(self):
        return self.get('its')

class BenchmarkRunner(object):
    def __init__(
        self, wrapper, trns_lags, msm_lags, dim, batch_size,
        wrapper_args=dict(), tica_args=dict(), ae_args=dict()):
        self.wrapper = wrapper
        self.trns_lags = trns_lags
        self.msm_lags = msm_lags
        self.dim = dim
        self.batch_size = batch_size
        self.wrapper_args = wrapper_args
        self.tica_args = tica_args
        self.ae_args = ae_args
    def its(self, dsc_data, nits):
        return pyemma.msm.its(
            dsc_data, lags=self.msm_lags, nits=nits).timescales
    def analysis(self, model, transform_loader, ref_data):
        lat_data = _whiten_data(model.transform(transform_loader))
        cca_data = _cca(lat_data, ref_data)[1].numpy()
        dsc_data = self.wrapper.discretizer(lat_data)
        its_data = self.its(dsc_data, 5)
        return cca_data, its_data
    def reference(self, dsc_data):
        its_data = self.its(dsc_data, None)
        return Result(
            method='ref',
            dim=self.dim,
            its=its_data)
    def pca(self, train_loader, test_loader, transform_loader, ref_data):
        model = _PCA()
        train_error, test_error = model.fit(
            train_loader, dim=self.dim, test_loader=test_loader)
        lat_data = _whiten_data(model.transform(transform_loader))
        cca_data, its_data = self.analysis(model, transform_loader, ref_data)
        return Result(
            method='pca',
            dim=self.dim,
            train_error=train_error,
            test_error=test_error,
            cca=cca_data,
            its=its_data)
    def tica(self, train_loader, test_loader, transform_loader, ref_data, lag):
        model = _TICA(
            kinetic_map=self.tica_args.setdefault('kinetic_map', True),
            symmetrize=self.tica_args.setdefault('symmetrize', False))
        train_error, test_error = model.fit(
            train_loader, dim=self.dim, test_loader=test_loader)
        lat_data = _whiten_data(model.transform(transform_loader))
        cca_data, its_data = self.analysis(model, transform_loader, ref_data)
        return Result(
            method='tica',
            dim=self.dim,
            lag=lag,
            train_error=train_error,
            test_error=test_error,
            cca=cca_data,
            its=its_data)
    def ae(self, train_loader, test_loader, transform_loader, ref_data, lag):
        model = _AE(
            train_loader.dataset[0][0].size(0),
            self.dim,
            hid_size=self.ae_args.setdefault('hid_size', []),
            dropout=self.ae_args.setdefault('dropout', _nn.Dropout(p=0.5)),
            activation=self.ae_args.setdefault('activation', _nn.LeakyReLU()),
            lat_activation=self.ae_args.setdefault('lat_activation', None),
            batch_normalization=self.ae_args.setdefault(
                'batch_normalization', None),
            bias=self.ae_args.setdefault('bias', True),
            lr=self.ae_args.setdefault('lr', 0.001),
            cuda=self.ae_args.setdefault('cuda', False))
        train_error, test_error = model.fit(
            train_loader,
            self.ae_args.setdefault('n_epochs', 100),
            test_loader=test_loader)
        lat_data = _whiten_data(model.transform(transform_loader))
        cca_data, its_data = self.analysis(model, transform_loader, ref_data)
        return Result(
            method='ae',
            dim=self.dim,
            lag=lag,
            train_error=train_error,
            test_error=test_error,
            cca=cca_data,
            its=its_data)
    def __call__(self):
        data, ref_data, dsc_data = self.wrapper(**self.wrapper_args)
        results = [self.reference(dsc_data)]
        data_0 = _create_dataset(data, lag=0)
        data_0_loader = _DataLoader(
            data_0, batch_size=self.batch_size)
        data_0_test, data_0_train = _stride_split(data_0, stride=3)
        data_0_train_loader = _DataLoader(
            data_0_train, batch_size=self.batch_size, shuffle=True)
        data_0_test_loader = _DataLoader(
            data_0_test, batch_size=self.batch_size)
        results.append(
            self.pca(
                data_0_train_loader,
                data_0_test_loader,
                data_0_loader,
                ref_data))
        for lag in self.trns_lags:
            data_lag = _create_dataset(data, lag=lag)
            data_lag_test, data_lag_train = _stride_split(data_lag, stride=3)
            data_lag_train_loader = _DataLoader(
                data_lag_train, batch_size=self.batch_size, shuffle=True)
            data_lag_test_loader = _DataLoader(
                data_lag_test, batch_size=self.batch_size)
            results.append(
                self.tica(
                    data_lag_train_loader,
                    data_lag_test_loader,
                    data_0_loader,
                    ref_data,
                    lag))
            results.append(
                self.ae(
                    data_lag_train_loader,
                    data_lag_test_loader,
                    data_0_loader,
                    ref_data,
                    lag))
        return results

################################################################################
#
#   MANUSCRIPT BENCHMARKS
#
################################################################################

def organize_results(results):
    data = dict()
    for _results in results:
        for r in _results:
            if r.key in data.keys():
                data[r.key]['train_error'].append(r.train_error)
                data[r.key]['test_error'].append(r.test_error)
                data[r.key]['cca'].append(r.cca)
                data[r.key]['its'].append(r.its)
            else:
                data.update({
                    r.key: dict(
                        train_error=[r.train_error],
                        test_error=[r.test_error],
                        cca=[r.cca],
                        its=[r.its])})
    for model in data.keys():
        for key in ['train_error', 'test_error', 'cca', 'its']:
            data[model][key] = _np.asarray(data[model][key])
    return data

class Organizer(object):
    def __init__(self, data, median=True, low=16, high=84):
        self.data = data
        self.median = median
        self.low = low
        self.high = high
    def statistics(self, x, axis=None):
        if self.median:
            average = _np.median(x, axis=axis)
        else:
            average = _np.mean(x, axis=axis)
        if self.low is None:
            low = _np.min(x, axis=axis)
        else:
            low = _np.percentile(x, self.low, axis=axis)
        if self.high is None:
            high = _np.max(x, axis=axis)
        else:
            high = _np.percentile(x, self.high, axis=axis)
        return low, average, high
    def trns_lag_collector(self, model, observable):
        if model not in ['tica', 'ae']:
            raise ValueError('unknown model: ' + str(model))
        if observable not in ['train_error', 'test_error', 'cca', 'its']:
            raise ValueError('unknown observable: ' + str(observable))
        data = []
        for lag in self.data['trns_lags']:
            data.append(self.data['%s_%d' % (model, lag)][observable])
        return _np.asarray(data)
    def train_error(self, model):
        if model not in ['tica', 'ae']:
            raise ValueError('unknown model: ' + str(model))
        data = self.trns_lag_collector(model, 'train_error')
        low, average, high = self.statistics(data, axis=1)
        lags = self.data['trns_lags']
        return [lags, average], [lags, low, high]
    def test_error(self, model):
        if model not in ['tica', 'ae']:
            raise ValueError('unknown model: ' + str(model))
        data = self.trns_lag_collector(model, 'test_error')
        low, average, high = self.statistics(data, axis=1)
        lags = self.data['trns_lags']
        return [lags, average], [lags, low, high]
    def cca(self, model, idx=0):
        lags = self.data['trns_lags']
        if model in ['tica', 'ae']:
            data = self.trns_lag_collector(model, 'cca')[:, :, idx]
            low, average, high = self.statistics(data, axis=1)
        elif model == 'pca':
            data = self.data[model]['cca'][:, idx]
            low, average, high = self.statistics(data, axis=0)
            low = [low] * len(lags)
            average = [average] * len(lags)
            high = [high] * len(lags)
        else:
            raise ValueError('unknown model: ' + str(model))
        return [lags, average], [lags, low, high]
    def its(self, model, idx=0, lag=None):
        if model in ['tica', 'ae']:
            if lag not in self.data['trns_lags']:
                raise ValueError('unknown lag time: ' + str(lag))
            key = '%s_%d' % (model, lag)
        elif model in ['pca', 'ref']:
            key = model
        else:
            raise ValueError('unknown model: ' + str(model))
        lags = self.data['msm_lags']
        data = self.data[key]['its'][:, :, idx]
        low, average, high = self.statistics(data, axis=0)
        return [lags, average], [lags, low, high]

def run_sqrt_model_benchmark(
    n_runs=100,
    length=10000,
    trns_lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    msm_lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    batch_size=64,
    wrapper_args=dict(), tica_args=dict(), ae_args=dict()):
    _tica_args = dict(kinetic_map=True, symmetrize=False)
    _ae_args = dict(
        hid_size=[50],
        dropout=_nn.Dropout(p=0.5),
        activation=_nn.LeakyReLU(),
        lat_activation=None,
        batch_normalization=None,
        bias=True,
        lr=0.001,
        cuda=False,
        n_epochs=50)
    _tica_args.update(tica_args)
    _ae_args.update(ae_args)
    runner = BenchmarkRunner(
        ToyModelWrapper(sample_sqrt_model, length, discretize_1d_model),
        trns_lags, msm_lags, 1, batch_size,
        wrapper_args=wrapper_args, tica_args=_tica_args, ae_args=_ae_args)
    results = organize_results([runner() for _ in range(n_runs)])
    results.update(
        msm_lags=_np.asarray(msm_lags),
        trns_lags=_np.asarray(trns_lags))
    return results

def run_swissroll_model_benchmark(
    dim,
    n_runs=100,
    length=10000,
    trns_lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    msm_lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    batch_size=64,
    wrapper_args=dict(), tica_args=dict(), ae_args=dict()):
    if dim == 1:
        hid_size = [200, 100]
        discretizer = discretize_1d_model
    elif dim == 2:
        hid_size = [100]
        discretizer = discretize_2d_model
    else:
        raise ValueError('dim must be 1 or 2.')
    _tica_args = dict(kinetic_map=True, symmetrize=False)
    _ae_args = dict(
        hid_size=hid_size,
        dropout=_nn.Dropout(p=0.5),
        activation=_nn.LeakyReLU(),
        lat_activation=None,
        batch_normalization=None,
        bias=True,
        lr=0.001,
        cuda=False,
        n_epochs=50)
    _tica_args.update(tica_args)
    _ae_args.update(ae_args)
    runner = BenchmarkRunner(
        ToyModelWrapper(sample_swissroll_model, length, discretizer),
        trns_lags, msm_lags, dim, batch_size,
        wrapper_args=wrapper_args, tica_args=_tica_args, ae_args=_ae_args)
    results = organize_results([runner() for _ in range(n_runs)])
    results.update(
        msm_lags=_np.asarray(msm_lags),
        trns_lags=_np.asarray(trns_lags))
    return results

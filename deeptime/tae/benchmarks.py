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

'''
Automatized benchmarks.
'''

import multiprocessing as mp
import numpy as np
import torch
import tae
import os

import tae
import torch
import pyemma
from time import time

try:
    import pyemma
except ImportError:
    print('running benchmarks requires the pyemma package')

try:
    from mdshare import load as _load
except ImportError:
    print('running benchmarks requires the mdshare package')

################################################################################
#
#   BENCHMARKING THE SQRT TOY MODEL
#
################################################################################

def evaluate_sqrt_model(
    length=10000,
    trns_lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    msm_lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    use_cuda=True):
    '''A wrapper to run the sqrt model benchmarks

    Arguments:
        length (int): length of the sampled trajectory
        trns_lags (list of int): lag times for the transformers
        msm_lags (list of int): lag times for the MSM validation
        use_cuda (boolean): use a GPU to run the benchmarks
    '''
    def analyse(lat_data, ref_data, msm_lags):
        cca = tae.utils.cca(torch.from_numpy(lat_data), ref_data)[1].numpy()
        centers = np.linspace(np.min(lat_data), np.max(lat_data), 101)
        centers = 0.5 * (centers[:-1] + centers[1:]).reshape(-1, 1)
        dtraj = pyemma.coordinates.assign_to_centers(lat_data, centers)
        its = pyemma.msm.its(dtraj, lags=msm_lags, nits=1).timescales
        return cca, its
    data, dtraj = tae.toymodels.sample_sqrt_model(length)
    ref_data = tae.utils.whiten_data(
        torch.from_numpy(dtraj.reshape(-1, 1).astype(np.float32)))
    ref_its = pyemma.msm.its(dtraj, lags=msm_lags, nits=1).timescales
    lat, trn, val = tae.pca(
        data, dim=1, validation_split=0.5, batch_size=100, whiten=True)
    cca, its = analyse(lat, ref_data, msm_lags)
    result = dict(
        trns_lags=np.asarray(trns_lags),
        msm_lags=np.asarray(msm_lags),
        ref_its=np.asarray(ref_its),
        pca_its=np.asarray(its),
        pca_cca=np.asarray(cca),
        pca_trn=np.asarray(trn),
        pca_val=np.asarray(val))
    for lag in trns_lags:
        lat, trn, val = tae.tica(
            data, dim=1, lag=lag, kinetic_map=True, symmetrize=True,
            validation_split=0.5, batch_size=100, whiten=True)
        cca, its = analyse(lat, ref_data, msm_lags)
        result.update({
            'tica_%d_its' % lag: np.asarray(its),
            'tica_%d_cca' % lag: np.asarray(cca),
            'tica_%d_trn' % lag: np.asarray(trn),
            'tica_%d_val' % lag: np.asarray(val)})
        lat, trn, val = tae.ae(
            data, dim=1, lag=lag, n_epochs=200, validation_split=0.5,
            batch_size=100, whiten=True, pin_memory=use_cuda, hid_size=[200, 100],
            cuda=use_cuda, async=use_cuda)
        cca, its = analyse(lat, ref_data, msm_lags)
        result.update({
            'ae_%d_its' % lag: np.asarray(its),
            'ae_%d_cca' % lag: np.asarray(cca),
            'ae_%d_trn' % lag: np.asarray(trn),
            'ae_%d_val' % lag: np.asarray(val)})
    return result

################################################################################
#
#   BENCHMARKING THE SWISSROLL TOY MODEL
#
################################################################################

def evaluate_swissroll_model(
    dim=None,
    length=30000,
    trns_lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    msm_lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    use_cuda=True):
    '''A wrapper to run the swissroll model benchmarks

    Arguments:
        dim (int): specify the latent dimension (1 or 2)
        length (int): length of the sampled trajectory
        trns_lags (list of int): lag times for the transformers
        msm_lags (list of int): lag times for the MSM validation
        use_cuda (boolean): use a GPU to run the benchmarks
    '''
    def analyse(lat_data, ref_data, msm_lags):
        cca = tae.utils.cca(torch.from_numpy(lat_data), ref_data)[1].numpy()
        if lat_data.shape[1] == 1:
            centers = np.linspace(np.min(lat_data), np.max(lat_data), 101)
            centers = 0.5 * (centers[:-1] + centers[1:]).reshape(-1, 1)
            dtraj = pyemma.coordinates.assign_to_centers(lat_data, centers)
        else:
            dtraj = pyemma.coordinates.cluster_regspace(
                lat_data, dmin=0.2, max_centers=400).dtrajs
        its = pyemma.msm.its(dtraj, lags=msm_lags, nits=3).timescales
        return cca, its
    data, dtraj = tae.toymodels.sample_swissroll_model(length)
    ref_data = tae.utils.whiten_data(
        torch.from_numpy(dtraj.reshape(-1, 1).astype(np.float32)))
    ref_its = pyemma.msm.its(dtraj, lags=msm_lags, nits=3).timescales
    lat, trn, val = tae.pca(
        data, dim=dim, validation_split=0.5, batch_size=100, whiten=True)
    cca, its = analyse(lat, ref_data, msm_lags)
    result = dict(
        trns_lags=np.asarray(trns_lags),
        msm_lags=np.asarray(msm_lags),
        ref_its=np.asarray(ref_its),
        pca_its=np.asarray(its),
        pca_cca=np.asarray(cca),
        pca_trn=np.asarray(trn),
        pca_val=np.asarray(val))
    for lag in trns_lags:
        lat, trn, val = tae.tica(
            data, dim=dim, lag=lag, kinetic_map=True, symmetrize=True,
            validation_split=0.5, batch_size=100, whiten=True)
        cca, its = analyse(lat, ref_data, msm_lags)
        result.update({
            'tica_%d_its' % lag: np.asarray(its),
            'tica_%d_cca' % lag: np.asarray(cca),
            'tica_%d_trn' % lag: np.asarray(trn),
            'tica_%d_val' % lag: np.asarray(val)})
        lat, trn, val = tae.ae(
            data, dim=dim, lag=lag, n_epochs=200, validation_split=0.5,
            batch_size=100, whiten=True, pin_memory=use_cuda, hid_size=[200, 100],
            cuda=use_cuda, async=use_cuda)
        cca, its = analyse(lat, ref_data, msm_lags)
        result.update({
            'ae_%d_its' % lag: np.asarray(its),
            'ae_%d_cca' % lag: np.asarray(cca),
            'ae_%d_trn' % lag: np.asarray(trn),
            'ae_%d_val' % lag: np.asarray(val)})
    return result

################################################################################
#
#   BENCHMARKING THE ALANINE DIPEPTIDE MD SIMULATIONS
#
################################################################################

def evaluate_ala2_md(
    n_trajs=5,
    length=50000,
    trns_lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    msm_lags=[1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50],
    use_cuda=True):
    '''A wrapper to run the alanine dipeptide benchmarks

    Arguments:
        n_trajs (int): number of bootstrapped trajectories
        length (int): length of each bootstrapped trajectory
        trns_lags (list of int): lag times for the transformers
        msm_lags (list of int): lag times for the MSM validation
        use_cuda (boolean): use a GPU to run the benchmarks
    '''
    def analyse(lat_data, ref_data, msm_lags):
        cca = tae.utils.cca(
            torch.cat([torch.from_numpy(array) for array in lat_data]),
            ref_data)[1].numpy()
        dtrajs = pyemma.coordinates.cluster_kmeans(
            lat_data, k=300, max_iter=50, stride=10).dtrajs
        its = pyemma.msm.its(dtrajs, lags=msm_lags, nits=3).timescales
        return cca, its
    with np.load(_load('alanine-dipeptide-3x250ns-backbone-dihedrals.npz')) as fh:
        n_frames = [fh[key].shape[0] for key in sorted(fh.keys())]
        selection = []
        for i in np.random.choice(
            len(n_frames), size=n_trajs, replace=True):
            selection.append(
                [i, np.random.randint(n_frames[i] - length)])
        ref_data = [fh['arr_%d' % i][l:l+length] for i, l in selection]
    with np.load(_load('alanine-dipeptide-3x250ns-heavy-atom-positions.npz')) as fh:
        data = [fh['arr_%d' % i][l:l+length] for i, l in selection]
    dtrajs = pyemma.coordinates.cluster_kmeans(
        ref_data, k=300, max_iter=50, stride=10).dtrajs
    ref_its = pyemma.msm.its(dtrajs, lags=msm_lags, nits=3).timescales
    ref_data = tae.utils.whiten_data(
        torch.cat([torch.from_numpy(array) for array in ref_data]))
    lat, trn, val = tae.pca(
        data, dim=2, validation_split=0.5, batch_size=100, whiten=True)
    cca, its = analyse(lat, ref_data, msm_lags)
    result = dict(
        trns_lags=np.asarray(trns_lags),
        msm_lags=np.asarray(msm_lags),
        ref_its=np.asarray(ref_its),
        pca_its=np.asarray(its),
        pca_cca=np.asarray(cca),
        pca_trn=np.asarray(trn),
        pca_val=np.asarray(val))
    for lag in trns_lags:
        lat, trn, val = tae.tica(
            data, dim=2, lag=lag, kinetic_map=True, symmetrize=True,
            validation_split=0.5, batch_size=100, whiten=True)
        cca, its = analyse(lat, ref_data, msm_lags)
        result.update({
            'tica_%d_its' % lag: np.asarray(its),
            'tica_%d_cca' % lag: np.asarray(cca),
            'tica_%d_trn' % lag: np.asarray(trn),
            'tica_%d_val' % lag: np.asarray(val)})
        lat, trn, val = tae.ae(
            data, dim=2, lag=lag, n_epochs=200, validation_split=0.5,
            batch_size=100, whiten=True, pin_memory=use_cuda, hid_size=[200, 100],
            cuda=use_cuda, async=use_cuda)
        cca, its = analyse(lat, ref_data, msm_lags)
        result.update({
            'ae_%d_its' % lag: np.asarray(its),
            'ae_%d_cca' % lag: np.asarray(cca),
            'ae_%d_trn' % lag: np.asarray(trn),
            'ae_%d_val' % lag: np.asarray(val)})
    return result

################################################################################
#
#   BENCHMARKING THE VILLIN MD SIMULATIONS
#
################################################################################

def evaluate_villin_md(
    data=None,
    n_blocks=10,
    trns_lags=[10, 20, 50, 100, 200, 500],
    msm_lags=[1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200, 250, 300, 400, 500, 700, 1000],
    use_cuda=True):
    '''An inner wrapper to run the villin benchmarks for a single featurization

    Arguments:
        data (numpy.ndarray): featurized md data
        n_blocks (int): number of blocks to divide the original trajectory in
        trns_lags (list of int): lag times for the transformers
        msm_lags (list of int): lag times for the MSM validation
        use_cuda (boolean): use a GPU to run the benchmarks
    '''
    def analyse(lat_data, msm_lags):
        dtrajs = pyemma.coordinates.cluster_kmeans(
            lat_data, k=300, max_iter=50, stride=10).dtrajs
        return pyemma.msm.its(dtrajs, lags=msm_lags, nits=2).timescales
    nmax = len(data)
    length = int(np.floor(0.5 + float(nmax) / float(n_blocks)))
    active_blocks = np.random.choice(n_blocks, size=n_blocks, replace=True)
    _data = [data[n * length:min((n + 1) * length, nmax), :] for n in active_blocks]
    result = dict(
        trns_lags=np.asarray(trns_lags),
        msm_lags=np.asarray(msm_lags))
    for lag in trns_lags:
        for dim in [2, 5]:
            lat, trn, val = tae.tica(
                _data, dim=2, lag=lag, kinetic_map=True, symmetrize=True,
                validation_split=0.5, batch_size=100, whiten=True)
            result.update({
                'tica_%d_%d_its' % (lag, dim): np.asarray(analyse(lat, msm_lags)),
                'tica_%d_%d_trn' % (lag, dim): np.asarray(trn),
                'tica_%d_%d_val' % (lag, dim): np.asarray(val)})
        lat, trn, val = tae.ae(
            _data, dim=2, lag=lag, n_epochs=200, validation_split=0.5,
            batch_size=100, whiten=True, pin_memory=use_cuda, hid_size=[200, 100],
            cuda=use_cuda, async=use_cuda)
        result.update({
            'ae_%d_its' % lag: np.asarray(analyse(lat, msm_lags)),
            'ae_%d_trn' % lag: np.asarray(trn),
            'ae_%d_val' % lag: np.asarray(val)})
    return result

def evaluate_villin_md_wrapper(
    path_to_data=None,
    trns_lags=[10, 20, 50, 100, 200, 500],
    msm_lags=[1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200, 250, 300, 400, 500, 700, 1000],
    use_cuda=True):
    '''An outer wrapper to run the villin benchmarks for all featurizations

    Arguments:
        path_to_data (str): path to the villin data which we are not allowed to share
        n_blocks (int): number of blocks to divide the original trajectory in
        trns_lags (list of int): lag times for the transformers
        msm_lags (list of int): lag times for the MSM validation
        use_cuda (boolean): use a GPU to run the benchmarks
    '''
    featurisations = dict({
        'bbt': 'villin-ff-1ns-backbone-torsions.npy',
        'cap': 'villin-ff-1ns-ca-positions.npy',
        'hap': 'villin-ff-1ns-heavy-atom-positions.npy',
        'icad': 'villin-ff-1ns-inverse-ca-distances.npy'})
    result = dict()
    for model in featurisations.keys():
        data = np.load(os.path.join(path_to_data, featurisations[model]))
        model_result = evaluate_villin_md(
            data=data, trns_lags=trns_lags,
            msm_lags=msm_lags, use_cuda=use_cuda)
        for key in model_result.keys():
            if key not in ['trns_lags', 'msm_lags']:
                result.update({'%s_%s' % (model, key): model_result[key]})
    result.update(trns_lags=trns_lags, msm_lags=msm_lags)
    return result

################################################################################
#
#   MANUSCRIPT BENCHMARKS
#
################################################################################

def worker(queue, gpu, seed, evaluate_func, evaluate_kwargs):
    with torch.cuda.device(gpu):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        try:
            result = evaluate_func(**evaluate_kwargs)
        except Exception as e:
            print(e)
            result = dict()
        queue.put(result)
        queue.task_done()

def spawn(
    seed_generator, task_index, n_gpus, evaluate_func, evaluate_kwargs=dict()):
    processes = []
    queue = mp.JoinableQueue()
    for gpu in range(n_gpus):
        seed = seed_generator(task_index, gpu, n_gpus=n_gpus)
        p = mp.Process(
            target=worker,
            args=[queue, gpu, seed, evaluate_func, evaluate_kwargs])
        processes.append(p)
        print('Spawning task:%d on gpu:%d with seed:%d' % (task_index, gpu, seed))
    for p in processes:
        p.start()
    queue.join()
    out = dict()
    for _ in processes:
        result = queue.get()
        for key in result.keys():
            if key in ['trns_lags', 'msm_lags']:
                if key not in out:
                    out.update({key: result[key]})
            else:
                try:
                    out[key].append(result[key])
                except KeyError:
                    out.update({key: [result[key]]})
    return out

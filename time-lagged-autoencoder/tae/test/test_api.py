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
from torch.utils.data import DataLoader
from ..utils import create_dataset
from ..utils import whiten_data
from ..api import pca
from ..api import tica
from ..api import ae

def generate_data_2state_hmm(length=10000):
    transition_matrix = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    phi = np.random.rand() * 2.0 * np.pi
    rot = np.asarray([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)]])
    trajs, rtrajs = [], []
    for _ in range(np.random.randint(1, 5)):
        dtraj = np.zeros(shape=(length + np.random.randint(100),), dtype=np.intc)
        for i in range(1, len(dtraj)):
            dtraj[i] = np.random.choice(
                2, size=1, p=transition_matrix[dtraj[i - 1], :])
        traj = np.random.randn(len(dtraj))
        traj[np.where(dtraj == 1)[0]] += 2.0
        traj_stacked = np.vstack((traj, np.zeros(len(traj))))
        traj_rot = np.dot(rot, traj_stacked).T
        trajs.append(traj[:])
        rtrajs.append(traj_rot[:])
    if len(trajs) == 1:
        trajs = trajs[0]
        rtrajs = rtrajs[0]
    else:
        trajs = np.concatenate(trajs)
    trajs -= np.mean(trajs)
    trajs /= np.std(trajs, ddof=1)
    return trajs, rtrajs

def checkpout_output(ref, data, out):
    if isinstance(data, (list, tuple)):
        np.testing.assert_array_equal(
            [o.shape[0] for o in out],
            [d.shape[0] for d in data])
        out = np.concatenate(out)
    else:
        assert data.shape[0] == out.shape[0]
    out = out.reshape(-1)
    np.testing.assert_allclose(np.abs(np.mean(ref * out)), 1.0, atol=0.001)

################################################################################
#
#   PCA
#
################################################################################

def test_pca_2state_hmm():
    ref, data = generate_data_2state_hmm()
    out, train_loss, test_loss = pca(data, dim=1, whiten=True)
    checkpout_output(ref, data, out)

################################################################################
#
#   TICA
#
################################################################################

def test_tica_2state_hmm():
    ref, data = generate_data_2state_hmm()
    out, train_loss, test_loss = tica(data, dim=1, lag=1, whiten=True)
    checkpout_output(ref, data, out)

################################################################################
#
#   AUTOENCODER
#
################################################################################

def test_ae_2state_hmm():
    ref, data = generate_data_2state_hmm()
    out, train_loss, test_loss = ae(
        data, dim=1, lag=1, n_epochs=20, whiten=True,
        bias=False, hid_size=[], alpha=None)
    checkpout_output(ref, data, out)

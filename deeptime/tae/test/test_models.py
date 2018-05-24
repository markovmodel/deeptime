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
from ..models import PCA
from ..models import TICA
from ..models import AE

def generate_data_2state_hmm(length=10000, lag=0, batch_size=100):
    transition_matrix = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    dtraj = np.zeros(shape=(length,), dtype=np.intc)
    for i in range(1, length):
        dtraj[i] = np.random.choice(
            2, size=1, p=transition_matrix[dtraj[i - 1], :])
    traj = np.random.randn(len(dtraj))
    traj[np.where(dtraj == 1)[0]] += 2.0
    traj_stacked = np.vstack((traj, np.zeros(len(traj))))
    phi = np.random.rand() * 2.0 * np.pi
    rot = np.asarray([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)]])
    traj_rot = np.dot(rot, traj_stacked).T
    return traj, \
        DataLoader(
            create_dataset(traj_rot, lag=lag),
            batch_size=batch_size,
            shuffle=True), \
        DataLoader(
            create_dataset(traj_rot, lag=0),
            batch_size=batch_size)

################################################################################
#
#   PCA
#
################################################################################

def test_pca_2state_hmm():
    traj, train_loader, transform_loader = generate_data_2state_hmm()
    pca = PCA()
    pca.fit(train_loader, dim=1)
    out = whiten_data(pca.transform(transform_loader)).numpy().reshape((-1,))
    traj -= np.mean(traj)
    traj /= np.std(traj, ddof=1)
    np.testing.assert_allclose(np.abs(np.mean(traj * out)), 1.0, atol=0.001)

################################################################################
#
#   TICA
#
################################################################################

def test_tica_2state_hmm():
    traj, train_loader, transform_loader = generate_data_2state_hmm(lag=1)
    tica = TICA()
    tica.fit(train_loader, dim=1)
    out = whiten_data(tica.transform(transform_loader)).numpy().reshape((-1,))
    traj -= np.mean(traj)
    traj /= np.std(traj, ddof=1)
    np.testing.assert_allclose(np.abs(np.mean(traj * out)), 1.0, atol=0.001)

################################################################################
#
#   AUTOENCODER
#
################################################################################

def test_ae_2state_hmm():
    traj, train_loader, transform_loader = generate_data_2state_hmm(lag=1)
    ae = AE(2, 1, bias=False, alpha=None)
    ae.fit(train_loader, 20)
    out = whiten_data(ae.transform(transform_loader)).numpy().reshape((-1,))
    traj -= np.mean(traj)
    traj /= np.std(traj, ddof=1)
    np.testing.assert_allclose(np.abs(np.mean(traj * out)), 1.0, atol=0.001)

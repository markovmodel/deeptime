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

import numpy as np
from ..toymodels import sample_hmm
from ..toymodels import sample_sqrt_model
from ..toymodels import sample_swissroll_model

def run_sample_hmm(ndim, nstates):
    length = 10000
    states = [np.random.randn(ndim) for i in range(nstates)]
    cov = np.random.rand(ndim, ndim)
    cov = np.matmul(cov.T, cov)
    transition_matrix = np.random.rand(nstates, nstates)
    transition_matrix = transition_matrix + transition_matrix.T
    transition_matrix /= transition_matrix.sum()
    pi = transition_matrix.sum(axis=1)
    transition_matrix /= pi[:, None]
    traj, dtraj = sample_hmm(length, cov, states, transition_matrix)
    sets = [np.where(dtraj == state)[0] for state in range(nstates)]
    np.testing.assert_allclose(
        [float(len(s)) / float(length) for s in sets],
        pi, atol=0.1)
    for i, s in enumerate(sets):
        mean = np.mean(traj[s, :], axis=0)
        np.testing.assert_allclose(
            mean, states[i], atol=0.2)
        traj[s, :] -= mean
    np.testing.assert_allclose(np.cov(traj.T), cov, atol=0.2)

def test_sample_hmm_random():
    for _ in range(3):
        ndim = np.random.randint(low=2, high=5)
        nstates = np.random.randint(low=2, high=5)
        run_sample_hmm(ndim, nstates)

def test_sample_sqrt_model():
    traj, dtraj = sample_sqrt_model(20000)
    np.testing.assert_allclose(
        np.mean(traj, axis=0), [0.0, 1.9], atol=0.2)
    np.testing.assert_allclose(
        np.std(traj, axis=0, ddof=1), [5.5, 1.3], atol=0.2)

def test_sample_swissroll_model():
    traj, dtraj = sample_swissroll_model(20000)
    np.testing.assert_allclose(
        np.mean(traj, axis=0), [-3.1, 11.2, 4.9], atol=1.0)
    np.testing.assert_allclose(
        np.std(traj, axis=0, ddof=1), [7.9, 3.8, 6.7], atol=0.4)

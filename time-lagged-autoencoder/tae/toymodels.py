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
A collection of "difficult" toymodels.
'''

import numpy as _np

__all__ = ['sample_sqrt_model', 'sample_swissroll_model']

def sample_hmm(length, cov, states, transition_matrix):
    '''Sample a hidden state trajectory and n-dimensional emissions.

    We sample a hidden state trajectory using the given transition matrix. For
    each hidden state, we compute Gaussian noise around the center of the state
    using the given covariance matrix.

    Arguments:
        length (int): length of the resulting trajectories
        cov (array-like of float): covariance matrix for the noise
        states (array-like of float): centers for each state's emissions
        transition_matrix (array-like of float): a transition matrix
    '''
    cov = _np.asarray(cov, dtype=_np.float32)
    states = _np.asarray(states, dtype=_np.float32)
    transition_matrix = _np.asarray(transition_matrix, dtype=_np.float32)
    dtraj = _np.zeros(shape=(length,), dtype=_np.intc)
    dtraj[0] = _np.random.randint(low=0, high=len(states))
    for i in range(1, length):
        dtraj[i] = _np.random.choice(
            len(states), size=1, p=transition_matrix[dtraj[i - 1], :])
    traj = states[dtraj, :] + _np.random.multivariate_normal(
        _np.zeros(len(cov)), cov, size=length, check_valid='ignore')
    return traj, dtraj

def sqrt_transform(traj):
    '''Mask an emission trajectory using an sqrt transform.

    We add the square root of the first dimension (which ideally a large
    variance) to the second (which is ideally the slowest degree of freedom)
    to mask the slow process.

    Arguments:
        traj (array-like of float): a trajectory of emissions
    '''
    transformed_traj = _np.asarray(traj).copy()
    transformed_traj[:, 1] += _np.sqrt(_np.abs(traj[:, 0]))
    return transformed_traj

def sample_sqrt_model(length):
    '''Sample a hidden state and an sqrt-transformed emission trajectory.

    We sample a hidden state trajectory and sqrt-masked emissions in two
    dimensions such that the two metastable states are not linearly separable.

    Arguments:
        length (int): length of the resulting trajectories
    '''
    cov = [[30.0, 0.0], [0.0, 0.015]]
    states = [[0.0, 1.0], [0.0, -1.0]]
    transition_matrix = [[0.95, 0.05], [0.05, 0.95]]
    traj, dtraj = sample_hmm(length, cov, states, transition_matrix)
    return sqrt_transform(traj), dtraj

def swissroll_transform(traj):
    '''Mask an emission trajectory using a swissroll transform.

    We roll two dimensional emissions into a swissroll style manifold in three
    dimensions.

    Arguments:
        traj (array-like of float): a trajectory of emissions
    '''
    x = traj[:, 0]
    return _np.vstack([x * _np.cos(x), traj[:, 1], x * _np.sin(x)]).T

def sample_swissroll_model(length):
    '''Sample a hidden state and a swissroll-transformed emission trajectory.

    We sample a hidden state trajectory and swissroll-masked emissions in two
    dimensions such that the four metastable states are not linearly separable.

    Arguments:
        length (int): length of the resulting trajectories
    '''
    cov = [[1.0, 0.0], [0.0, 1.0]]
    states = [[7.5, 7.5], [7.5, 15.0], [15.0, 15.0], [15.0, 7.5]]
    transition_matrix = [
        [0.95, 0.05, 0.00, 0.00],
        [0.05, 0.90, 0.05, 0.00],
        [0.00, 0.05, 0.90, 0.05],
        [0.00, 0.00, 0.05, 0.95]]
    traj, dtraj = sample_hmm(length, cov, states, transition_matrix)
    return swissroll_transform(traj), dtraj

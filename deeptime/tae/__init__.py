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
A toolbox for dimension reduction of time series data with a
time-lagged autoencoder.
'''

__author__ = 'Christoph Wehmeyer'
__email__ = 'christoph.wehmeyer@fu-berlin.de'

try:
    import torch
except ImportError:
    from sys import exit
    print(
        'Please install pytorch>=0.4 according to the instructions on '
        'http://pytorch.org before you continue!')
    exit(1)

from .api import pca, tica, ae, vae, vampnet
from .models import PCA, TICA, AE, VAE, VAMPNet
from . import utils
from . import toymodels

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

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = 'x.y.z'
del get_distribution, DistributionNotFound

__author__ = 'Andreas Mardt, Luca Pasquali'
__email__ = 'andreas.mardt@fu-berlin.de, luca.pasquali@fu-berlin.de'

from .vampnet import VampnetTools

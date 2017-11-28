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

from setuptools import setup, find_packages

description = '''
Collection of functions to implement neural networks based
on the variational approach for Markov processes,
as described in https://arxiv.org/abs/1710.06012
'''

setup(
#    use_scm_version=dict(root='..', relative_to=__file__),
    use_scm_version=True,
    name='vampnet',
    author='Andreas Mardt, Luca Pasquali',
    author_email='andreas.mardt@fu-berlin.de, luca.pasquali@fu-berlin.de',
    url='https://github.com/markovmodel/deeptime/tree/master/vampnet',
    description=description,
    packages=find_packages(),
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'],
    zip_safe=False)

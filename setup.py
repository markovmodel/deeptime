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

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys

try:
    import torch
except ImportError:
    print(
        'Please install pytorch>=0.4 according to the instructions on '
        'http://pytorch.org before you continue!')
    sys.exit(1)

try:
    import tensorflow
except ImportError:
    print(
        'Please install tensorflow according to the instructions on '
        'https://www.tensorflow.org/install/ before you continue!')
    sys.exit(1)

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['deeptime.tae']
    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

setup(
    cmdclass={'test': PyTest},
    use_scm_version=True,
    name='deeptime',
    maintainer='Christoph Wehmeyer',
    maintainer_email='christoph.wehmeyer@fu-berlin.de',
    url='https://github.com/markovmodel/deeptime',
    description='Deep learning meets molecular dynamics.',
    packages=find_packages(),
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'],
    tests_require=['pytest'],
    zip_safe=False)

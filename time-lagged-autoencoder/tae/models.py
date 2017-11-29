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
Implementations of PCA, TICA, and AE.
'''

from torch import svd as _svd
from torch import nn as _nn
from torch import optim as _optim
from torch import diag as _diag
from torch import cat as _cat
from torch.autograd import Variable as _Variable
from .utils import get_mean as _get_mean
from .utils import get_covariance as _get_covariance
from .utils import Transform as _Transform

__all__ = ['PCA', 'TICA', 'AE']

################################################################################
#
#   PCA
#
################################################################################

class PCA(object):
    '''Perform a principal component analysis for dimensionality reduction.

    We compute the first <dim> eigenvectors of the instantaneous covariance
    matrix and use them to rotate/project the data into a lower dimensional
    subspace.
    '''
    def __init__(self):
        self.loss_function = _nn.MSELoss(size_average=False)
    def get_loss(self, loader):
        '''Train the model on the provided data loader.

        Arguments:
            loader (DataLoader): the data for loss calculation
        '''
        if loader is None:
            return None
        loss = 0.0
        for x, y in loader:
            x, y = self.transformer(x, y, variable=True)
            loss += self.loss_function(x.mm(self.score_matrix), y).data[0]
        return loss / float(len(loader.dataset))
    def fit(self, train_loader, dim=None, test_loader=None):
        '''Train the model on the provided data loader.

        Arguments:
            train_loader (DataLoader): the training data
            dim (int): the target dimensionality
            test_loader (DataLoader): the data for validation
        '''
        self.x_mean, y_mean = _get_mean(train_loader)
        self.cxx, cxy, cyy = _get_covariance(
            train_loader, self.x_mean, y_mean)
        self.transformer = _Transform(
            x_mean=self.x_mean, y_mean=self.x_mean)
        u, s, v = _svd(self.cxx)
        if dim is None:
            dim = s.size()[0]
        self.decoder_matrix = u[:, :dim]
        self.encoder_matrix = v.t()[:dim, :]
        self.score_matrix = _Variable(
            self.decoder_matrix.mm(self.encoder_matrix))
        return self.get_loss(train_loader), self.get_loss(test_loader)
    def transform(self, loader):
        '''Apply the model on the provided data loader.

        Arguments:
            loader (DataLoader): the data you wish to transform
        '''
        latent = []
        for x, _ in loader:
            x = self.transformer.x(x)
            latent.append(x.mm(self.encoder_matrix.t()))
        return _cat(latent)

################################################################################
#
#   TICA
#
################################################################################

class TICA(object):
    '''Perform a time-lagged independent component analysis for
    dimensionality reduction.

    We compute a rank-d approximation to the Koopman operator and use it to
    rotate/project the data into a lower dimensional subspace.

    Arguments:
        kinetic_map (boolean): use the kinetic map variant of TICA
        symmetrize (boolean): enforce symmetry and reversibility
    '''
    def __init__(self, kinetic_map=True, symmetrize=False):
        self.loss_function = _nn.MSELoss(size_average=False)
        self.kinetic_map = kinetic_map
        self.symmetrize = symmetrize
    def get_loss(self, loader):
        '''Train the model on the provided data loader.

        Arguments:
            loader (DataLoader): the data for loss calculation
        '''
        if loader is None:
            return None
        loss = 0.0
        for x, y in loader:
            x, y = self.transformer(x, y, variable=True)
            loss += self.loss_function(x.mm(self.koopman_matrix), y).data[0]
        return loss / float(len(loader.dataset))
    def fit(self, train_loader, dim=None, test_loader=None):
        '''Train the model on the provided data loader.

        Arguments:
            train_loader (DataLoader): the training data
            dim (int): the target dimensionality
            test_loader (DataLoader): the data for validation
        '''
        self.x_mean, self.y_mean = _get_mean(train_loader)
        self.cxx, self.cxy, self.cyy = _get_covariance(
            train_loader, self.x_mean, self.y_mean)
        if self.symmetrize:
            self.cxx = 0.5 * (self.cxx + self.cyy)
            self.cyy.copy_(self.cxx)
            self.cxy = 0.5 * (self.cxy + self.cxy.t())
        self.transformer = _Transform(
            x_mean=self.x_mean, x_covariance=self.cxx,
            y_mean=self.y_mean, y_covariance=self.cyy)
        self.ixx = self.transformer.x.mul
        self.iyy = self.transformer.y.mul
        u, s, v = _svd(self.ixx.mm(self.cxy.mm(self.iyy)))
        if dim is None:
            dim = s.size()[0]
        self.decoder_matrix = u[:, :dim]
        self.encoder_matrix = v.t()[:dim, :]
        if self.kinetic_map:
            self.encoder_matrix = _diag(s[:dim]).mm(self.encoder_matrix)
        else:
            self.decoder_matrix = self.decoder_matrix.mm(_diag(s[:dim]))
        self.koopman_matrix = _Variable(
            self.decoder_matrix.mm(self.encoder_matrix))
        return self.get_loss(train_loader), self.get_loss(test_loader)
    def transform(self, loader):
        '''Apply the model on the provided data loader.

        Arguments:
            loader (DataLoader): the data you wish to transform
        '''
        latent = []
        for x, _ in loader:
            x = self.transformer.x(x)
            latent.append(x.mm(self.encoder_matrix.t()))
        return _cat(latent)

################################################################################
#
#   AUTOENCODER
#
################################################################################

class _BaseAE(_nn.Module):
    def __init__(
        self, dropout, activation, lat_activation, batch_normalization):
        super(_BaseAE, self).__init__()
        self.dropout = dropout
        self.activation = activation
        if lat_activation is not None:
            self.lat_activation = lat_activation
        if batch_normalization is not None:
            self.batch_normalization = batch_normalization
        self.use_cuda = False
    def encode(self, x):
        raise NotImplementedError('overwrite in subclass')
    def decode(self, z):
        raise NotImplementedError('overwrite in subclass')
    def forward(self, x):
        raise NotImplementedError('overwrite in subclass')
    def train_step(
        self, loader):
        self.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = self.transformer(x, y, variable=True, train=True)
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()
            self.optimizer.zero_grad()
            try:
                x = self.batch_normalization(x)
            except AttributeError:
                pass
            y_recon = self(x)
            loss = self.loss_function(y_recon, y)
            loss.backward()
            train_loss += loss.data[0]
            self.optimizer.step()
        return train_loss / float(len(loader.dataset))
    def test_step(self, loader):
        self.eval()
        test_loss = 0
        if loader is None:
            return None
        for i, (x, y) in enumerate(loader):
            x, y = self.transformer(x, y, variable=True)
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()
            try:
                x = self.batch_normalization(x)
            except AttributeError:
                pass
            y_recon = self(x)
            test_loss += self.loss_function(y_recon, y).data[0]
        return test_loss / float(len(loader.dataset))
    def fit(self, train_loader, n_epochs, test_loader=None):
        '''Train the model on the provided data loader.

        Arguments:
            train_loader (DataLoader): the training data
            n_epochs (int): number of training epochs
            test_loader (DataLoader): the data for validation
        '''
        x_mean, y_mean = _get_mean(train_loader)
        cxx, cxy, cyy = _get_covariance(train_loader, x_mean, y_mean)
        self.transformer = _Transform(
            x_mean=x_mean, x_covariance=cxx, y_mean=y_mean, y_covariance=cyy)
        train_loss, test_loss = [], []
        for epoch in range(n_epochs):
            train_loss.append(
                self.train_step(
                    train_loader))
            test_loss.append(
                self.test_step(test_loader))
        return train_loss, test_loss
    def transform(self, loader):
        '''Apply the model on the provided data loader.

        Arguments:
            loader (DataLoader): the data you wish to transform
        '''
        self.eval()
        latent = []
        for i, (x, _) in enumerate(loader):
            x = self.transformer.x(
                x, variable=True, volatile=True, requires_grad=False)
            if self.use_cuda:
                x = x.cuda()
            try:
                x = self.batch_normalization(x)
            except AttributeError:
                pass
            y = self.encode(x)
            if self.cuda:
                y = y.cpu()
            latent.append(y)
        return _cat(latent).data

class AE(_BaseAE):
    '''Use a time-lagged autoencoder model for dimensionality reduction.

    We train a time-lagged autoencoder type neural network.

    Arguments:
        inp_size (int): dimensionality of the full space
        lat_size (int): dimensionality of the desired latent space
        hid_size (sequence of int): sizes of the hidden layers
        dropout (Dropout): dropout layer for each hidden layer
        activation (Activation) activation layer for each hidden layer
        lat_activation (Activation) activation layer for the latent layer
        batch_normalization (BatchNormalization) use batch normalization
        bias (boolean): specify usage of bias neurons
        lr (float): learning rate parameter for Adam
        cuda (boolean): use the GPU
    '''
    def __init__(
        self, inp_size, lat_size, hid_size=[],
        dropout=_nn.Dropout(p=0.5), activation=_nn.LeakyReLU(),
        lat_activation=None, batch_normalization=None,
        bias=True, lr=0.001, cuda=False):
        super(AE, self).__init__(
            dropout, activation, lat_activation, batch_normalization)
        sizes = [inp_size] + list(hid_size) + [lat_size]
        self.enc_layers = [
            _nn.Linear(
                sizes[i-1], sizes[i], bias=bias) for i in range(1, len(sizes))]
        self.dec_layers = [
            _nn.Linear(
                sizes[i], sizes[i-1], bias=bias) for i in reversed(
                    range(1, len(sizes)))]
        for i, layer in enumerate(self.enc_layers):
            self.add_module('enc_%d' % i, layer)
        for i, layer in enumerate(self.dec_layers):
            self.add_module('dec_%d' % i, layer)
        self.loss_function = _nn.MSELoss(size_average=False)
        self.optimizer = _optim.Adam(self.parameters(), lr=lr)
        if cuda:
            self.use_cuda = True
            self.cuda()
    def encode(self, x):
        y = x
        for layer in self.enc_layers[:-1]:
            y = self.dropout(self.activation(layer(y)))
        return self.enc_layers[-1](y)
    def decode(self, z):
        try:
            y = self.lat_activation(z)
        except AttributeError:
            y = z
        for layer in self.dec_layers[:-1]:
            y = self.dropout(self.activation(layer(y)))
        return self.dec_layers[-1](y)
    def forward(self, x):
        return self.decode(self.encode(x))

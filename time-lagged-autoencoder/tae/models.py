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
Implementations of PCA, TICA, and AE.
'''

from torch import svd as _svd
from torch import nn as _nn
from torch import optim as _optim
from torch import diag as _diag
from torch import cat as _cat
from torch import randn as _randn
from torch import sum as _sum
from torch.autograd import Variable as _Variable
from .utils import get_mean as _get_mean
from .utils import get_covariance as _get_covariance
from .utils import Transform as _Transform

__all__ = ['PCA', 'TICA', 'AE', 'VAE']

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
        self.decoder_matrix = v[:, :dim]
        self.encoder_matrix = u.t()[:dim, :]
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
#   AUTOENCODER BASE CLASS
#
################################################################################

class BaseAE(_nn.Module):
    '''Basic shape of a time-lagged autoencoder family model for
    dimensionality reduction.

    We train a time-lagged variational autoencoder type neural network.

    Arguments:
        inp_size (int): dimensionality of the full space
        lat_size (int): dimensionality of the desired latent space
        hid_size (sequence of int): sizes of the hidden layers
        dropout (Dropout): dropout layer for each hidden layer
        alpha (float) activation parameter for the rectified linear units
        prelu (bool) use a learnable ReLU
        bias (boolean): specify usage of bias neurons
        lr (float): learning rate parameter for Adam
        cuda (boolean): use the GPU
    '''
    def __init__(
        self, inp_size, lat_size, hid_size,
        dropout, alpha, prelu, bias, lr, cuda, async):
        super(BaseAE, self).__init__()
        sizes = [inp_size] + list(hid_size) + [lat_size]
        self._last = len(sizes) - 2
        if isinstance(dropout, float):
            dropout = _nn.Dropout(p=dropout)
        self._setup(sizes, bias, alpha, prelu, dropout)
        self._mse_loss_function = _nn.MSELoss(size_average=False)
        self.optimizer = _optim.Adam(self.parameters(), lr=lr)
        self.async = async
        if cuda:
            self.use_cuda = True
            self.cuda() # the async=... parameter is not accepted, here
        else:
            self.use_cuda = False
    def _create_activation(self, key, idx, alpha, prelu, suffix=''):
        if alpha is None:
            activation = None
        elif alpha < 0.0:
            raise ValueError('alpha must be a non-negative number')
        elif alpha == 0.0:
            activation = _nn.ReLU()
        elif prelu:
            activation = _nn.PReLU(num_parameters=1, init=alpha)
        else:
            activation = _nn.LeakyReLU(negative_slope=alpha)
        if activation is not None:
            setattr(self, key + '_act_%d%s' % (idx, suffix), activation)
        layer = getattr(self, key + '_prm_%d%s' % (idx, suffix))
        _nn.init.kaiming_normal(layer.weight.data, a=alpha, mode='fan_in')
        try:
            layer.bias.data.uniform_(0.0, 0.1)
        except AttributeError:
            pass
    def _try_to_apply_module(self, key, value):
        try:
            return getattr(self, key)(value)
        except AttributeError:
            return value
    def _apply_layer(self, key, idx, value):
        return self._try_to_apply_module(
            key + '_drp_%d' % idx, self._try_to_apply_module(
                key + '_act_%d' % idx, self._try_to_apply_module(
                    key + '_prm_%d' % idx, value)))
    def loss_function(self, y, model_output):
        raise NotImplementedError('Implement in child class')
    def train_step(self, loader):
        self.train()
        train_loss = 0
        for x, y in loader:
            x, y = self.transformer(x, y, variable=True, train=True)
            if self.use_cuda:
                x = x.cuda(async=self.async)
                y = y.cuda(async=self.async)
            self.optimizer.zero_grad()
            loss = self.loss_function(y, self(x))
            loss.backward()
            train_loss += loss.data[0]
            self.optimizer.step()
        return train_loss / float(len(loader.dataset))
    def test_step(self, loader):
        self.eval()
        test_loss = 0
        if loader is None:
            return None
        for x, y in loader:
            x, y = self.transformer(x, y, variable=True)
            if self.use_cuda:
                x = x.cuda(async=self.async)
                y = y.cuda(async=self.async)
            test_loss += self.loss_function(y, self(x)).data[0]
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
        for x, _ in loader:
            x = self.transformer.x(
                x, variable=True, volatile=True, requires_grad=False)
            if self.use_cuda:
                x = x.cuda(async=self.async)
            y = self.encode(x)
            if self.cuda:
                y = y.cpu()
            latent.append(y)
        return _cat(latent).data

################################################################################
#
#   AUTOENCODER
#
################################################################################

class AE(BaseAE):
    '''Use a time-lagged autoencoder model for dimensionality reduction.

    We train a time-lagged autoencoder type neural network.

    Arguments:
        inp_size (int): dimensionality of the full space
        lat_size (int): dimensionality of the desired latent space
        hid_size (sequence of int): sizes of the hidden layers
        dropout (Dropout): dropout layer for each hidden layer
        alpha (float) activation parameter for the rectified linear units
        prelu (bool) use a learnable ReLU
        bias (boolean): specify usage of bias neurons
        lr (float): learning rate parameter for Adam
        cuda (boolean): use the GPU
    '''
    def __init__(
        self, inp_size, lat_size, hid_size=[],
        dropout=0.5, alpha=0.01, prelu=False,
        bias=True, lr=0.001, cuda=False, async=False):
        super(AE, self).__init__(
            inp_size, lat_size, hid_size,
            dropout, alpha, prelu, bias, lr, cuda, async)
    def _setup(self, sizes, bias, alpha, prelu, dropout):
        for c, idx in enumerate(range(1, len(sizes))):
            setattr(
                self,
                'enc_prm_%d' % c,
                _nn.Linear(sizes[idx - 1], sizes[idx], bias=bias))
            self._create_activation('enc', c, alpha, prelu)
            if c < self._last:
                if dropout is not None:
                    setattr(self, 'enc_drp_%d' % c, dropout)
        for c, idx in enumerate(reversed(range(1, len(sizes)))):
            setattr(
                self,
                'dec_prm_%d' % c,
                _nn.Linear(sizes[idx], sizes[idx - 1], bias=bias))
            if c < self._last:
                self._create_activation('dec', c, alpha, prelu)
                if dropout is not None:
                    setattr(self, 'dec_drp_%d' % c, dropout)
            else:
                self._create_activation('dec', c, None, None)
    def loss_function(self, y, model_output):
        return self._mse_loss_function(model_output, y)
    def encode(self, x):
        y = x
        for idx in range(self._last):
            y = self._apply_layer('enc', idx, y)
        return getattr(self, 'enc_prm_%d' % self._last)(y)
    def decode(self, z):
        y = self._try_to_apply_module('enc_act_%d' % self._last, z)
        for idx in range(self._last):
            y = self._apply_layer('dec', idx, y)
        return getattr(self, 'dec_prm_%d' % self._last)(y)
    def forward(self, x):
        return self.decode(self.encode(x))

################################################################################
#
#   VARIATIONAL AUTOENCODER
#
################################################################################

class VAE(BaseAE):
    '''Use a time-lagged variational autoencoder model for dimensionality
    reduction.

    We train a time-lagged variational autoencoder type neural network.

    Arguments:
        inp_size (int): dimensionality of the full space
        lat_size (int): dimensionality of the desired latent space
        hid_size (sequence of int): sizes of the hidden layers
        beta (float) : KLD weight for optimization
        dropout (Dropout): dropout layer for each hidden layer
        alpha (float) activation parameter for the rectified linear units
        prelu (bool) use a learnable ReLU
        bias (boolean): specify usage of bias neurons
        lr (float): learning rate parameter for Adam
        cuda (boolean): use the GPU
    '''
    def __init__(
        self, inp_size, lat_size, hid_size=[], beta=1.0,
        dropout=0.5, alpha=0.01, prelu=False,
        bias=True, lr=0.001, cuda=False, async=False):
        super(VAE, self).__init__(
            inp_size, lat_size, hid_size,
            dropout, alpha, prelu, bias, lr, cuda, async)
        self.beta = beta
    def _setup(self, sizes, bias, alpha, prelu, dropout):
        for c, idx in enumerate(range(1, len(sizes) - 1)):
            setattr(
                self,
                'enc_prm_%d' % c,
                _nn.Linear(sizes[idx - 1], sizes[idx], bias=bias))
            self._create_activation('enc', c, alpha, prelu)
            if dropout is not None:
                setattr(self, 'enc_drp_%d' % c, dropout)
        setattr(
            self,
            'enc_prm_%d_mu' % self._last,
            _nn.Linear(sizes[-2], sizes[-1], bias=bias))
        self._create_activation('enc', self._last, None, None, suffix='_mu')
        setattr(
            self,
            'enc_prm_%d_lv' % self._last,
            _nn.Linear(sizes[-2], sizes[-1], bias=bias))
        self._create_activation('enc', self._last, None, None, suffix='_lv')
        for c, idx in enumerate(reversed(range(1, len(sizes)))):
            setattr(
                self,
                'dec_prm_%d' % c,
                _nn.Linear(sizes[idx], sizes[idx - 1], bias=bias))
            if c < self._last:
                self._create_activation('dec', c, alpha, prelu)
                if dropout is not None:
                    setattr(self, 'dec_drp_%d' % c, dropout)
            else:
                self._create_activation('dec', c, None, None)
    def loss_function(self, y, model_output):
        y_recon, mu, lv = model_output
        mse = self._mse_loss_function(y_recon, y)
        kld = -0.5 * _sum(1.0 + lv - mu.pow(2) - lv.exp())
        return mse + self.beta * kld / float(y.size(1))
    def _encode(self, x):
        y = x
        for idx in range(self._last):
            y = self._apply_layer('enc', idx, y)
        mu = getattr(self, 'enc_prm_%d_mu' % self._last)(y)
        lv = getattr(self, 'enc_prm_%d_lv' % self._last)(y)
        return mu, lv
    def _reparameterize(self, mu, lv):
        if self.training:
            std = lv.mul(0.5).exp_()
            eps = _Variable(_randn(*std.size()))
            if self.use_cuda:
                eps = eps.cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu
    def encode(self, x):
        return self._reparameterize(*self._encode(x))
    def decode(self, z):
        y = z
        for idx in range(self._last):
            y = self._apply_layer('dec', idx, y)
        return getattr(self, 'dec_prm_%d' % self._last)(y)
    def forward(self, x):
        mu, lv = self._encode(x)
        return self.decode(self._reparameterize(mu, lv)), mu, lv

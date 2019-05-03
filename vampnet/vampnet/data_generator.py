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

"""sample generator for the MCMM project's clustering stage"""

import numpy as np

################################################################################
#                                                                              #
#   defining test potentials                                                   #
#                                                                              #
################################################################################

class BrownianDynamics(object):
    r"""base class for Brownian dynamics integration"""
    def __init__(self, dim, dt, kT, mass, damping):
        self.dim = dim
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self.daming = damping
        self.coeff_A = dt / (mass * damping)
        self.coeff_B = np.sqrt(2.0 * dt * kT / (mass * damping))
    def gradient(self, x):
        r"""gradient of the yet unkown potential"""
        raise NotImplementedError("implement in child class")
    def step(self, x):
        r"""perform a single Brownian dynamics step"""
        return x - self.coeff_A * self.gradient(x) \
            + self.coeff_B * np.random.normal(size=self.dim)


################################################################################
#                                                                              #
#   defining test potentials                                                   #
#                                                                              #
################################################################################

def asymmetric_double_well_energy(x):
    r"""computes the potential energy at point x"""
    _x = x - 2.0
    return 2.0 * _x - 6.0 * _x**2 + _x**4

def asymmetric_double_well_gradient(x):
    r"""computes the potential's gradient at point x"""
    return 4.0 * x**3 - 24.0 * x**2 + 36.0 * x - 6.0

def prinz_energy(x):
    return 4*(x**8 + 0.8 * np.exp(-80*x**2) + 0.2*np.exp(-80*(x-0.5)**2) + 0.5*np.exp(-40.*(x+0.5)**2))

def prinz_gradient(x):
    return 4*(8*x**7 - 128. * np.exp(-80*x**2)*x - 32.*np.exp(-80*(x-0.5)**2) *(x-0.5) - 40*np.exp(-40.*(x+0.5)**2) *(x+0.5))

def folding_model_energy(rvec, rcut):
    r"""computes the potential energy at point rvec"""
    r = np.linalg.norm(rvec) - rcut
    rr = r**2
    if r < 0.0:
        return -2.5 * rr
    return 0.5 * (r - 2.0) * rr

def folding_model_gradient(rvec, rcut):
    r"""computes the potential's gradient at point rvec"""
    rnorm = np.linalg.norm(rvec)
    if rnorm == 0.0:
        return np.zeros(rvec.shape)
    r = rnorm - rcut
    if r < 0.0:
        return -5.0 * r * rvec / rnorm
    return (1.5 * r - 2.0) * rvec / rnorm


################################################################################
#                                                                              #
#   defining wrapper classes                                                   #
#                                                                              #
################################################################################

class AsymmetricDoubleWell(BrownianDynamics):
    r"""encapsulates the asymmetric double well potential"""
    def __init__(self, dt, kT, mass=1.0, damping=1.0):
        super(AsymmetricDoubleWell, self).__init__(1, dt, kT, mass, damping)
    def gradient(self, x):
        return asymmetric_double_well_gradient(x)
    def sample(self, x0, nsteps, nskip=1):
        r"""generate nsteps sample points"""
        x = np.zeros(shape=(nsteps+1,))
        x[0] = x0
        for t in range(nsteps):
            q = x[t]
            for s in range(nskip):
                q = self.step(q)
            x[t+1] = q
        return x

class FoldingModel(BrownianDynamics):
    r"""encapsulates the folding model potential"""
    def __init__(self, dt, kT, mass=1.0, damping=1.0, rcut=3.0):
        super(FoldingModel, self).__init__(5, dt, kT, mass, damping)
        self.rcut = rcut
    def gradient(self, x):
        return folding_model_gradient(x, self.rcut)
    def sample(self, rvec0, nsteps, nskip=1):
        r"""generate nsteps sample points"""
        rvec = np.zeros(shape=(nsteps+1, self.dim))
        rvec[0, :] = rvec0[:]
        for t in range(nsteps):
            q = rvec[t, :]
            for s in range(nskip):
                q = self.step(q)
            rvec[t+1, :] = q[:]
        return rvec

class PrinzModel(BrownianDynamics):
    r"""encapsulates the Prinz potential"""
    def __init__(self, dt, kT, mass=1.0, damping=1.0):
        super(PrinzModel, self).__init__(1, dt, kT, mass, damping)
    def gradient(self, x):
        return prinz_gradient(x)
    def sample(self, x0, nsteps, nskip=1):
        r"""generate nsteps sample points"""
        x = np.zeros(shape=(nsteps+1,))
        x[0] = x0
        for t in range(nsteps):
            q = x[t]
            for s in range(nskip):
                q = self.step(q)
            x[t+1] = q
        return x


################################################################################
#                                                                              #
#   main area                                                                  #
#                                                                              #
################################################################################

def get_asymmetric_double_well_data(nstep, x0 = 0., nskip=1, dt=0.01, kT=10.0, mass=1.0, damping=1.0):
    r"""wrapper for the asymmetric double well generator"""
    adw = AsymmetricDoubleWell(dt, kT, mass=mass, damping=damping)
    return adw.sample(x0, nstep, nskip=nskip)

def get_folding_model_data(
        nstep, rvec0 = np.zeros((5)), nskip=1, dt=0.01, kT=10.0, mass=1.0, damping=1.0, rcut=3.0):
    r"""wrapper for the folding model generator"""
    fm = FoldingModel(dt, kT, mass=mass, damping=damping, rcut=rcut)
    return fm.sample(rvec0, nstep, nskip=nskip)

def get_prinz_pot(nstep, x0 = 0., nskip=1, dt=0.01, kT=10.0, mass=1.0, damping=1.0):
    r"""wrapper for the Prinz model generator"""
    pw = PrinzModel(dt, kT, mass=mass, damping=damping)
    return pw.sample(x0, nstep, nskip=nskip)

def get_alanine_data(input_type = 'coordinates', return_dihedrals = True):
    
    import mdshare
    
    retval = []

    if input_type == 'distances':

        local_filename = mdshare.fetch('alanine-dipeptide-3x250ns-heavy-atom-distances.npz')

        traj_whole = np.load(local_filename)['arr_0']

    elif input_type == 'coordinates':

        local_filename = mdshare.fetch('alanine-dipeptide-3x250ns-heavy-atom-positions.npz')

        traj_whole = np.load(local_filename)['arr_0']
    
    retval.append(traj_whole)

    if return_dihedrals:
        dihedral = np.load(mdshare.fetch('alanine-dipeptide-3x250ns-backbone-dihedrals.npz'))['arr_0']
        retval.append(dihedral)
    
    
    return retval


def build_generator_on_source(data_source, batch_size, lag, output_size):
    '''Function used to create a generator that will fetch data from a data source through an iterator.
    This can be passed as parameter to a keras fit_generator method.

    Parameters
    ----------
    data_source: pyemma source object.
        Data files source. This has to be initialized with chunksize = batch_size

    batch_size: int
        Batch size to be used for the training

    lag: int
        time frames lag to be used in the training of the VAMPnets

    output_size: int
        How many output nodes the network has
    '''

    counter_batches = 0


    # How many batches before the iterator has to be reinitialized
    steps_epoch = np.sum(np.ceil((data_source.trajectory_lengths()-lag)/batch_size))

    data_iterator = data_source.iterator(chunk = batch_size,
                                         lag = lag,
                                         return_trajindex=False)

    while True:       

        input_data = list(data_iterator.next())        

        # Create empty labels to accomodate keras' interface requirements
        labels = np.empty((input_data[0].shape[0],2*output_size)).astype('float32')
        data = input_data, labels
        counter_batches += 1

        if counter_batches == steps_epoch:
            data_iterator = data_source.iterator(chunk = batch_size,
                                                 lag = lag,
                                                 return_trajindex=False)
            counter_batches = 0

        yield data

        
    
def build_generator_on_source_shuffle(data_source, batch_size, lag, output_size, preloaded_batches = 1):
    '''Function used to create a generator that will randomly access data and fetch them from a data
    source through an iterator. This can be passed as parameter to a keras fit_generator method.

    Parameters
    ----------
    data_source: pyemma source object.
        Data files source. This has to be initialized with chunksize = batch_size

    batch_size: int
        Batch size to be used for the training

    lag: int
        time frames lag to be used in the training of the VAMPnets

    output_size: int
        How many output nodes the network has
        
    preloaded_batches: int
        How many batches of data should be loaded at once; higher values will improve
        execution speed but also memory consumption
    '''

    counter_batches = 0


    # How many batches before the iterator has to be reinitialized
    steps_epoch = np.ceil(np.sum((data_source.trajectory_lengths()-lag)/ (batch_size* preloaded_batches)))
    input_size = data_source.dimension()
    
    
    traj_lengths = data_source.trajectory_lengths()
    remaining_frames = np.concatenate([[index_traj*np.ones((traj_len - lag)), np.arange(traj_len - lag)] for index_traj, traj_len in enumerate(traj_lengths)], axis = 1).T.astype('int')
    indexes = np.arange(remaining_frames.shape[0])
    np.random.shuffle(indexes)
    
    while True:       
        
        start = counter_batches * batch_size * preloaded_batches
        end = min(start + batch_size * preloaded_batches, remaining_frames.shape[0])
        
        frames = remaining_frames[indexes[start:end]]
        
        fake_ind = frames[:,0]*(traj_lengths.sum()) + frames[:,1]
        arg_sort = np.argsort(fake_ind)
        sort_arg_sort = np.argsort(arg_sort)
        
        frames_tau = frames + np.array([np.zeros((frames.shape[0])), np.ones((frames.shape[0]))*lag], dtype = 'int').T
        
        
        data_iterator_t = data_source.iterator(stride=frames[arg_sort],
                                               return_trajindex=False)
        data_iterator_tau = data_source.iterator(stride=frames_tau[arg_sort],
                                               return_trajindex=False)
        
        data = np.empty((2, batch_size * preloaded_batches, input_size))
        start_iter = 0
        for iter_data, iter_data_tau in zip(data_iterator_t, data_iterator_tau):
            temp_frames = iter_data.shape[0]
            end_iter = start_iter + temp_frames
            data[0, start_iter:end_iter] = iter_data
            data[1, start_iter:end_iter] = iter_data_tau
            start_iter = end_iter

            
        data = data[:, sort_arg_sort]
        
        index_preloaded = 0
        labels = np.empty((batch_size,2*output_size)).astype('float32')
        
        while index_preloaded < preloaded_batches:
            
            start_batch = index_preloaded * batch_size
            end_batch = start_batch + batch_size
            index_preloaded += 1
            
            if end_batch > data.shape[1]:
                end_batch = data.shape[1]
                index_preloaded = preloaded_batches
                labels = np.empty((end_batch - start_batch,2*output_size)).astype('float32')
                
            output_data = [data[0, start_batch:end_batch], data[1, start_batch:end_batch]], labels
            
            yield output_data

        
        counter_batches += 1

        if counter_batches == steps_epoch:
            
            counter_batches = 0
            indexes = np.arange(remaining_frames.shape[0])
            np.random.shuffle(indexes)
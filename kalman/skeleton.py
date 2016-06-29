#!/usr/bin/python2
# coding=utf-8
"""
    This is a skeleton for test case
"""
from __future__ import print_function, absolute_import
import random
import numpy as np
import sys
try:
    from kalman.kalman import KalmanFilter
    from kalman.tools import gc_clean
    if sys.version_info < (3,):
        from kalman.libs.fortran_libs_py2 import gauss_f
    else:
        from kalman.libs.fortran_libs_py3 import gauss_f
except:
    from kalman import KalmanFilter
    from tools import gc_clean
    if sys.version_info < (3,):
        from libs.fortran_libs_py2 import gauss_f
    else:
        from libs.fortran_libs_py3 import gauss_f

use_fortran = True

if sys.version_info < (3,):
    range = xrange


class SkelReality(object):
    """
        This class contains the analytical solution of our problem
        It also provide way to get a noisy field around this solution
    """
    It = - 1
    nIt = 0
    err = 999.
    dt = 0.
    noiselevel = 0.
    size = 0
    field = np.zeros([size])

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _noiselevel, _dt, _shape):
        self.noiselevel = _noiselevel
        self.dt = _dt
        self.addnoisev = np.vectorize(self.addnoise)
        self.size = np.prod(_shape)
        self.field = np.zeros(_shape)
        gc_clean()

    def getsol(self):
        """
            Extract the solution, here it's trivial
        :return: the analytical solution
        """
        return self.field

    def addnoise(self, value):
        """
            Add noise around a value
        :param value:
        :return:
        """
        return random.gauss(value, self.noiselevel)

    def getsolwithnoise(self):
        """
            Get a noisy field around the analytical solution
        :return: a noisy field
        """
        # field_with_noise = np.zeros(self.field.shape)
        # for i in range(self.field.shape):
        #     field_with_noise[i] = random.gauss(self.field[i], self.noiselevel)
        # field_with_noise = np.array([random.gauss(d, self.noiselevel)
        #                              for d in self.field.flat]).reshape(self.field.shape)
        # return self.addnoisev(self.field)
        if use_fortran:
            return gauss_f(self.field.flat, self.noiselevel).reshape(self.field.shape)
        else:
            return self.addnoisev(self.field)

    def compute(self):
        """
            Compute the analytical solution
        """
        pass

    def step(self):
        """
            Compute the analytical solution
        """
        pass

    def reinit(self):
        """
            reinitialize the iteration number
        """
        self.It = - 1
        self.step()


class SkelSimulation(object):
    """
    This class contains everything for the simulation
    """
    It = - 1
    nIt = 0
    err = 999.
    dt = 0.
    noiselevel = 0.
    size = 0
    field = np.zeros([0])
    Mat = np.array([[]])
    rhs = np.array([])

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _noiselevel, _dt, _shape):
        self.noiselevel = _noiselevel
        self.dt = _dt
        self.size = np.product(_shape)
        self.field = np.zeros(_shape)
        self.Mat = np.zeros(self.size)
        self.rhs = np.zeros(self.size)
        gc_clean()

    def getsol(self):
        """
            Get current solution
        :return: current solution
        """
        return self.field

    def setsol(self, field):
        """
            Set current solution
        :param field: field
        """
        self.field = field

    def step(self):
        """
            Increment through the next time step of the simulation.
        """
        self.field = self.Mat.dot(self.field) + self.rhs
        self.It += 1


class SkelKalmanWrapper(object):
    """
        This class is use around the simulation to apply the kalman filter
    """
    It = - 1
    nIt = 0
    err = 999.
    dt = 0.
    size = 0

    def __init__(self, _reality, _sim):
        self.reality = _reality
        self.kalsim = _sim

        self.It = self.kalsim.It
        self.nIt = self.kalsim.nIt
        self.err = self.kalsim.err
        self.dt = self.kalsim.dt
        self.size = self.kalsim.size

        _M = self.getwindow()  # Observation matrix.
        self.kalman = KalmanFilter(self.kalsim, _M)
        self.kalman.S = np.eye(self.kalman.size_s) * self.reality.noiselevel ** 2   # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * self.reality.noiselevel ** 2   # Estimated error in measurements.
        self.kalman.Q = np.eye(self.kalman.size_s) * self.kalsim.noiselevel ** 2    # Estimated error in process.
        gc_clean()

    def getwindow(self):
        """
            Produce the observation matrix : designate what we conserve of the noisy field
        :return: observation matrix
        """
        M = np.eye(self.kalsim.size)
        return M

    def setmes(self, field):
        """
            Reshape noisy field and gives it to the kalman filter
        :param field:
        """
        self.kalman.Y = self.kalman.M.dot(np.reshape(field, self.kalman.size_s))

    def getsol(self):
        """
            Extract the solution from kalman filter
        :return: current solution
        """
        return self.kalman.X

    def setsol(self, field):
        """
            Set the current solution, for both the simulation and the kalman filter
        :param field: field
        """
        self.kalman.X = field
        self.kalsim.setsol(field)

    def step(self):
        """
            Compute the next step of the simulation and apply kalman filter to the result
        """
        self.kalsim.step()
        self.setsol(self.kalsim.getsol())
        self.setmes(self.reality.getsolwithnoise())
        self.kalman.apply()
        self.kalsim.setsol(self.getsol())
        self.It = self.kalsim.It


class EDP(object):
    """
        This class contains everything for this test case :
        * The analytical solution
        * A simulation
        * A filtered simulation
        * how to plot the results
    """
    noise_real = 20.
    noise_sim = 20.
    dt = 0.
    nIt = 0
    name = "Skeleton"

    def __init__(self):
        self.simulation = SkelSimulation(self.noise_sim, self.dt, [0])
        self.kalsim = SkelSimulation(self.noise_sim, self.dt, [0])
        self.dt = self.simulation.dt
        self.reality = SkelReality(self.noise_real, self.dt, [0])
        self.kalman = SkelKalmanWrapper(self.reality, self.kalsim)
        gc_clean()

    def compute(self, simu):
        """
            Generator : each call produce a time step the a simulation
        :param simu: the simulation to perform (filtered or not)
        :return: iteration number
        """
        self.reality.reinit()
        for i in range(simu.nIt):
            if i > 0:
                if not self.reality == simu:
                    self.reality.step()
                simu.step()
            yield i

    def norm(self, field):
        """
            compute the L2 norm of the obtained trajectory
        :param field:
        :return:
        """
        return np.sqrt(np.sum(np.square(field)))

    def run_test_case(self, graphs):
        """
            Run the test case
        :return:
        """
        self.reality.compute()
        self.compute(self.simulation)
        self.compute(self.kalman)

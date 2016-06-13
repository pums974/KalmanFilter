#!/usr/bin/python2
# coding=utf-8
"""
    This is a skeleton for test case
"""
import random
import numpy as np
from kalman import KalmanFilter


class SkelReality(object):
    """
        This class contains the analytical solution of our problem
        It also provide way to get a noisy field around this solution
    """
    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _noiselevel):
        self.noiselevel = _noiselevel
        self.field = np.zeros([0])

    @property
    def getsol(self):
        """
            Extract the solution, here it's trivial
        :return: the analytical solution
        """
        return self.field

    @property
    def getsolwithnoise(self):
        """
            Get a noisy field around the analytical solution
        :return: a noisy field
        """
        # field_with_noise = np.zeros(self.field.shape)
        # for i in range(self.field.shape):
        #     field_with_noise[i] = random.gauss(self.field[i], self.noiselevel)
        field_with_noise = np.array([random.gauss(d, self.noiselevel)
                                     for d in self.field.flat]).reshape(self.field.shape)
        return field_with_noise

    @property
    def compute(self):
        """
            Generator : each call gives you next step of the analytical solution
        """
        yield 0


class SkelSimulation(object):
    """
    This class contains everything for the simulation
    """
    # ---------------------------------METHODS-----------------------------------
    def __init__(self):
        self.field = np.zeros([0])
        self.Mat = np.array([[]])
        self.rhs = np.array([])

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


class SkelKalmanWrapper(object):
    """
        This class is use around the simulation to apply the kalman filter
    """
    def __init__(self, _reality, _sim):
        self.reality = _reality
        self.kalsim = _sim
        _M = np.array([[]])  # Observation matrix.
        self.kalman = KalmanFilter(self.kalsim, _M)
        self.kalman.S = np.eye(self.kalman.size_s) * 0.2  # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * 0.2  # Estimated error in measurements.
        self.kalman.Q = np.eye(self.kalman.size_s) * 0.  # Estimated error in process.

    def setmes(self, field):
        """
            Reshape noisy field and gives it to the kalman filter
        :param field:
        """
        self.kalman.Y = self.kalman.M.dot(np.reshape(field, self.kalman.size_s))

    @property
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
        self.setsol(self.kalsim.getsol)
        self.setmes(self.reality.getsolwithnoise)
        self.kalman.apply()
        self.kalsim.setsol(self.getsol)


class EDP(object):
    """
        This class contains everything for this test case :
        * The analytical solution
        * A simulation
        * A filtered simulation
        * how to plot the results
    """
    noiselevel = 20.

    def __init__(self):
        self.reality = SkelReality(self.noiselevel)
        self.simulation = SkelSimulation()
        self.kalsim = SkelSimulation()
        self.kalman = SkelKalmanWrapper(self.reality, self.kalsim)

    def compute(self, simu):
        """
            Compute the next step of a simulation (filtered or not)
        :param simu: the simulation to perform
        :return: interation number
        """
        pass

    def norm(self, field):
        """
            compute the L2 norm of the obtained trajectory
        :param field:
        :return:
        """
        return np.sqrt(np.sum(np.square(field)))

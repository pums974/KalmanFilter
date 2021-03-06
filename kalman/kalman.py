#!/usr/bin/python2
# coding=utf-8
"""
    This is our implementation of the Kalman filter
"""
from __future__ import print_function, absolute_import
import numpy as np
import sys
import matplotlib.pyplot as plt
try:
    from tools import gc_clean
    if sys.version_info < (3, ):
        from libs.fortran_libs_py2 import kalman_apply_f, kalman_apply_obs_f
    else:
        from libs.fortran_libs_py3 import kalman_apply_f, kalman_apply_obs_f
except:
    from kalman.tools import gc_clean
    if sys.version_info < (3, ):
        from kalman.libs.fortran_libs_py2 import kalman_apply_f, kalman_apply_obs_f
    else:
        from kalman.libs.fortran_libs_py3 import kalman_apply_f, kalman_apply_obs_f

use_fortran = True

if sys.version_info < (3, ):
    range = xrange


class KalmanFilter(object):
    """
        This contains everything for our Kalman filter independently of the test case
    """

    def __init__(self, _simulation, _M):
        self.simulation = _simulation
        self.size_s = _simulation.Mat.shape[0]
        self.size_o = _M.shape[0]
        self.Phi = np.zeros([self.size_s, self.size_s])
        self.M = np.zeros([self.size_o, self.size_s])

        self.Phi = _simulation.Mat  # State transition matrix.
        self.M = _M  # Observation matrix.
        # Estimated error in measurements.
        self.R = np.zeros([self.size_o, self.size_o])
        # Estimated error in process.
        self.Q = np.zeros([self.size_s, self.size_s])
        # Initial covariance estimate.
        self.S = np.zeros([self.size_s, self.size_s])
        self.X = np.zeros(self.size_s)
        self.Y = np.zeros(self.size_o)
        self.Id = np.eye(self.size_s)
        self.counter = 0
        self.counterplot = -1
        gc_clean()

    @staticmethod
    def plot_matrix(matrix):
        """
        Plot a matrix
        :param matrix:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.set_aspect('equal')
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.coolwarm)
        # labelsx = range(0, len(matrix[0]), int(len(matrix[0])/10))
        # labelsy = range(0, len(matrix[1]), int(len(matrix[1])/10))
        # plt.xticks(labelsx)
        # plt.yticks(labelsy)
        plt.colorbar()
        plt.show()

    def apply(self):
        """
            apply the kalman filter
        """
        self.counter += 1
        if use_fortran and self.counter != self.counterplot:
            self.S, X = kalman_apply_f(self.Phi, self.S, self.Q, self.M,
                                       self.R, self.Y, self.X)
            self.X = X.flatten()
        else:
            # -------------------------Prediction step-------------------------
            self.S = self.Phi.dot(self.S).dot(np.transpose(self.Phi)) + self.Q

            # ------------------------Observation step-------------------------
            innovation_covariance = self.M.dot(self.S).dot(np.transpose(
                self.M)) + self.R
            innovation = self.Y - self.M.dot(self.X)

            # ---------------------------Update step---------------------------
            K = self.S.dot(np.transpose(self.M)).dot(np.linalg.inv(
                innovation_covariance))
            self.X = self.X + K.dot(innovation)

            self.S = (self.Id - K.dot(self.M)).dot(self.S)

class KalmanFilterObservator(object):
    """
        This contains everything for our Kalman filter independently of the test case
    """

    def __init__(self, _simulation, _M):
        self.simulation = _simulation
        self.size_s = _simulation.Mat.shape[0]
        self.size_o = _M.shape[0]
        self.Phi = np.empty([self.size_s])
        self.M = np.empty([self.size_s])

        for i in range(self.size_s):
            self.Phi[i] = _simulation.Mat[i, i]  # State transition matrix.
        self.M = _M  # Observation matrix.
        # Estimated error in measurements.
        self.R = np.empty([self.size_o])
        # Estimated error in process.
        self.Q = np.empty([self.size_s])
        # Initial covariance estimate.
        self.S = np.empty([self.size_s])
        self.X = np.empty(self.size_s)
        self.Y = np.empty(self.size_o)
        self.Id = np.eye(self.size_s)
        self.counter = 0
        self.counterplot = -1
        
#        if self.M != self.Id:
#            print("M must be Id in order to use kalman in observator mode")
#            exit()

        gc_clean()

    @staticmethod
    def plot_matrix(matrix):
        """
        Plot a matrix
        :param matrix:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.set_aspect('equal')
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.coolwarm)
        # labelsx = range(0, len(matrix[0]), int(len(matrix[0])/10))
        # labelsy = range(0, len(matrix[1]), int(len(matrix[1])/10))
        # plt.xticks(labelsx)
        # plt.yticks(labelsy)
        plt.colorbar()
        plt.show()

    def apply(self):
        """
            apply the kalman filter
        """
        self.counter += 1
        if use_fortran and self.counter != self.counterplot:
            self.S, self.X = kalman_apply_obs_f(self.Phi, self.S, self.Q,
                                       self.R, self.Y, self.X)
        else:
            for i in range(self.size_s):
                # -------------------------Prediction step-------------------------
                self.S[i] = self.Phi[i]*self.S[i]*self.Phi[i] + self.Q[i]

                # ------------------------Observation step-------------------------
                innovation_covariance = self.S[i] + self.R[i]
                innovation = self.Y[i] - self.X[i]

                # ---------------------------Update step---------------------------
                K = self.S[i] / innovation_covariance
                self.X[i] = self.X[i] + K * innovation
                self.S[i] = (1. - K) * self.S[i]

            if self.counter == self.counterplot:
                self.plot_matrix(self.S)

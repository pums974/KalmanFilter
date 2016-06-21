#!/usr/bin/python2
# coding=utf-8
"""
    This is our implementation of the Kalman filter
"""
from __future__ import print_function, absolute_import
import numpy as np
import sys
import matplotlib.pyplot as plt
if sys.version_info < (3,):
    from kalman.libs.fortran_libs_py2 import kalman_apply_f
else:
    from kalman.libs.fortran_libs_py3 import kalman_apply_f


use_fortran = True


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
        self.R = np.zeros([self.size_o, self.size_o])  # Estimated error in measurements.
        self.Q = np.zeros([self.size_s, self.size_s])  # Estimated error in process.
        self.S = np.zeros([self.size_s, self.size_s])  # Initial covariance estimate.
        self.X = np.zeros(self.size_s)
        self.Y = np.zeros(self.size_o)
        self.Id = np.eye(self.size_s)
        self.counter = 0
        self.counterplot = -1

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
            self.S, X = kalman_apply_f(self.Phi, self.S, self.Q, self.M, self.R, self.Y, self.X)
            self.X = X.flatten()
        else:
            # -------------------------Prediction step-----------------------------
            self.S = self.Phi.dot(self.S).dot(np.transpose(self.Phi)) + self.Q

            # ------------------------Observation step-----------------------------
            innovation_covariance = self.M.dot(self.S).dot(np.transpose(self.M)) + self.R
            innovation = self.Y - self.M.dot(self.X)

            # ---------------------------Update step-------------------------------
            K = self.S.dot(np.transpose(self.M)).dot(np.linalg.inv(innovation_covariance))
            self.X = self.X + K.dot(innovation)

            self.S = (self.Id - K.dot(self.M)).dot(self.S)

            if self.counter == self.counterplot:
                self.plot_matrix(self.S)

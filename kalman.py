#!/usr/bin/python2
# coding=utf-8
"""
    This is our implementation of the Kalman filter
"""
import numpy as np


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

    def apply(self):
        """
            apply the kalman filter
        """
        # -------------------------Prediction step-----------------------------
        self.S = self.Phi.dot(self.S).dot(np.transpose(self.Phi)) + self.Q

        # ------------------------Observation step-----------------------------
        innovation_covariance = self.M.dot(self.S).dot(np.transpose(self.M)) + self.R
        innovation = self.Y - self.M.dot(self.X)

        # ---------------------------Update step-------------------------------
        K = self.S.dot(np.transpose(self.M)).dot(np.linalg.inv(innovation_covariance))
        self.X = self.X + K.dot(innovation)
        self.S = (self.Id - K.dot(self.M)).dot(self.S)

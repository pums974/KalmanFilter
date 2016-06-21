#!/usr/bin/python2
# coding=utf-8
"""
    Drop will solve the problem
    a = - g - h v
    and use the Kalman filter to mix
    1st order simulation and noisy measurements
    In order to produce a trajectory closer to the analytic solution
    The result is one plot
"""

from __future__ import print_function, absolute_import
import math
import numpy as np
import matplotlib.pyplot as plt
from kalman.kalman import KalmanFilter
from kalman.skeleton import *
import random


class Reality(SkelReality):
    """
        This class contains the analytical solution of our problem
        It also provide way to get a noisy field around this solution
    """
    g = - 9.81
    power = 100.
    dir = 45.
    frot = 0.0

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _dt, _noiselevel):
        SkelReality.__init__(self, _noiselevel, _dt, [4])
        self.init = np.array([0.,
                              self.power * math.cos(self.dir * math.pi / 180.),
                              0.,
                              self.power * math.sin(self.dir * math.pi / 180.)])

    def step(self):
        """
            Compute the analytical solution
        """
        self.field = np.zeros([self.size])
        self.It += 1
        time = self.It * self.dt
        # avec frottement
        # This is the solution of
        # a = - g - h * v
        # x(0) = (0, 0)
        # c1 = 0.
        # c2 = self.init[1] + c1
        # c3 = math.exp(-self.frot * time)
        # c4 = self.init[0] + c2 / self.frot
        # self.field[0] = c4 - c2 / self.frot * c3 - c1 * time
        # self.field[1] = c2 * math.exp(-self.frot * time) - c1
        # c1 = - self.g / self.frot
        # c2 = self.init[3] + c1
        # c4 = self.init[2] + c2 / self.frot
        # self.field[2] = c4 - c2 / self.frot * c3 - c1 * time
        # self.field[3] = c2 * math.exp(-self.frot * time) - c1

        # sans frottement
        # This is the solution of
        # a = - g
        # x(0) = (0, 0)
        self.field[0] = self.init[0] + self.init[1] * time
        self.field[1] = self.init[1]
        self.field[2] = self.init[2] + self.init[3] * time + 0.5 * self.g * time * time
        self.field[3] = self.init[3] + self.g * time


class Simulation(SkelSimulation):
    """
    This class contains everything for the simulation
    """
    # --------------------------------PARAMETERS--------------------------------
    power = 100
    dir = 45
    g = - 9.81
    frot = 0.0

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _dt, _noiselevel):
        SkelSimulation.__init__(self, _noiselevel, _dt, [4])

        # x_n+1 = x_n + dt * v_n + O(dt^2)
        # v_n+1 = (1 - h*dt) v_n - dt*g + O(dt^2)
        c1 = 1.
        c2 = self.dt
        c3 = 0.
        c5 = 1. - self.frot * self.dt
        c6 = self.dt

        self.Mat = np.array([[c1, c2, 0, 0],
                             [0, c5, 0, 0],
                             [0, 0, c1, c2],
                             [0, 0, 0, c5]])
        self.rhs = np.array([0.,
                             0.,
                             +c3 * self.g,
                             +c6 * self.g])

    def step(self):
        """
            Increment through the next time step of the simulation.
        """
        c3 = 0.
        c6 = self.dt
        g1 = random.gauss(0, self.noiselevel)
        g2 = random.gauss(self.g, self.noiselevel)
        self.rhs = np.array([+c3 * g1,
                             +c6 * g1,
                             +c3 * g2,
                             +c6 * g2])
        SkelSimulation.step(self)


class KalmanWrapper(SkelKalmanWrapper):
    """
        This class is use around the simulation to apply the kalman filter
    """
    def __init__(self, _reality, _sim):
        SkelKalmanWrapper.__init__(self, _reality, _sim)
        self.kalman.S = np.eye(self.kalman.size_s) * 0.  # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * self.reality.noiselevel ** 2  # Estimated error in measurements.

        G = np.array([[_sim.dt],
                      [_sim.dt ** 2 * 0.5],
                      [_sim.dt],
                      [_sim.dt**2 * 0.5]])
        self.kalman.Q = G.dot(np.transpose(G)) * self.kalsim.noiselevel ** 2   # Estimated error in process.
        # self.kalman.Q = np.eye(self.kalman.size_s) * self.kalsim.noiselevel ** 2  # Estimated error in process.

    def getwindow(self):
        """
            Produce the observation matrix : designate what we conserve of the noisy field
        :return: observation matrix
        """

        M = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])

        # M = np.eye(self.kalsim.size)
        return M

    def setmes(self, field):
        """

        :param field:
        """
        self.kalman.Y = self.kalman.M.dot(field)

    def getsol(self):
        """
            Extract the solution from the kalman filter
            Its dim goes from (n,1) to (n)
        :return:
        """
        return self.kalman.X.flatten()

    def setsol(self, field):
        """
            Set the current solution, we have to add a dummy dimension for the kalman filter
        :param field: field
        """
        self.kalman.X = field
        self.kalsim.setsol(field)


class Drop(EDP):
    """
        This class contains everything for this test case :
        * The analytical solution
        * A simulation
        * A filtered simulation
        * how to plot the results
    """
    Tfin = 15.
    nIt = 300
    noise_real = 20
    noise_sim = 10
    dt = Tfin / nIt
    name = "Drop"

    def __init__(self):
        EDP.__init__(self)
        print("Norme L2 |  mesure  |   simu   |  kalman")

        self.reinit()

    def reinit(self):
        """
            Reinit everything
        :return:
        """
        self.simulation = Simulation(self.dt, self.noise_sim)
        self.kalsim = Simulation(self.dt, self.noise_sim)

        self.dt = self.simulation.dt
        self.reality = Reality(self.dt, self.noise_real)

        self.simulation.nIt = self.nIt
        self.kalsim.nIt = self.nIt
        self.reality.nIt = self.nIt

        self.kalman = KalmanWrapper(self.reality, self.kalsim)

    @staticmethod
    def plot(field):
        """
            plot one trajectory
        :param field: field
        """
        fig = plt.figure()
        ax = fig.gca()
        plt.xlabel('X position')
        plt.ylabel('Y position')
        ax.set_xlim(0, 600)
        # plt.title('Measurement of a Cannonball in Flight')
        plt.plot(field[:, 0], field[:, 2])
        # plt.legend(('reality', 'simulated', 'measured', 'kalman'))
        plt.show()

    @staticmethod
    def plotall(field_ref, field_mes, field_sim, field_kal):
        """
            Plot all the trajectory on the same graph
        :param field_ref: the analytical solution
        :param field_mes: a noisy trajectory around the analytical solution
        :param field_sim: a simulated trajectory
        :param field_kal: a filtered trajectory
        """
        fig = plt.figure()
        ax = fig.gca()
        plt.xlabel('X position')
        plt.ylabel('Y position')
        ax.set_xlim(0, 1200)
        ax.set_ylim(-300, 400)
        # plt.title('Measurement of a Cannonball in Flight')
        plt.plot(field_ref[:, 0], field_ref[:, 2], '-',
                 field_mes[:, 0], field_mes[:, 2], ':',
                 field_sim[:, 0], field_sim[:, 2], '-',
                 field_kal[:, 0], field_kal[:, 2], '--')
        plt.legend(('reality', 'measured', 'simulated', 'kalman'))
        plt.show()

    def run_test_case(self, graphs):
        """
            Run the test case
        :return:
        """
        self.reinit()

        Sol_ref = np.zeros([self.nIt, 4])
        Sol_mes = np.zeros([self.nIt, 4])
        Sol_sim = np.zeros([self.nIt, 4])
        Sol_kal = np.zeros([self.nIt, 4])

        # ----------------- Compute reality and measurement --------------------
        for it in self.compute(self.reality):
            Sol_ref[it, :] = self.reality.getsol()
            Sol_mes[it, :] = self.reality.getsolwithnoise()

        Norm_ref = self.norm(Sol_ref)
        Err_mes = self.norm(Sol_ref - Sol_mes) / Norm_ref

        # ------------------------ Compute simulation without Kalman ----------------------------
        self.reality.reinit()
        # Initial solution
        self.simulation.setsol(self.reality.getsol())

        for it in self.compute(self.simulation):
            Sol_sim[it, :] = self.simulation.getsol()

        Err_sim = self.norm(Sol_ref - Sol_sim) / Norm_ref

        # ------------------------ Compute simulation with Kalman ----------------------------

        self.reality.reinit()
        # Initial solution
        self.kalman.setsol(self.reality.getsol())

        for it in self.compute(self.kalman):
            Sol_kal[it, :] = self.kalman.getsol()

        Err_kal = self.norm(Sol_ref - Sol_kal) / Norm_ref

        # ------------------------ Final output ----------------------------

        print("%8.2e | %8.2e | %8.2e | %8.2e" % (Norm_ref, Err_mes, Err_sim, Err_kal))
        if graphs:
            self.plotall(Sol_ref, Sol_mes, Sol_sim, Sol_kal)

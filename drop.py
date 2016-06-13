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

from __future__ import print_function
import math
import numpy as np
import matplotlib.pyplot as plt
from kalman import KalmanFilter
from skeleton import *


class Reality(SkelReality):
    """
        This class contains the analytical solution of our problem
        It also provide way to get a noisy field around this solution
    """
    g = - 9.81
    power = 100.
    dir = 45.
    frot = 0.1

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _tfin, _it, _noiselevel):
        SkelReality.__init__(self, _noiselevel)
        self.noiselevel = _noiselevel
        self.Tfin = _tfin
        self.It = _it
        self.dt = _tfin / _it
        self.field = np.zeros([4])
        self.init = np.array([0.,
                              self.power * math.cos(self.dir * math.pi / 180.),
                              0.,
                              self.power * math.sin(self.dir * math.pi / 180.)])

    @property
    def compute(self):
        """
            Generator : each call gives you next step of the analytical solution
        """
        for i in range(self.It):
            time = i * self.dt
            # avec frottement
            c1 = 0.
            c2 = self.init[1] + c1
            c3 = math.exp(-self.frot * time)
            c4 = self.init[0] + c2 / self.frot
            self.field[0] = c4 - c2 / self.frot * c3 - c1 * time
            self.field[1] = c2 * math.exp(-self.frot * time) - c1
            c1 = - self.g / self.frot
            c2 = self.init[3] + c1
            c4 = self.init[2] + c2 / self.frot
            self.field[2] = c4 - c2 / self.frot * c3 - c1 * time
            self.field[3] = c2 * math.exp(-self.frot * time) - c1

            # sans frottement
            # self.field[0] = self.init[0] + self.init[1] * time
            # self.field[1] = self.init[1]
            # self.field[2] = self.init[2] + self.init[3] * time + 0.5 * self.g * time * time
            # self.field[3] = self.init[3] + self.g * time

            yield i


class Simulation(SkelSimulation):
    """
    This class contains everything for the simulation
    """
    # --------------------------------PARAMETERS--------------------------------
    power = 100
    dir = 45
    g = 9.81 + 1.
    frot = 0.1 - 0.01

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _tfin, _it):
        SkelSimulation.__init__(self)
        self.Tfin = _tfin
        self.It = _it
        self.dt = _tfin / _it
        self.field = np.zeros([_it, 4])
        # too precise
        # c0 = (2.+self.frot*self.dt)
        # c1 = (2.+self.frot*self.dt)/c0
        # c2 = (2.*self.dt)/c0
        # c3 = self.dt*self.dt / c0
        # c4 = 1.+self.frot*self.dt
        # c5 = 1./c4
        # c6 = self.dt/c4

        # still too precise
        # c4 = 1.+self.frot*self.dt
        # c5 = 1./c4
        # c6 = self.dt/c4
        # c1 = 1.
        # c2 = self.dt*c5
        # c3 = self.dt*c6

        # can't be less precise
        c5 = 1. - self.frot * self.dt
        c6 = self.dt
        c1 = 1.
        c2 = self.dt * c5
        c3 = self.dt * c6

        self.Mat = np.array([[c1, c2, 0, 0],
                             [0, c5, 0, 0],
                             [0, 0, c1, c2],
                             [0, 0, 0, c5]])
        self.rhs = np.array([0,
                             0,
                             -c3 * self.g,
                             -c6 * self.g])


class KalmanWrapper(SkelKalmanWrapper):
    """
        This class is use around the simulation to apply the kalman filter
    """
    def __init__(self, _reality, _sim):
        SkelKalmanWrapper.__init__(self, _reality, _sim)
        self.It = _sim.It
        _M = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0]])  # Observation matrix.

        self.kalman = KalmanFilter(self.kalsim, _M)
        self.kalman.S = np.eye(self.kalman.size_s) * 0.2  # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * 0.2  # Estimated error in measurements.
        self.kalman.Q = np.eye(self.kalman.size_s) * 0.  # Estimated error in process.

    def setmes(self, field):
        """

        :param field:
        """
        self.kalman.Y = self.kalman.M.dot(field[..., np.newaxis])

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
        self.kalman.X = field[..., np.newaxis]
        self.kalsim.setsol(field)

    def step(self):
        """
            Increment to the next step of the simulation
        """
        self.kalsim.step()
        self.setsol(self.kalsim.getsol())
        self.kalman.apply()
        self.kalsim.setsol(self.getsol())


class Drop(EDP):
    """
        This class contains everything for this test case :
        * The analytical solution
        * A simulation
        * A filtered simulation
        * how to plot the results
    """
    Tfin = 15.
    Iterations = 200
    noiselevel = 20.

    def __init__(self):
        EDP.__init__(self)
        self.reality = Reality(self.Tfin, self.Iterations, self.noiselevel)
        self.simulation = Simulation(self.Tfin, self.Iterations)
        self.kalsim = Simulation(self.Tfin, self.Iterations)
        self.kalman = KalmanWrapper(self.reality, self.kalsim)

    def compute(self, simu):
        """
            Compute the next step of a simulation (filtered or not)
        :param simu: the simulation to perform
        :return: interation number
        """
        for i in range(simu.It):
            if i > 0:
                simu.step()
            yield i

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
        ax.set_xlim(0, 600)
        ax.set_ylim(-300, 400)
        # plt.title('Measurement of a Cannonball in Flight')
        plt.plot(field_ref[:, 0], field_ref[:, 2], '-',
                 field_mes[:, 0], field_mes[:, 2], ':',
                 field_sim[:, 0], field_sim[:, 2], '-',
                 field_kal[:, 0], field_kal[:, 2], '--')
        plt.legend(('reality', 'measured', 'simulated', 'kalman'))
        plt.show()


# ------------------------ Begin program ----------------------------

edp = Drop()
Sol_ref = np.zeros([edp.Iterations, 4])
Sol_mes = np.zeros([edp.Iterations, 4])
Sol_sim = np.zeros([edp.Iterations, 4])
Sol_kal = np.zeros([edp.Iterations, 4])

# ----------------- Compute reality and measurement --------------------
for it in edp.reality.compute:
    Sol_ref[it, :] = edp.reality.getsol
    Sol_mes[it, :] = edp.reality.getsolwithnoise

Norm_ref = edp.norm(Sol_ref)
# edp.plot(Sol_ref)

Err_mes = edp.norm(Sol_ref - Sol_mes) / Norm_ref
print("Norme H1 de la mesure", Err_mes)
# edp.plot(Sol_mes)

# ------------------------ Compute simulation without Kalman ----------------------------

# Bad initial solution
edp.simulation.setsol(Sol_ref[0, :])

for it in edp.compute(edp.simulation):
    Sol_sim[it, :] = edp.simulation.getsol()

Err_sim = edp.norm(Sol_ref - Sol_sim) / Norm_ref
print("Norme H1 de la simu", Err_sim)
# edp.plot(Sol_sim)

# ------------------------ Compute simulation with Kalman ----------------------------

# Bad initial solution
edp.kalsim.setsol(Sol_mes[0, :])
edp.kalman.setmes(Sol_mes[1, :])

it = 0
for it in edp.compute(edp.kalman):
    Sol_kal[it, :] = edp.kalman.getsol()
    if it < edp.Iterations-1:
        edp.kalman.setmes(Sol_mes[it + 1, :])
    # edp.plot(Sol_kal)

Err_kal = edp.norm(Sol_ref - Sol_kal) / Norm_ref
print("Norme H1 de la simu filtre", Err_kal)
# edp.plot(Sol_kal)

# ------------------------ Final plot ----------------------------

edp.plotall(Sol_ref, Sol_mes, Sol_sim, Sol_kal)

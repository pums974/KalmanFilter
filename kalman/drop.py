#!/usr/bin/python2
# coding=utf-8
"""
This is a test case for kalman filter.

Drop will solve the problem
a = - g - h v
and use the Kalman filter to mix
1st order simulation and noisy measurements
In order to produce a trajectory closer to the analytic solution
The result is one plot
"""
from __future__ import absolute_import, print_function

import math
import random

import matplotlib.pyplot as plt
import numpy as np

try:
    from kalman.kalman import KalmanFilter
    from kalman.skeleton import *
    if sys.version_info < (3, ):
        from kalman.libs.fortran_libs_py2 import gauss_f, gauss_f2, noisy_q, norm_l2
    else:
        from kalman.libs.fortran_libs_py3 import gauss_f, gauss_f2, noisy_q, norm_l2
except:
    from kalman import KalmanFilter
    from skeleton import *
    if sys.version_info < (3, ):
        from libs.fortran_libs_py2 import gauss_f, gauss_f2, noisy_q, norm_l2
    else:
        from libs.fortran_libs_py3 import gauss_f, gauss_f2, noisy_q, norm_l2

use_fortran = True

if sys.version_info < (3, ):
    range = xrange


time_it = 0


class Reality(SkelReality):
    """
    This class contains the analytical solution of our problem.

    It also provide way to get a noisy field around this solution
    """

    g = -9.81
    power = 100.
    dir = 45.
    frot = 0.0

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _dt, _noiselevel):
        """Initilisation."""
        SkelReality.__init__(self, _noiselevel, _dt, [4])
        self.init = np.array([0.,
                              self.power * math.cos(self.dir * math.pi / 180.),
                              0.,
                              self.power * math.sin(self.dir * math.pi / 180.)])

    def step(self):
        """Compute the analytical solution."""
        # self.field = np.zeros([self.size])
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
        self.field[2] = self.init[2] + self.init[3] * time \
            + 0.5 * self.g * time * time
        self.field[3] = self.init[3] + self.g * time


class Simulation(SkelSimulation):
    """This class contains everything for the simulation."""

    # --------------------------------PARAMETERS--------------------------------
    power = 100
    dir = 45
    g = -9.81
    frot = 0.0

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _dt, _noiselevel):
        """Initilisation."""
        SkelSimulation.__init__(self, _noiselevel, _dt, [4])

        # x_n+1 = x_n + dt * v_n + O(dt^2)
        # v_n+1 = (1 - h*dt) v_n - dt*g + O(dt^2)
        c1 = 1.
        c2 = self.dt
        c3 = 0.5 * self.dt * self.dt
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
        """Increment through the next time step of the simulation."""
        c3 = 0.5 * self.dt * self.dt
        c6 = self.dt
        if use_fortran:
            g1 = gauss_f([0.], self.noiselevel)[0]
            g2 = gauss_f([self.g], self.noiselevel)[0]
        else:
            g1 = random.gauss(0, self.noiselevel)
            g2 = random.gauss(self.g, self.noiselevel)
        self.rhs = np.array([+c3 * g1,
                             +c6 * g1,
                             +c3 * g2,
                             +c6 * g2])
        SkelSimulation.step(self)


class KalmanWrapper(SkelKalmanWrapper):
    """This class is use around the simulation to apply the kalman filter."""

    def __init__(self, _reality, _sim):
        """Initilisation."""
        SkelKalmanWrapper.__init__(self, _reality, _sim)

        # Initial covariance estimate.
        self.kalman.S = np.eye(self.kalman.size_s) * 0.

        # Estimated error in measurements.
        self.kalman.R = np.eye(self.kalman.size_o) * \
            self.reality.noiselevel ** 2

        # Estimated error in process.
        # G = np.array([[_sim.dt],
        #               [_sim.dt ** 2 * 0.5],
        #               [_sim.dt],
        #               [_sim.dt**2 * 0.5]])
        # self.kalman.Q = G.dot(np.transpose(G)) * self.kalnoise ** 2
        # self.kalman.Q = np.eye(self.kalman.size_s) * self.kalnoise ** 2
        # noise1 = (_sim.dt * _sim.noiselevel) ** 2
        # noise2 = 0.
        # noise3 = noise1 / 4.
        # noise4 = 0.
        # noise5 = 0.
        # noise6 = 0.
        # self.kalman.Q = np.array([[noise1, noise2, noise3, noise4],
        #                           [noise2, noise1, noise5, noise6],
        #                           [noise3, noise5, noise1, noise2],
        #                           [noise4, noise6, noise2, noise1]])

        # print(self.kalman.Q)

        # randomize
        # self.kalman.Q = gauss_f(self.kalman.Q.flat,
        #                         self.kalman.Q[1, 1]/100).reshape(self.kalman.Q.shape)
        # self.kalman.Q = np.random.random(16).reshape(self.kalman.Q.shape)
        # self.kalman.Q *= (_sim.dt ** 2 * _sim.noiselevel ** 2) / 2.
        # self.kalman.Q += np.eye(self.kalman.size_s) * (_sim.dt ** 2 * _sim.noiselevel ** 2) / 2.
        # # enforce positivity
        # self.kalman.Q = np.abs(self.kalman.Q)
        # # enforce symmetricity
        # self.kalman.Q = (self.kalman.Q + self.kalman.Q.transpose()) * 0.5

        self.Q_acc = self.kalman.Q * 0.
        self.n_acc = 0

        self.mystep = SkelKalmanWrapper.step
        # self.mystep = self.randstep

    def getwindow(self):
        """
        Produce the observation matrix : designate what we conserve of the noisy field.

        :return: observation matrix
        """
        # M = np.array([[1, 0, 0, 0],
        #               [0, 0, 1, 0]])

        M = np.eye(self.kalsim.size)
        return M

    def setmes(self, field):
        """
        Set the measured field.

        :param field:measured field
        """
        self.kalman.Y = self.kalman.M.dot(field)

    def getsol(self):
        """
        Extract the solution from the kalman filter.

        Its dim goes from (n,1) to (n)
        """
        return self.kalman.X #.flatten()

    def setsol(self, field):
        """
        Set the current solution, we have to add a dummy dimension for the kalman filter.

        :param field: field
        """
        self.kalman.X = field
        self.kalsim.setsol(field)

    def step(self):
        """
            Compute the next step of the simulation and apply kalman filter to the result
        """
        self.mystep(self)

    @staticmethod
    def randstep(self):
        global time_it

        self.kalsim.step()
        self.setmes(self.reality.getsolwithnoise())

        old_S = self.kalman.S.copy()
        old_Q = self.kalman.Q.copy()
        old_X = self.kalsim.getsol()
        ref = self.reality.getsol()

        # Try ref Q
        mid_val = (self.kalsim.dt ** 2 * self.kalsim.noiselevel ** 2)
        # for j in range(old_Q.shape[1]):
        #     for i in range(old_Q.shape[0]):
        #         if i == j:
        #             self.kalman.Q[i, j] = mid_val
        #         else:
        #             self.kalman.Q[i, j] = 0.
        self.kalman.Q = noisy_q(0, old_Q, mid_val)

        self.kalman.X = old_X.copy()
        self.kalman.S = old_S.copy()
        self.kalman.apply()
        # err = np.sum(np.square(ref - self.kalman.X))
        err = norm_l2(ref, self.kalman.X)
        best_Q = self.kalman.Q.copy()
        err1 = err
        step = 0

        # Try around ref Q
        for it in range(1000):
            # for j in range(old_Q.shape[1]):
            #     for i in range(old_Q.shape[0]):
            #         if i == j:
            #             self.kalman.Q[i, j] = mid_val * 0.5 + np.random.random() * mid_val
            #         else:
            #             self.kalman.Q[i, j] = np.random.random() * mid_val * 0.5
            # # enforce symmetry
            # self.kalman.Q = (self.kalman.Q + self.kalman.Q.transpose()) * 0.5
            self.kalman.Q = noisy_q(1, old_Q, mid_val)

            self.kalman.X = old_X.copy()
            self.kalman.S = old_S.copy()
            self.kalman.apply()
            # err = np.sum(np.square(ref - self.kalman.X))
            err = norm_l2(ref, self.kalman.X)
            if err < err1:
                best_Q = self.kalman.Q.copy()
                err1 = err
                step = 1

        # Try around old_Q
        for it in range(1000):
            # for j in range(old_Q.shape[1]):
            #     for i in range(old_Q.shape[0]):
            #         self.kalman.Q[i, j] = old_Q[i, j] * 0.5 + np.random.random() * old_Q[i, j]
            # # enforce symmetry
            # self.kalman.Q = (self.kalman.Q + self.kalman.Q.transpose()) * 0.5
            self.kalman.Q = noisy_q(2, old_Q, mid_val)

            self.kalman.X = old_X.copy()
            self.kalman.S = old_S.copy()
            self.kalman.apply()
            # err = np.sum(np.square(ref - self.kalman.X))
            err = norm_l2(ref, self.kalman.X)
            if err < err1:
                best_Q = self.kalman.Q.copy()
                err1 = err
                step = 2

        # print(np.max((self.Q_acc - Q1)/(self.Q_acc*self.n_acc)))
        self.kalman.Q = best_Q.copy()
        self.kalman.X = old_X.copy()
        self.kalman.S = old_S.copy()
        self.Q_acc += self.kalman.Q
        self.n_acc += 1

        self.kalman.apply()
        self.kalsim.setsol(self.getsol())
        self.It = self.kalsim.It

        # self.kalman.Q = old_Q.copy()
        time_it += 1
        # print(self.Q_acc/self.n_acc, self.n_acc)
        # print(step)


class Drop(EDP):
    """
    This class contains everything for the drop test case.

    In particular it contains
    * The analytical solution
    * A simulation
    * A filtered simulation
    * how to plot the results
    """

    # Tfin = 15.
    nIt = 300
    noise_real = 20
    noise_sim = 10
    dt = 0.5
    name = "Drop"

    def __init__(self):
        """Initilisation."""
        EDP.__init__(self)
        # print("Norme L2 |  mesure  |   simu   |  kalman")

        self.reinit()
        # self.Q = self.kalman.kalman.Q

    def reinit(self):
        """Reinit everything."""

        _ = gauss_f([0], 0)
        _ = gauss_f2(1)

        self.simulation = Simulation(self.dt, self.noise_sim)
        self.kalsim = Simulation(self.dt, self.noise_sim)

        self.dt = self.simulation.dt
        self.reality = Reality(self.dt, self.noise_real)

        self.simulation.nIt = self.nIt
        self.kalsim.nIt = self.nIt
        self.reality.nIt = self.nIt

        self.kalman = KalmanWrapper(self.reality, self.kalsim)

        # self.Q = self.kalman.kalman.Q

    @staticmethod
    def plot(field):
        """
        Plot one trajectory.

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
        Plot all the trajectory on the same graph.

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
        plt.plot(field_ref[:, 0], field_ref[:, 2], 'k--',
                 field_mes[:, 0], field_mes[:, 2], '.',
                 field_sim[:, 0], field_sim[:, 2], 'b-',
                 field_kal[:, 0], field_kal[:, 2], 'r-')
        plt.legend(('reality', 'measured', 'simulated', 'kalman'))
        plt.show()

    def run_test_case(self, graphs):
        """Run the test case."""
        # self.reinit()

        Sol_ref = np.zeros([self.nIt, 4])
        Sol_mes = np.zeros([self.nIt, 4])
        Sol_sim = np.zeros([self.nIt, 4])
        Sol_kal = np.zeros([self.nIt, 4])

        # ----------------- Compute reality and measurement -------------------
        for it in self.compute(self.reality):
            Sol_ref[it, :] = self.reality.getsol()
            Sol_mes[it, :] = self.reality.getsolwithnoise()

        Norm_ref = self.norm(Sol_ref)
        Err_mes = self.norm(Sol_ref - Sol_mes) / Norm_ref

        # ------------------------ Compute simulation without Kalman ----------
        self.reality.reinit()
        # Initial solution
        self.simulation.setsol(self.reality.getsol())

        for it in self.compute(self.simulation):
            Sol_sim[it, :] = self.simulation.getsol()

        Err_sim = self.norm(Sol_ref - Sol_sim) / Norm_ref

        # ------------------------ Compute simulation with Kalman -------------

        self.reality.reinit()
        # Initial solution
        self.kalman.setsol(self.reality.getsol())

        for it in self.compute(self.kalman):
            Sol_kal[it, :] = self.kalman.getsol()

        Err_kal = self.norm(Sol_ref - Sol_kal) / Norm_ref

        # ------------------------ Final output ----------------------------

        print("%8.2e | %8.2e | %8.2e | %8.2e" %
              (Norm_ref, Err_mes, Err_sim, Err_kal))

        # self.Q = self.kalman.Q_acc / self.kalman.n_acc
        # print(Q, self.kalman.n_acc)
        # print(np.sum(Q, axis=1))
        # print(self.kalman.kalman.Q, 2)

        if graphs:
            self.plotall(Sol_ref, Sol_mes, Sol_sim, Sol_kal)

        return Err_kal


def try_noisy_q(edp):

    list_noise_real = [0.05, 0.1, 0.2, 20]
    list_noise_sim  = [0.05, 0.1, 0.2, 10]
    list_dt = [0.05, 0.1, 0.2]

    # list_noise_real = [0.05]
    # list_noise_sim  = [0.05]
    # list_dt = [0.05]

    for nr in list_noise_real:
        for ns in list_noise_sim:
            for dt in list_dt:
                edp.noise_real = nr
                edp.noise_sim = ns
                edp.dt = dt
                time_it = 0

                edp.reinit()

                noise1 = (edp.dt * edp.noise_sim) ** 2
                noise2 = 0.
                noise3 = 0.
                noise4 = 0.
                noise5 = 0.
                noise6 = 0.
                edp.kalman.kalman.Q = np.array([[noise1, noise2, noise3, noise4],
                                                [noise2, noise1, noise5, noise6],
                                                [noise3, noise5, noise1, noise2],
                                                [noise4, noise6, noise2, noise1]])
                print("noise_real      = ", nr)
                print("noise_sim       = ", ns)
                print("time step       = ", dt)
                print("number of steps = ", edp.nIt)
                print("(dt+ns)**2      = ", noise1)
                print("run without random Q (Q = Id * (dt+ns)**2)")
                print("Norme L2 |  mesure  |   simu   |  kalman")
                err1 = edp.run_test_case(False)
                # print(edp.kalman.kalman.Q)

                # edp.reinit()
                #
                # noise1 = (edp.dt * edp.noise_sim) ** 2
                # noise2 = noise1 / 4.
                # noise3 = noise1 / 4.
                # noise4 = noise1 / 4.
                # noise5 = noise1 / 4.
                # noise6 = noise1 / 4.
                # edp.kalman.kalman.Q = np.array([[noise1, noise2, noise3, noise4],
                #                                 [noise2, noise1, noise5, noise6],
                #                                 [noise3, noise5, noise1, noise2],
                #                                 [noise4, noise6, noise2, noise1]])
                # err1 = edp.run_test_case(False)
                # print(edp.kalman.kalman.Q)

                edp.reinit()
                noise1 = (edp.dt * edp.noise_sim) ** 2
                noise2 = 0.
                noise3 = 0.
                noise4 = 0.
                noise5 = 0.
                noise6 = 0.
                edp.kalman.kalman.Q = np.array([[noise1, noise2, noise3, noise4],
                                                [noise2, noise1, noise5, noise6],
                                                [noise3, noise5, noise1, noise2],
                                                [noise4, noise6, noise2, noise1]])
                edp.kalman.mystep = edp.kalman.randstep
                print("run with random Q (2000 rand Q is tried at each time step)")
                print("Norme L2 |  mesure  |   simu   |  kalman")
                err1 = edp.run_test_case(False)
                print("normalized average Q obtained")
                print(edp.kalman.Q_acc/edp.kalman.n_acc/noise1)

    # Q_acc = edp.Q
    # n_acc = 1
    # err = 1000
    # print(edp.Q / n_acc)
    # for it in range(100000):
    #     err1 = edp.run_test_case(False)
    #     if err1 < err:
    #         Q_acc += edp.Q
    #         n_acc += 1
    #         err = err1
    #         # print(err)
    #         print(Q_acc / n_acc)
    #     edp.reinit()
    #     edp.kalman.kalman.Q = Q_acc / n_acc
    #     # Q_acc = edp.Q * 0.
    #     # n_acc = 0


if __name__ == "__main__":
    edp = Drop()
    try_noisy_q(edp)

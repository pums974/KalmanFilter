#!/usr/bin/python2
# coding=utf-8
"""
    Convection will solve a simple convection problem
    dt T + a . grad T = 0
    and use the Kalman filter to mix
    1st order upwind simulation and noisy measurements
    In order to produce converged field closer to the analytic solution
    The result is :
    * one plot with the analytic solution
    * one plot with the noisy measurements
    * one animation of the simulation during time convergence
    * one animation of the filtered simulation during time convergence
"""
from __future__ import print_function, absolute_import

import numpy as np

from kalman.kalman import KalmanFilter
from kalman.skeleton import *
from kalman.grid import Grid_Upwind
import math


class Reality(SkelReality):
    """
        This class contains the analytical solution of our problem
        It also provide way to get a noisy field around this solution
    """
    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _grid, _dtheta, _noiselevel, _dt):
        self.grid = _grid
        SkelReality.__init__(self, _noiselevel, _dt, _grid.shape)
        self.dtheta = _dtheta

    def step(self):
        """
            Compute the analytical solution
        """
        self.field = np.zeros([self.grid.nx, self.grid.ny])
        self.It += 1
        time = self.It * self.dt
        width = 2.
        x0 = self.grid.Lx * math.cos(time * self.dtheta * 0.5) * 0.25
        y0 = self.grid.Lx * math.sin(time * self.dtheta * 0.5) * 0.25
        tmp1 = np.square(self.grid.coordx - x0)
        tmp2 = np.square(self.grid.coordy - y0)
        self.field = np.maximum(1. - (tmp1 + tmp2) * width, self.field)


class Simulation(SkelSimulation):
    """
    This class contains everything for the simulation
    """
    # --------------------------------PARAMETERS--------------------------------
    cfl = 1. / 2.

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _grid, _noiselevel):
        self.grid = _grid
        maxa = np.max(_grid.velofield)
        self.dt = self.cfl * min([_grid.dx, _grid.dy]) / maxa

        SkelSimulation.__init__(self, _noiselevel, self.dt, _grid.shape)

        nx = _grid.nx
        ny = _grid.ny
        indx = self.grid.indx
        # compute matrix
        # time
        self.Mat = np.eye(self.size, self.size)
        # Updwind
        for i in range(nx):
            for j in range(ny):
                self.field = np.zeros([nx, ny])
                self.field[i, j] = self.dt
                self.Mat[indx(i, j)] -= np.reshape(self.grid.derx(self.field), [self.size])
                self.Mat[indx(i, j)] -= np.reshape(self.grid.dery(self.field), [self.size])
        self.Mat = self.Mat.transpose()
        # rhs and boundary conditions
        for j in range(ny):
            self.Mat[indx(0, j)] = np.zeros([self.size])
            self.Mat[indx(nx - 1, j)] = np.zeros([self.size])
            self.Mat[indx(0, j)][indx(0, j)] = 1.
            self.Mat[indx(nx - 1, j)][indx(nx - 1, j)] = 1.
            self.rhs[indx(0, j)] = 0.
            self.rhs[indx(nx - 1, j)] = 0.
        for i in range(nx):
            self.Mat[indx(i, 0)] = np.zeros([self.size])
            self.Mat[indx(i, ny - 1)] = np.zeros([self.size])
            self.Mat[indx(i, 0)][indx(i, 0)] = 1.
            self.Mat[indx(i, ny - 1)][indx(i, ny - 1)] = 1.
            self.rhs[indx(i, 0)] = 0.
            self.rhs[indx(i, ny - 1)] = 0.
        self.field = np.zeros([self.size])

    def getsol(self):
        """
            Get current solution
        :return: current solution
        """
        return np.reshape(self.field, [self.grid.nx, self.grid.ny])

    def setsol(self, field):
        """
            Set current solution
        :param field: field
        """
        self.field = np.reshape(field, [self.size])


class KalmanWrapper(SkelKalmanWrapper):
    """
        This class is use around the simulation to apply the kalman filter
    """
    def __init__(self, _reality, _sim):
        SkelKalmanWrapper.__init__(self, _reality, _sim)
        self.kalman.S = np.eye(self.kalman.size_s) * 0.  # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * self.reality.noiselevel ** 2  # Estimated error in measurements.

        indx = self.kalsim.grid.indx
        # G = np.zeros([self.kalman.size_s, 1])
        # for i in range(self.kalsim.grid.nx):
        #     for j in range(self.kalsim.grid.ny):
        #         G[indx(i, j)] = self.kalsim.grid.dx ** 2 / 2. \
        #                         + self.kalsim.grid.dy ** 2 / 2. \
        #                         + self.kalsim.dt ** 2 / 2.
        # self.kalman.Q = G.dot(np.transpose(G)) * self.kalsim.noiselevel ** 2  # Estimated error in process.
        self.kalman.Q = np.eye(self.kalman.size_s) * self.kalsim.noiselevel ** 2  # Estimated error in process.

    def getwindow(self):
        """
            Produce the observation matrix : designate what we conserve of the noisy field
        :return: observation matrix
        """

        indx = self.kalsim.grid.indx
        M = np.eye(self.kalsim.size)
        return M

    def getsol(self):
        """
            Extract the solution from kalman filter : has to reshape it
        :return: current solution
        """
        return np.reshape(self.kalman.X, [self.kalsim.grid.nx, self.kalsim.grid.ny])

    def setsol(self, field):
        """
            Set the current solution, for both the simulation and the kalman filter
        :param field: field
        """
        self.kalman.X = np.reshape(field, self.kalman.size_s)
        self.kalsim.setsol(field)


class Convection(EDP):
    """
        This class contains everything for this test case :
        * The analytical solution
        * A simulation
        * A filtered simulation
        * how to plot the results
    """
    T_fin = 20.
    nIt = 0
    noise_real = 0.065
    noise_sim = 0.007
    # noise_sim = 0.005
    Lx = 4.
    Ly = 4.
    nx = 17
    ny = 17
    dtheta = 2. * math.pi / 10.
    dt = 0.
    name = "Convection"

    def __init__(self):
        EDP.__init__(self)
        self.grid = Grid_Upwind(self.nx, self.ny, self.Lx, self.Ly)
        for i in range(self.nx):
            for j in range(self.ny):
                self.grid.velofield[i][j][0] = - self.grid.coordy[i][j] * self.dtheta
                self.grid.velofield[i][j][1] = self.grid.coordx[i][j] * self.dtheta
        self.reinit()
        maxa = np.max(self.grid.velofield)
        print("cfl = ", maxa * max([self.dt / self.grid.dx, self.dt / self.grid.dy]))
        print("dt = ", self.dt)
        print("Norme Linf |   simu   |  kalman")

    def reinit(self):
        """
            Reinit everything
        :return:
        """
        self.simulation = Simulation(self.grid, self.noise_sim)
        self.kalsim = Simulation(self.grid, self.noise_sim)

        self.dt = self.simulation.dt
        self.nIt = int(self.T_fin / self.dt) + 1
        self.reality = Reality(self.grid, self.dtheta, self.noise_real, self.dt)

        self.simulation.nIt = self.nIt
        self.kalsim.nIt = self.nIt
        self.reality.nIt = self.nIt

        self.kalman = KalmanWrapper(self.reality, self.kalsim)

    def compute(self, simu):
        """
            Generator : each call produce a time step the a simulation
        :param simu: the simulation to perform (filtered or not)
        :return: iteration number
        """
        for i in EDP.compute(self, simu):
            simu.err = self.norm(simu.getsol() - self.reality.getsol())
            yield i

    def norm(self, field):
        """
            Compute the H1 norm of the field
        :param field: field
        :return: norm H1
        """
        return self.grid.norm_h1(field)

    def plot(self, field):
        """
            Plot one field
        :param field: field
        """
        self.grid.plot(field)

    def animate(self, simu):
        """
            Perform the simulation and produce animation a the same time
        :param simu: the simulation to perform
        """

        self.grid.animate(simu, self.compute)

    def animatewithnoise(self, simu):
        """
            Perform the simulation and produce animation a the same time
        :param simu: the simulation to perform
        """

        self.grid.animatewithnoise(simu, self.compute, self.norm)

    def run_test_case(self, graphs):
        """
            Run the test case
        :return:
        """
        self.reinit()

        # ----------------- Compute reality and measurement --------------------
        if graphs:
            self.animate(self.reality)
        else:
            for it in self.compute(self.reality):
                pass
        Sol_ref = self.reality.getsol()
        if graphs:
            self.animatewithnoise(self.reality)
        Sol_mes = self.reality.getsolwithnoise()

        Norm_ref = self.norm(Sol_ref)
        Err_mes = self.norm(Sol_ref - Sol_mes) / Norm_ref

        # ------------------------ Compute simulation without Kalman ----------------------------
        self.reality.reinit()
        # Initial solution
        self.simulation.setsol(self.reality.getsol())
        if graphs:
            self.animate(self.simulation)
        else:
            for it in self.compute(self.simulation):
                pass
        Sol_sim = self.simulation.getsol()
        Err_sim = self.norm(Sol_ref - Sol_sim) / Norm_ref

        # ------------------------ Compute simulation with Kalman ----------------------------
        self.reality.reinit()
        # Initial solution
        self.kalman.setsol(self.reality.getsol())

        if graphs:
            self.animate(self.kalman)
        else:
            for it in self.compute(self.kalman):
                pass
        Sol_kal = self.kalman.getsol()
        Err_kal = self.norm(Sol_ref - Sol_kal) / Norm_ref

        # ------------------------ Final output ----------------------------

        print("%8.2e   | %8.2e | %8.2e" % (np.max(Sol_ref), np.max(Sol_sim), np.max(Sol_kal)))
        print("%8.2e | %8.2e | %8.2e | %8.2e" % (Norm_ref, Err_mes, Err_sim, Err_kal))


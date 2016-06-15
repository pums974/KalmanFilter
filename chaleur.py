#!/usr/bin/python2
# coding=utf-8
"""
    Chaleur will solve the problem
    - lap T = S
    and use the Kalman filter to mix
    2st order simulation and noisy measurements
    In order to produce converged field closer to the analytic solution
    The result is :
    * one plot with the analytic solution
    * one plot with the noisy measurements
    * one animation of the simulation during time convergence
    * one animation of the filtered simulation during time convergence
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from kalman import KalmanFilter
from skeleton import *
from grid import Grid_DF2


class Reality(SkelReality):
    """
        This class contains the analytical solution of our problem
        It also provide way to get a noisy field around this solution
    """
    # ---------------------------------METHODS-----------------------------------

    def __init__(self, _grid, _source_term, _noiselevel):
        SkelReality.__init__(self, _noiselevel)
        self.source_term = _source_term
        self.grid = _grid
        self.field = np.zeros([self.grid.nx, self.grid.ny])

    def compute(self):
        """
            Compute the analytical solution
        """
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                self.field[i][j] = 1. - self.source_term * (
                    (self.grid.coordx[i][j]) ** 2 + (self.grid.coordy[i][j]) ** 2) / 4.


class Simulation(SkelSimulation):
    """
    This class contains everything for the simulation
    """
    # --------------------------------PARAMETERS--------------------------------
    dt = 0.01
    cfl = 1. / 4.
    conv = 1.
    conv_crit = 1e-3
    err = 999

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _grid, _power,_noiselevel):
        SkelSimulation.__init__(self)
        self.noiselevel = _noiselevel

        def indx(_i, _j):
            """
                Swith from coordinate to line number in the matrix
            :param _i: x coordinate
            :param _j: y coordinate
            :return: line number
            """
            return _j + _i * ny

        self.power = _power
        self.grid = _grid
        nx = _grid.nx
        ny = _grid.ny
        self.size = nx * ny
        if self.cfl:
            self.dt = min([self.grid.dx ** 2, self.grid.dy ** 2]) * self.cfl
        print("cfl = ", max([self.dt / (self.grid.dx ** 2), self.dt / (self.grid.dy ** 2)]))
        print("dt = ", self.dt)

        # compute matrix
        # time
        self.Mat = np.eye(self.size, self.size)
        # Laplacien
        for i in range(nx):
            for j in range(ny):
                self.field = np.zeros([nx, ny])
                self.field[i, j] = self.dt
                self.Mat[indx(i, j)] += np.reshape(self.grid.dderx(self.field), [self.size])
                self.Mat[indx(i, j)] += np.reshape(self.grid.ddery(self.field), [self.size])
        # rhs and boundary conditions
        self.rhs = np.zeros([self.size]) + self.power * self.dt
        self.Mat = self.Mat.transpose()
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
        self.oldfield = np.zeros([self.size])

    @property
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

    def step(self):
        """
            Increment through the next time step of the simulation.
        """
        def indx(_i, _j):
            """
                Swith from coordinate to line number in the matrix
            :param _i: x coordinate
            :param _j: y coordinate
            :return: line number
            """
            return _j + _i * self.grid.ny

        power = np.random.normal(self.power, self.noiselevel, self.rhs.shape)
        self.rhs = np.zeros([self.size]) + power * self.dt
        for j in range(self.grid.ny):
            self.rhs[indx(0, j)] = 0.
            self.rhs[indx(self.grid.nx - 1, j)] = 0.
        for i in range(self.grid.nx):
            self.rhs[indx(i, 0)] = 0.
            self.rhs[indx(i, self.grid.ny - 1)] = 0.
        SkelSimulation.step(self)


class KalmanWrapper(SkelKalmanWrapper):
    """
        This class is use around the simulation to apply the kalman filter
    """
    conv = 1.
    conv_crit = 1e-2
    err = 999

    def __init__(self, _reality, _sim):

        def indx(_i, _j):
            """
                Swith from coordinate to line number in the matrix
            :param _i: x coordinate
            :param _j: y coordinate
            :return: line number
            """
            return _j + _i * self.kalsim.grid.ny

        SkelKalmanWrapper.__init__(self, _reality, _sim)
        self.size = self.kalsim.size
        _M = self.getwindow()  # Observation matrix.
        self.kalman = KalmanFilter(self.kalsim, _M)
        self.kalman.S = np.eye(self.kalman.size_s) * self.reality.noiselevel ** 2   # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * self.reality.noiselevel ** 2   # Estimated error in measurements.

        G = np.zeros([self.kalman.size_s, 1])
        for i in range(self.kalsim.grid.nx):
            for j in range(self.kalsim.grid.ny):
                # G[indx(i, j)] = self.kalsim.grid.dx ** 4 / 24. \
                #               + self.kalsim.grid.dy ** 4 / 24. \
                #               + self.kalsim.dt ** 2 / 2.
                G[indx(i, j)] = self.kalsim.dt * self.kalsim.noiselevel ** 2

        self.kalman.Q = G.dot(np.transpose(G))
        # self.kalman.Q = np.eye(self.kalman.size_s) * 0.  # Estimated error in process.

    def getwindow(self):
        """
            Produce the observation matrix : designate what we conserve of the noisy field
        :return: observation matrix
        """
        def indx(_i, _j):
            """
                Swith from coordinate to line number in the matrix
            :param _i: x coordinate
            :param _j: y coordinate
            :return: line number
            """
            return _j + _i * self.kalsim.grid.ny
        M = np.eye(self.kalsim.size)
        # ep = 1
        # size_o = 2 * ep * self.kalsim.grid.nx + 2 * ep * (self.kalsim.grid.ny - 2 * ep)
        # M = np.zeros([size_o, self.kalsim.size])
        # k = 0
        # for i in range(self.kalsim.grid.nx):
        #     for j in range(0, ep):
        #         M[k][indx(i, j)] = 1.
        #         k += 1
        #     for j in range(self.kalsim.grid.ny - ep, self.kalsim.grid.ny):
        #         M[k][indx(i, j)] = 1.
        #         k += 1
        # for j in range(ep, self.kalsim.grid.ny - ep):
        #     for i in range(0, ep):
        #         M[k][indx(i, j)] = 1.
        #         k += 1
        #     for i in range(self.kalsim.grid.nx - ep, self.kalsim.grid.nx):
        #         M[k][indx(i, j)] = 1.
        #         k += 1
        return M

    @property
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

    def step(self):
        """
            Compute the next step of the simulation and apply kalman filter to the result
        """
        self.kalsim.step()
        self.setsol(self.kalsim.getsol)
        self.setmes(self.reality.getsolwithnoise())
        self.kalman.apply()
        self.kalsim.setsol(self.getsol)


class Chaleur(EDP):
    """
        This class contains everything for this test case :
        * The analytical solution
        * A simulation
        * A filtered simulation
        * how to plot the results
    """
    Lx = 2.
    Ly = 3.
    nx = 10
    ny = 20
    source_term = 2.
    noisereal = .2
    noisesim = .5

    def __init__(self):
        EDP.__init__(self)
        self.grid = Grid_DF2(self.nx, self.ny, self.Lx, self.Ly)
        self.reality = Reality(self.grid, self.source_term, self.noisereal)
        self.simulation = Simulation(self.grid, self.source_term, self.noisesim)
        self.kalsim = Simulation(self.grid, self.source_term, self.noisesim)
        self.kalman = KalmanWrapper(self.reality, self.kalsim)

    def compute(self, simu):
        """
            Generator : each call produce a time step the a simulation
        :param simu: the simulation to perform (filtered or not)
        :return: iteration number
        """
        i = 0
        while simu.conv > simu.conv_crit:
            i += 1
            oldfield = simu.getsol
            simu.step()
            newfield = simu.getsol
            simu.conv = self.norm_l2(oldfield - newfield)
            simu.err = self.norm(newfield - self.reality.getsol)
            yield i

    def norm(self, field):
        """
            Compute the H1 norm of the field
        :param field: field
        :return: norm H1
        """
        return self.grid.norm_h1(field)

    def norm_l2(self, field):
        """
            Compute the L2 nor of the field
        :param field: field
        :return:  norm L2
        """
        return self.grid.norm_l2(field)

    def plot(self, field):
        """
            Plot one field
        :param field: field
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        surf = ax.plot_surface(self.grid.coordx, self.grid.coordy, field, rstride=1,
                               cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def animate(self, simu):
        """
            Perform the simulation and produce animation a the same time
        :param simu: the simulation to perform
        """
        def plot_update(i):
            """
                Update the plot
            :param i: iteration number
            :return: surface to be plotted
            """
            ax.clear()
            surf = ax.plot_surface(self.grid.coordx, self.grid.coordy, simu.getsol, rstride=1,
                                   cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_title('It = ' + str(i) + ',\n conv = ' + str(simu.conv) + ',\n err = ' + str(simu.err))
            return surf,

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        _ = animation.FuncAnimation(fig, plot_update, self.compute(simu), blit=False, interval=10,
                                    repeat=False)
        plt.show()

    def run_test_case(self, graphs):
        """
            Run the test case
        :return:
        """
        # ----------------- Compute reality and measurement --------------------
        self.reality.compute()

        Sol_ref = self.reality.getsol
        Sol_mes = self.reality.getsolwithnoise()

        Norm_ref = self.norm(Sol_ref)
        if graphs:
            self.plot(Sol_ref)

        Err_mes = self.norm(Sol_ref - Sol_mes) / Norm_ref
        print("Norme H1 de la mesure", Err_mes)
        if graphs:
            self.plot(Sol_mes)

        # ------------------------ Compute simulation without Kalman ----------------------------
        print("Simulation sans Kalman...")
        # Bad initial solution and boundary condition
        self.simulation.setsol(self.reality.getsolwithnoise())

        if graphs:
            self.animate(self.simulation)
        else:
            for it in self.compute(self.simulation):
                pass
        Sol_sim = self.simulation.getsol
        Err_sim = self.norm(Sol_ref - Sol_sim) / Norm_ref
        print("Norme H1 de la simu", Err_sim)

        # ------------------------ Compute simulation with Kalman ----------------------------
        print("Simulation avec Kalman...")
        # Bad initial solution
        self.kalman.setsol(self.reality.getsolwithnoise())

        if graphs:
            self.animate(self.kalman)
        else:
            for it in self.compute(self.kalman):
                pass
        Sol_kal = self.kalman.getsol
        Err_kal = self.norm(Sol_ref - Sol_kal) / Norm_ref
        print("Norme H1 de la simu filtre", Err_kal)

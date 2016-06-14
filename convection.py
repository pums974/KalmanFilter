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
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from kalman import KalmanFilter
from skeleton import *
from grid import Grid_Upwind
import math


class Reality(SkelReality):
    """
        This class contains the analytical solution of our problem
        It also provide way to get a noisy field around this solution
    """
    err = 999

    # ---------------------------------METHODS-----------------------------------

    def __init__(self, _grid, _dtheta, _noiselevel, _dt):
        SkelReality.__init__(self, _noiselevel)
        self.dtheta = _dtheta
        self.dt = _dt
        self.grid = _grid
        self.field = np.zeros([self.grid.nx, self.grid.ny])
        self.It = - 1

    def step(self):
        """
            Compute the analytical solution
        """
        self.field = np.zeros([self.grid.nx, self.grid.ny])
        self.It += 1
        t = self.It * self.dt
        x0 = self.grid.Lx * math.cos(t * self.dtheta * 0.5) * 0.25
        y0 = self.grid.Lx * math.sin(t * self.dtheta * 0.5) * 0.25
        tmp1 = np.square(self.grid.coordx - x0)
        tmp2 = np.square(self.grid.coordy - y0)
        self.field = np.maximum(1. - tmp1 - tmp2, self.field)

    def reinit(self):
        """
            reinitialize the iteration number
        """
        self.It = - 1
        self.step()


class Simulation(SkelSimulation):
    """
    This class contains everything for the simulation
    """
    # --------------------------------PARAMETERS--------------------------------
    cfl = 1. / 2.
    err = 999

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _grid):
        SkelSimulation.__init__(self)

        def indx(_i, _j):
            """
                Swith from coordinate to line number in the matrix
            :param _i: x coordinate
            :param _j: y coordinate
            :return: line number
            """
            return _j + _i * ny

        self.grid = _grid
        nx = _grid.nx
        ny = _grid.ny
        self.size = nx * ny

        maxa = np.max(_grid.velofield)
        self.dt = self.cfl * min([_grid.dx, _grid.dy]) / maxa
        print("cfl = ", maxa * max([self.dt / _grid.dx, self.dt / _grid.dy]))
        print("dt = ", self.dt)

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
        # rhs and boundary conditions
        self.rhs = np.zeros([self.size])
        self.Mat = self.Mat.transpose()
        for j in range(ny):
            self.Mat[indx(0, j)] = np.zeros([self.size])
            self.Mat[indx(nx - 1, j)] = np.zeros([self.size])
            # self.Mat[indx(0, j)][indx(0, j)] = 1.
            # self.Mat[indx(nx - 1, j)][indx(nx - 1, j)] = 1.
            self.rhs[indx(0, j)] = 0.
            self.rhs[indx(nx - 1, j)] = 0.
        for i in range(nx):
            self.Mat[indx(i, 0)] = np.zeros([self.size])
            self.Mat[indx(i, ny - 1)] = np.zeros([self.size])
            # self.Mat[indx(i, 0)][indx(i, 0)] = 1.
            # self.Mat[indx(i, ny - 1)][indx(i, ny - 1)] = 1.
            self.rhs[indx(i, 0)] = 0.
            self.rhs[indx(i, ny - 1)] = 0.
        self.field = np.zeros([self.size])

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


class KalmanWrapper(SkelKalmanWrapper):
    """
        This class is use around the simulation to apply the kalman filter
    """
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
        self.kalman.S = np.eye(self.kalman.size_s) * 0.2  # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * 0.2  # Estimated error in measurements.

        # G = np.zeros([self.kalman.size_s, 1])
        # for i in range(self.kalsim.grid.nx):
        #     for j in range(self.kalsim.grid.ny):
        #         G[indx(i, j)] = self.kalsim.grid.dx ** 2 / 2. \
        #                         + self.kalsim.grid.dy ** 2 / 2. \
        #                         + self.kalsim.dt ** 2 / 2.
        # self.kalman.Q = G.dot(np.transpose(G))
        self.kalman.Q = np.eye(self.kalman.size_s) * 0.  # Estimated error in process.

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
        # self.setsol(self.kalsim.getsol)
        # self.setmes(self.reality.getsolwithnoise())
        # self.kalman.apply()
        # self.kalsim.setsol(self.getsol)
        self.setsol(self.reality.getsol)


class Convection(EDP):
    """
        This class contains everything for this test case :
        * The analytical solution
        * A simulation
        * A filtered simulation
        * how to plot the results
    """
    Lx = 4.
    Ly = 4.
    nx = 20
    ny = 20
    dtheta = 2. * math.pi / 10.
    noiselevel = .2
    T_fin = 10.

    def __init__(self):
        EDP.__init__(self)
        self.grid = Grid_Upwind(self.nx, self.ny, self.Lx, self.Ly)
        for i in range(self.nx):
            for j in range(self.ny):
                self.grid.velofield[i][j][0] = - self.grid.coordy[i][j] * self.dtheta
                self.grid.velofield[i][j][1] = self.grid.coordx[i][j] * self.dtheta

        self.simulation = Simulation(self.grid)
        self.kalsim = Simulation(self.grid)

        self.dt = self.simulation.dt
        self.n_it = int(self.T_fin / self.dt)

        self.reality = Reality(self.grid, self.dtheta, self.noiselevel, self.dt)
        self.kalman = KalmanWrapper(self.reality, self.kalsim)

    def compute(self, simu):
        """
            Generator : each call produce a time step the a simulation
        :param simu: the simulation to perform (filtered or not)
        :return: iteration number
        """
        self.reality.reinit()
        for i in range(self.n_it):
            if i > 0:
                self.reality.step()
            simu.step()
            simu.err = self.norm(simu.getsol - self.reality.getsol)
            yield i

    def norm(self, field):
        """
            Compute the H1 norm of the field
        :param field: field
        :return: norm H1
        """
        return self.norm_l2(field)

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
            ax.set_title('It = ' + str(i) + ',\n err = ' + str(simu.err))
            return surf,

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        _ = animation.FuncAnimation(fig, plot_update, self.compute(simu), blit=False, interval=10,
                                    repeat=False)
        plt.show()

    def animatewithnoise(self, simu):
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
            surf = ax.plot_surface(self.grid.coordx, self.grid.coordy, simu.getsolwithnoise(), rstride=1,
                                   cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            simu.err = self.norm(simu.getsol - simu.getsolwithnoise())
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_title('It = ' + str(i) + ',\n err = ' + str(simu.err))
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
        if graphs :
            self.animate(self.reality)
        else:
            for it in self.compute(self.reality):
                pass
        Sol_ref = self.reality.getsol
        if graphs:
            self.animatewithnoise(self.reality)
        Sol_mes = self.reality.getsolwithnoise()

        Norm_ref = self.norm(Sol_ref)
        # edp.plot(Sol_ref)

        Err_mes = self.norm(Sol_ref - Sol_mes) / Norm_ref
        print("Erreur H1 de la mesure", Err_mes)
        # edp.plot(Sol_mes)

        # ------------------------ Compute simulation without Kalman ----------------------------
        print("Simulation sans Kalman...")
        self.reality.reinit()
        # Bad initial solution and boundary condition
        self.simulation.setsol(self.reality.getsolwithnoise())
        if graphs:
            self.animate(self.simulation)
        else:
            for it in self.compute(self.simulation):
                pass
        Sol_sim = self.simulation.getsol
        Err_sim = self.norm(Sol_ref - Sol_sim) / Norm_ref
        print("Erreur H1 de la simu", Err_sim)

        # ------------------------ Compute simulation with Kalman ----------------------------
        print("Simulation avec Kalman...")
        self.reality.reinit()
        # Bad initial solution
        self.kalman.setsol(self.reality.getsolwithnoise())

        if graphs:
            self.animate(self.kalman)
        else:
            for it in self.compute(self.kalman):
                pass
        Sol_kal = self.kalman.getsol
        Err_kal = self.norm(Sol_ref - Sol_kal) / Norm_ref
        print("Erreur H1 de la simu filtre", Err_kal)

        print("Max de la solution de référence", np.max(Sol_ref))
        print("Max de la solution simulé", np.max(Sol_sim))
        print("Max de la solution simulé avec filtre", np.max(Sol_kal))


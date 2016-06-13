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
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from kalman import KalmanFilter
from skeleton import *

class Grid(object):
    """
        2D grid with finite difference derivative of second order
    """

    def __init__(self, _nx, _ny, _lx, _ly):
        self.Lx = _lx
        self.Ly = _ly
        self.nx = _nx
        self.ny = _ny
        coordx = np.linspace(-self.Lx / 2., self.Lx / 2., _nx)
        coordy = np.linspace(-self.Ly / 2., self.Ly / 2., _ny)
        self.dx = self.Lx / (self.nx - 1.)
        self.dy = self.Ly / (self.ny - 1.)
        self.coordy, self.coordx = np.meshgrid(coordy, coordx)
        self.mat_derx = np.zeros([self.nx, self.ny, 3])
        self.mat_dderx = np.zeros([self.nx, self.ny, 3])
        self.mat_dery = np.zeros([self.nx, self.ny, 3])
        self.mat_ddery = np.zeros([self.nx, self.ny, 3])
        for i in range(self.nx):
            for j in range(self.ny):
                self.mat_derx[i][j] = self.calc_derx(i, j, [0, 1, 0])
                self.mat_dderx[i][j] = self.calc_derx(i, j, [0, 0, 1])
                self.mat_dery[i][j] = self.calc_dery(i, j, [0, 1, 0])
                self.mat_ddery[i][j] = self.calc_dery(i, j, [0, 0, 1])

    def calc_derx(self, i, j, der):
        """
            Compute the matrix of the derivative in the x direction per point
        :param i: first coordinate in the mesh of the point of interest
        :param j: second coordinate in the mesh of the point of interest
        :param der: vector relative to the derivative wanted
        :return: matrix M such that M*field = dx field
        """
        mini_mat = np.zeros([3, 3])
        start = max([0, i - 1])
        end = min([start + 3, self.nx])
        start = end - 3
        for k in range(start, end):
            mini_mat[0][k - start] = 1
            mini_mat[1][k - start] = self.coordx[k][j] - self.coordx[i][j]
            mini_mat[2][k - start] = 0.5 * ((self.coordx[k][j] - self.coordx[i][j]) ** 2.)
        mini_mat = np.linalg.inv(mini_mat)
        return mini_mat.dot(der)

    def calc_dery(self, i, j, der):
        """
            Compute the matrix of the derivative in the y direction per point
        :param i: first coordinate in the mesh of the point of interest
        :param j: second coordinate in the mesh of the point of interest
        :param der: vector relative to the derivative wanted
        :return: matrix M such that M*field = dy field
        """
        mini_mat = np.zeros([3, 3])
        start = max([0, i - 1])
        end = min([start + 3, self.ny])
        start = end - 3
        for k in range(start, end):
            mini_mat[0][k - start] = 1
            mini_mat[1][k - start] = self.coordy[i][k] - self.coordy[i][j]
            mini_mat[2][k - start] = 0.5 * ((self.coordy[i][k] - self.coordy[i][j]) ** 2.)
        mini_mat = np.linalg.inv(mini_mat)
        return mini_mat.dot(der)

    def derx(self, field):
        """
            Compute the first x-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        derx = np.zeros([self.nx, self.ny])
        for i in range(self.nx):
            for j in range(self.ny):
                start = max([0, i - 1])
                end = min([start + 3, self.nx])
                start = end - 3
                for k in range(start, end):
                    derx[i][j] += field[k][j] * self.mat_derx[i][j][k - start]
        return derx

    def dderx(self, field):
        """
            Compute the second x-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        dderx = np.zeros([self.nx, self.ny])
        for i in range(self.nx):
            for j in range(self.ny):
                start = max([0, i - 1])
                end = min([start + 3, self.nx])
                start = end - 3
                for k in range(start, end):
                    dderx[i][j] += field[k][j] * self.mat_dderx[i][j][k - start]

        return dderx

    def dery(self, field):
        """
            Compute the first y-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        dery = np.zeros([self.nx, self.ny])
        for i in range(self.nx):
            for j in range(self.ny):
                start = max([0, j - 1])
                end = min([start + 3, self.ny])
                start = end - 3
                for k in range(start, end):
                    dery[i][j] += field[i][k] * self.mat_dery[i][j][k - start]
        return dery

    def ddery(self, field):
        """
            Compute the second y-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        ddery = np.zeros([self.nx, self.ny])
        for i in range(self.nx):
            for j in range(self.ny):
                start = max([0, j - 1])
                end = min([start + 3, self.ny])
                start = end - 3
                for k in range(start, end):
                    ddery[i][j] += field[i][k] * self.mat_ddery[i][j][k - start]

        return ddery

    def norm_h1(self, field):
        """
            Compute the H1 norm of a field :
            H1(field)^2 = L2(field)^2 + L2(dx field)^2 + L2(dx field)^2
        :param field: field
        :return: norm
        """
        dx = self.derx(field)
        dy = self.dery(field)
        return np.sqrt(np.sum(np.square(field)) + np.sum(np.square(dx)) + np.sum(np.square(dy)))

    @staticmethod
    def norm_l2(field):
        """
            Compute the L2 norm of a field :
            L1(field)^2 = sum_ij ( field_ij ^2 )
        :param field: field
        :return: norm
        """
        return np.sqrt(np.sum(np.square(field)))


class Reality(SkelReality):
    """
        This class contains the analytical solution of our problem
        It also provide way to get a noisy field around this solution
    """
    # ---------------------------------METHODS-----------------------------------

    def __init__(self, _grid, _source_term, _noiselevel):
        self.source_term = _source_term
        self.noiselevel = _noiselevel
        self.grid = _grid
        self.field = np.zeros([self.grid.nx, self.grid.ny])

    @property
    def getsolwithnoise(self):
        """
            Get a noisy field around the analytical solution
        :return: a noisy field
        """
        field_with_noise = np.zeros([self.grid.nx, self.grid.ny])
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                field_with_noise[i][j] = random.gauss(self.field[i][j], self.noiselevel)
        return field_with_noise

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
    def __init__(self, _grid, _source_term):
        def indx(_i, _j):
            """
                Swith from coordinate to line number in the matrix
            :param _i: x coordinate
            :param _j: y coordinate
            :return: line number
            """
            return _j + _i * ny

        self.source_term = _source_term
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
        self.rhs = np.zeros([self.size]) + self.source_term * self.dt
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
        self.field = self.Mat.dot(self.field) + self.rhs


class KalmanWrapper(SkelKalmanWrapper):
    """
        This class is use around the simulation to apply the kalman filter
    """
    conv = 1.
    conv_crit = 1e-2
    err = 999

    def __init__(self, _reality, _sim):
        self.reality = _reality
        self.kalsim = _sim
        self.size = self.kalsim.size
        _M = self.getwindow()  # Observation matrix.
        self.kalman = KalmanFilter(self.kalsim, _M)
        self.kalman.S = np.eye(self.kalman.size_s) * 0.2  # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * 0.2  # Estimated error in measurements.
        self.kalman.Q = np.eye(self.kalman.size_s) * 0.  # Estimated error in process.

    def getwindow(self):
        """
            Produce the observation matrix : designate what we conserve of the noisy field
        :return: observation matrix
        """
        M = np.eye(self.kalsim.size)
        # size_o = (self.kalsim.grid.nx - 4) * (self.kalsim.grid.ny - 4)
        # M = np.zeros([size_o, self.kalsim.size])
        # k = 0
        # for i in range(2, self.kalsim.grid.nx - 2):
        #     for j in range(2, self.kalsim.grid.ny - 2):
        #         M[k][j + i * self.kalsim.grid.ny] = 1.
        #         k += 1
        return M

    def setmes(self, field):
        """
            Reshape noisy field and gives it to the kalman filter
        :param field: noisy field
        """
        self.kalman.Y = self.kalman.M.dot(np.reshape(field, self.kalman.size_s))

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
        self.setmes(self.reality.getsolwithnoise)
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
    noiselevel = .2

    def __init__(self):
        self.grid = Grid(self.nx, self.ny, self.Lx, self.Ly)
        self.reality = Reality(self.grid, self.source_term, self.noiselevel)
        self.simulation = Simulation(self.grid, self.source_term)
        self.kalsim = Simulation(self.grid, self.source_term)
        self.kalman = KalmanWrapper(self.reality, self.kalsim)

    def compute(self,simu):
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
            simu.conv = self.norml2(oldfield - newfield)
            simu.err = self.normh1(simu.getsol - self.reality.getsol)
            yield i

    def normh1(self, field):
        """
            Compute the H1 norm of the field
        :param field: field
        :return: norm H1
        """
        return self.grid.norm_h1(field)

    def norml2(self, field):
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


# ------------------------ Begin program ----------------------------

edp = Chaleur()

# ----------------- Compute reality and measurement --------------------
edp.reality.compute()

Sol_ref = edp.reality.getsol
Sol_mes = edp.reality.getsolwithnoise

Norm_ref = edp.normh1(Sol_ref)
edp.plot(Sol_ref)

Err_mes = edp.normh1(Sol_ref - Sol_mes) / Norm_ref
print("Norme H1 de la mesure", Err_mes)
edp.plot(Sol_mes)

# ------------------------ Compute simulation without Kalman ----------------------------
print("Simulation sans Kalman...")
# Bad initial solution and boundary condition
edp.simulation.setsol(edp.reality.getsolwithnoise)

if False:
    for it in edp.compute(edp.simulation):
        pass
    Sol_sim = edp.simulation.getsol
    edp.plot(Sol_sim)
else:
    edp.animate(edp.simulation)
    Sol_sim = edp.simulation.getsol
Err_sim = edp.normh1(Sol_ref - Sol_sim) / Norm_ref
print("Norme H1 de la simu", Err_sim)

# ------------------------ Compute simulation with Kalman ----------------------------
print("Simulation avec Kalman...")
# Bad initial solution
edp.kalman.setsol(edp.reality.getsolwithnoise)

if False:
    for it in edp.compute(edp.kalman):
        pass
    Sol_kal = edp.kalman.getsol
    edp.plot(Sol_kal)
else:
    edp.animate(edp.kalman)
    Sol_kal = edp.kalman.getsol
Err_kal = edp.normh1(Sol_ref - Sol_kal) / Norm_ref
print("Norme H1 de la simu filtre", Err_kal)

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
from __future__ import print_function, absolute_import
import numpy as np

from kalman.kalman import KalmanFilter
from kalman.skeleton import *
from kalman.grid import Grid_DF2
import math
import sys


class Reality(SkelReality):
    """
        This class contains the analytical solution of our problem
        It also provide way to get a noisy field around this solution
    """
    # ---------------------------------METHODS-----------------------------------

    def __init__(self, _grid, _power, _noiselevel, _dt):
        self.grid = _grid
        SkelReality.__init__(self, _noiselevel, _dt, _grid.shape)
        self.power = _power
        c1 = 16 * _power/(math.pi ** 2)
        c2 = math.pi / self.grid.Lx
        c3 = 0.5 * math.pi / self.grid.Ly
        c4 = math.pi * math.pi / (4. * self.grid.Ly ** 2)
        c5 = math.pi * math.pi / (self.grid.Lx ** 2)
        add = np.zeros([self.grid.nx, self.grid.ny])
        self.nmode = 1
        self.mmode = 1
        self.initfield = np.zeros([self.grid.nx, self.grid.ny])
        for m in range(1, self.nmode+1):
            for n in range(1, self.mmode+1):
                c10 = c1 / ((2. * n - 1.)*(2. * m - 1.))
                c20 = c2 * (2. * n - 1.)
                c30 = c3 * (2. * m - 1.)
                for i in range(self.grid.nx):
                    for j in range(self.grid.ny):
                        x = self.grid.coordx[i][j]
                        y = self.grid.coordy[i][j]
                        add[i][j] = c10 * math.sin(c20 * x) \
                                        * math.sin(c30 * y)
                self.initfield += add

    def step(self):
        """
            Compute the analytical solution
        """
        self.field = np.zeros([self.grid.nx, self.grid.ny])
        self.It += 1
        time = self.It * self.dt
        c4 = math.pi * math.pi / (4. * self.grid.Ly ** 2)
        c5 = math.pi * math.pi / (self.grid.Lx ** 2)
        add = np.zeros([self.grid.nx, self.grid.ny])
        for m in range(1, self.nmode+1):
            for n in range(1, self.mmode+1):
                c40 = c4 * (2. * n - 1.) ** 2
                c50 = c5 * (2. * m - 1.) ** 2
                for i in range(self.grid.nx):
                    for j in range(self.grid.ny):
                        add[i][j] = self.initfield[i][j] * math.exp(-(c40 + c50) * time)
                self.field += add
                # print(m, n, self.grid.norm_h1(add))
                # if self.grid.norm_h1(add) < 0.001:
                #     break

                # self.field[i][j] = 1.


class Simulation(SkelSimulation):
    """
    This class contains everything for the simulation
    """
    # --------------------------------PARAMETERS--------------------------------
    cfl = 1. / 4.

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _grid, _power, _noiselevel):
        self.grid = _grid
        self.dt = min([self.grid.dx ** 2, self.grid.dy ** 2]) * self.cfl

        SkelSimulation.__init__(self, _noiselevel, self.dt, self.grid.shape)
        self.power = _power

        # compute matrix
        self.calcmat()

        # rhs and boundary conditions
        self.calcmatbc()

        self.field = np.zeros([self.size])
        self.calcbc()

    def calcmat(self):
        """
            compute the matrix for diffusion equation
        :return:
        """
        indx = self.grid.indx
        nx = self.grid.nx
        ny = self.grid.ny
        # time
        self.Mat = np.eye(self.size, self.size)
        # Laplacien
        for i in range(nx):
            for j in range(ny):
                self.field = np.zeros([nx, ny])
                self.field[i, j] = self.dt
                self.Mat[indx(i, j)] += np.reshape(self.grid.dderx(self.field), [self.size])
                self.Mat[indx(i, j)] += np.reshape(self.grid.ddery(self.field), [self.size])
        self.Mat = self.Mat.transpose()

    def calcmatbc(self):
        """
            Insert boundary condition inside the matrix of the diffusion problem
        :return:
        """
        indx = self.grid.indx
        nx = self.grid.nx
        ny = self.grid.ny

        def zeroing(_i, _j):
            """
                Set the line corresponding to the point i,j to 0 except on the diagonal
            :param _i:
            :param _j:
            :return:
            """
            self.Mat[indx(_i, _j)] = np.zeros([self.size])
            self.Mat[indx(_i, _j)][indx(_i, _j)] = 1.

        for i in range(nx):
            zeroing(i, 0)
            zeroing(i, ny - 1)
        for j in range(ny):
            zeroing(0, j)
            zeroing(nx - 1, j)

    def calcbc(self):
        """
            Impose boundary condition on the field
        :return:
        """
        indx = self.grid.indx
        nx = self.grid.nx
        ny = self.grid.ny

        def dirichlet(_i, _j, val):
            """
                Impose a Dirichlet boundary condition
            :param _i:
            :param _j:
            :param val:
            :return:
            """
            self.field[indx(_i, _j)] = val

        def neumann(_i, _j, val, _dir):
            """
                Impose a neumann boundary condition
            :param _i:
            :param _j:
            :param val:
            :param _dir:
            :return:
            """
            if _dir[0] != 0:
                der = self.grid.derx
            else:
                der = self.grid.dery

            self.field[indx(_i, _j)] = - val
            newval = 0.
            c = 0.
            for k in range(2, -1, -1):
                dir1 = np.array([_i, _j]) + k * _dir
                tmp = np.zeros([nx, ny])
                tmp[dir1[0], dir1[1]] = 1.
                tmp = der(tmp)
                c = tmp[_i, _j]
                newval -= c * self.field[indx(dir1[0], dir1[1])]

            self.field[indx(_i, _j)] = newval / c

        for i in range(nx):
            dirichlet(i, 0, 0.)
            # dirichlet(i, ny - 1, 1.)
            neumann(i, ny - 1, 0., np.array([0, -1]))
        for j in range(ny):
            dirichlet(0, j, 0.)
            dirichlet(nx - 1, j, 0.)
            # neumann(nx - 1, j, 1., 0)

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
        power = np.random.normal(self.power, self.noiselevel, self.rhs.shape)
        self.rhs = np.zeros([self.size]) + power * self.dt
        SkelSimulation.step(self)
        self.calcbc()


class KalmanWrapper(SkelKalmanWrapper):
    """
        This class is use around the simulation to apply the kalman filter
    """
    def __init__(self, _reality, _sim):
        SkelKalmanWrapper.__init__(self, _reality, _sim)
        self.size = self.kalsim.size
        _M = self.getwindow()  # Observation matrix.
        self.kalman = KalmanFilter(self.kalsim, _M)
        self.kalman.S = np.eye(self.kalman.size_s) * self.reality.noiselevel ** 2   # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * self.reality.noiselevel ** 2   # Estimated error in measurements.

        indx = self.kalsim.grid.indx
        G = np.zeros([self.kalman.size_s, 1])
        for i in range(self.kalsim.grid.nx):
            for j in range(self.kalsim.grid.ny):
                G[indx(i, j)] = self.kalsim.grid.dx ** 4 / 24. \
                              + self.kalsim.grid.dy ** 4 / 24. \
                              + self.kalsim.dt ** 2 / 2.
                # G[indx(i, j)] = self.kalsim.dt

        self.kalman.Q = G.dot(np.transpose(G)) * self.kalsim.noiselevel ** 2  # Estimated error in process.
        # self.kalman.Q = np.eye(self.kalman.size_s) * self.kalsim.noiselevel ** 2  # Estimated error in process.

    def getwindow(self):
        """
            Produce the observation matrix : designate what we conserve of the noisy field
        :return: observation matrix
        """
        indx = self.kalsim.grid.indx
        # M = np.eye(self.kalsim.size)
        ep = 1
        size_o = 2 * ep * self.kalsim.grid.nx + 2 * ep * (self.kalsim.grid.ny - 2 * ep)
        M = np.zeros([size_o, self.kalsim.size])
        k = 0
        for i in range(self.kalsim.grid.nx):
            for j in range(0, ep):
                M[k][indx(i, j)] = 1.
                k += 1
            for j in range(self.kalsim.grid.ny - ep, self.kalsim.grid.ny):
                M[k][indx(i, j)] = 1.
                k += 1
        for j in range(ep, self.kalsim.grid.ny - ep):
            for i in range(0, ep):
                M[k][indx(i, j)] = 1.
                k += 1
            for i in range(self.kalsim.grid.nx - ep, self.kalsim.grid.nx):
                M[k][indx(i, j)] = 1.
                k += 1
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
    nx = 20
    ny = 20
    power = 1.
    noise_real = 0.1
    noise_sim = 0.1
    dt = 0.
    nIt = 150
    name = "Chaleur"

    def __init__(self):
        EDP.__init__(self)
        self.grid = Grid_DF2(self.nx, self.ny, self.Lx, self.Ly)
        self.grid.coordx += self.grid.Lx / 2.
        self.grid.coordy += self.grid.Ly / 2.
        self.reinit()
        print("cfl = ", max([self.dt / (self.grid.dx ** 2), self.dt / (self.grid.dy ** 2)]))
        print("dt = ", self.dt)
        print("Norme H1 |  reality |   simu   |  kalman")

    def reinit(self):
        """
            Reinit everything
        :return:
        """
        self.simulation = Simulation(self.grid, 0., self.noise_sim)
        self.kalsim = Simulation(self.grid, 0., self.noise_sim)

        self.dt = self.simulation.dt
        self.reality = Reality(self.grid, self.power, self.noise_real, self.dt)

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
        self.simulation.setsol(self.reality.getsolwithnoise())
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
        self.kalman.setsol(self.reality.getsolwithnoise())

        if graphs:
            self.animate(self.kalman)
        else:
            for it in self.compute(self.kalman):
                pass
        Sol_kal = self.kalman.getsol()
        Err_kal = self.norm(Sol_ref - Sol_kal) / Norm_ref

        # ------------------------ Final output ----------------------------

        print("%8.2e | %8.2e | %8.2e | %8.2e" % (Norm_ref, Err_mes, Err_sim, Err_kal))
        Norm_ref = self.grid.norm_inf(Sol_ref)
        Err_mes = self.grid.norm_inf(Sol_ref - Sol_mes)
        Err_sim = self.grid.norm_inf(Sol_ref - Sol_sim)
        Err_kal = self.grid.norm_inf(Sol_ref - Sol_kal)
        print("%8.2e | %8.2e | %8.2e | %8.2e" % (Norm_ref, Err_mes, Err_sim, Err_kal))

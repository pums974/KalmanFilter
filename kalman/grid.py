#!/usr/bin/python2
# coding=utf-8
"""
    2D grid with finite difference derivative of second order
"""
from __future__ import print_function, absolute_import
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import animation
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
import sys
try:
    if sys.version_info < (3, ):
        from libs.fortran_libs_py2 import derx_df2_f, dery_df2_f, derx_upwind_f, dery_upwind_f
    else:
        from libs.fortran_libs_py3 import derx_df2_f, dery_df2_f, derx_upwind_f, dery_upwind_f
except:
    if sys.version_info < (3, ):
        from kalman.libs.fortran_libs_py2 import derx_df2_f, dery_df2_f, derx_upwind_f, dery_upwind_f
    else:
        from kalman.libs.fortran_libs_py3 import derx_df2_f, dery_df2_f, derx_upwind_f, dery_upwind_f

use_fortran = True


class Grid(object):
    """
        Generic 2D grid
    """

    def __init__(self, _nx, _ny, _lx, _ly):
        self.Lx = _lx
        self.Ly = _ly
        self.nx = _nx
        self.ny = _ny
        self.size = _nx * _ny
        self.shape = [_nx, _ny]
        coordx = np.linspace(-self.Lx / 2., self.Lx / 2., _nx)
        coordy = np.linspace(-self.Ly / 2., self.Ly / 2., _ny)
        self.dx = self.Lx / (self.nx - 1.)
        self.dy = self.Ly / (self.ny - 1.)
        self.coordy, self.coordx = np.meshgrid(coordy, coordx)

    def indx(self, _i, _j):
        """
            Swith from coordinate to line number in the matrix
        :param _i: x coordinate
        :param _j: y coordinate
        :return: line number
        """
        return _j + _i * self.ny

    def derx(self, field):
        """
            Compute the first x-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        derx = np.zeros([self.nx, self.ny])
        return derx

    def dery(self, field):
        """
            Compute the first y-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        dery = np.zeros([self.nx, self.ny])
        return dery

    def dderx(self, field):
        """
            Compute the first x-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        dderx = np.zeros([self.nx, self.ny])
        return dderx

    def ddery(self, field):
        """
            Compute the first y-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        ddery = np.zeros([self.nx, self.ny])
        return ddery

    def norm_l2(self, field):
        """
            Compute the L2 norm of a field :
            L1(field)^2 = sum_ij ( field_ij ^2 )
        :param field: field
        :return: norm
        """
        return np.sqrt(np.sum(np.square(field)))

    def norm_h1(self, field):
        """
            Compute the H1 norm of a field :
            H1(field)^2 = L2(field)^2 + L2(dx field)^2 + L2(dx field)^2
        :param field: field
        :return: norm
        """
        dx = self.derx(field)
        dy = self.dery(field)
        return np.sqrt(np.sum(np.square(field)) + np.sum(np.square(dx)) +
                       np.sum(np.square(dy)))

    def norm_inf(self, field):
        """
            Compute the H1 norm of a field :
            H1(field)^2 = L2(field)^2 + L2(dx field)^2 + L2(dx field)^2
        :param field: field
        :return: norm
        """
        return np.max(np.abs(field))

    def plot(self, field):
        """
            Plot one field
        :param field: field
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(self.coordx[1, 1], self.coordx[self.nx - 1, self.ny - 1])
        ax.set_ylim(self.coordy[1, 1], self.coordy[self.nx - 1, self.ny - 1])
        ax.set_zlim(0, 1)
        surf = ax.plot_surface(self.coordx,
                               self.coordy,
                               field,
                               rstride=1,
                               cstride=1,
                               cmap=cm.coolwarm,
                               linewidth=0,
                               antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def animate(self, simu, compute):
        """
            Perform the simulation and produce animation a the same time
        :param simu: the simulation to perform
        :param compute: the generator to compute time steps
        """

        def plot_update(i):
            """
                Update the plot
            :param i: iteration number
            :return: surface to be plotted
            """
            ax.clear()
            surf = ax.plot_surface(self.coordx,
                                   self.coordy,
                                   simu.getsol(),
                                   rstride=1,
                                   cstride=1,
                                   cmap=cm.coolwarm,
                                   linewidth=0,
                                   antialiased=False)
            ax.set_xlim(self.coordx[1, 1],
                        self.coordx[self.nx - 1, self.ny - 1])
            ax.set_ylim(self.coordy[1, 1],
                        self.coordy[self.nx - 1, self.ny - 1])
            ax.set_zlim(0, 1)
            ax.set_title('It = ' + str(i) + ',\n err = ' + str(simu.err))
            return surf,

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(self.coordx[1, 1], self.coordx[self.nx - 1, self.ny - 1])
        ax.set_ylim(self.coordy[1, 1], self.coordy[self.nx - 1, self.ny - 1])
        ax.set_zlim(0, 1)

        _ = animation.FuncAnimation(fig,
                                    plot_update,
                                    compute(simu),
                                    blit=False,
                                    interval=10,
                                    repeat=False)
        plt.show()

    def animatewithnoise(self, simu, compute, norm):
        """
            Perform the simulation and produce animation a the same time
        :param simu: the simulation to perform
        :param compute: the generator to compute time steps
        :param norm: the norm to compute noise norm
        """

        def plot_update(i):
            """
                Update the plot
            :param i: iteration number
            :return: surface to be plotted
            """
            ax.clear()
            surf = ax.plot_surface(self.coordx,
                                   self.coordy,
                                   simu.getsolwithnoise(),
                                   rstride=1,
                                   cstride=1,
                                   cmap=cm.coolwarm,
                                   linewidth=0,
                                   antialiased=False)
            simu.err = norm(simu.getsol() - simu.getsolwithnoise())
            ax.set_xlim(self.coordx[1, 1],
                        self.coordx[self.nx - 1, self.ny - 1])
            ax.set_ylim(self.coordy[1, 1],
                        self.coordy[self.nx - 1, self.ny - 1])
            ax.set_zlim(0, 1)
            ax.set_title('It = ' + str(i) + ',\n err = ' + str(simu.err))
            return surf,

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(self.coordx[1, 1], self.coordx[self.nx - 1, self.ny - 1])
        ax.set_ylim(self.coordy[1, 1], self.coordy[self.nx - 1, self.ny - 1])
        ax.set_zlim(0, 1)

        _ = animation.FuncAnimation(fig,
                                    plot_update,
                                    compute(simu),
                                    blit=False,
                                    interval=10,
                                    repeat=False)
        plt.show()


class Grid_DF2(Grid):
    """
        2D grid with finite difference derivative of second order
    """

    def __init__(self, _nx, _ny, _lx, _ly):
        Grid.__init__(self, _nx, _ny, _lx, _ly)
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
            mini_mat[2][k - start] = 0.5 * \
                ((self.coordx[k][j] - self.coordx[i][j]) ** 2.)
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
        start = max([0, j - 1])
        end = min([start + 3, self.ny])
        start = end - 3
        for k in range(start, end):
            mini_mat[0][k - start] = 1
            mini_mat[1][k - start] = self.coordy[i][k] - self.coordy[i][j]
            mini_mat[2][k - start] = 0.5 * \
                ((self.coordy[i][k] - self.coordy[i][j]) ** 2.)
        mini_mat = np.linalg.inv(mini_mat)
        return mini_mat.dot(der)

    def derx(self, field):
        """
            Compute the first x-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        if use_fortran:
            derx = derx_df2_f(field, self.mat_derx)
        else:
            derx = np.zeros([self.nx, self.ny])
            for i in range(self.nx):
                for j in range(self.ny):
                    start = max([0, i - 1])
                    end = min([start + 3, self.nx])
                    start = end - 3
                    for k in range(start, end):
                        derx[i][j] += field[k][j] * \
                            self.mat_derx[i][j][k - start]
        return derx

    def dderx(self, field):
        """
            Compute the second x-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        if use_fortran:
            dderx = derx_df2_f(field, self.mat_dderx)
        else:
            dderx = np.zeros([self.nx, self.ny])
            for i in range(self.nx):
                for j in range(self.ny):
                    start = max([0, i - 1])
                    end = min([start + 3, self.nx])
                    start = end - 3
                    for k in range(start, end):
                        dderx[i][j] += field[k][j] * \
                            self.mat_dderx[i][j][k - start]
        return dderx

    def dery(self, field):
        """
            Compute the first y-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        if use_fortran:
            dery = dery_df2_f(field, self.mat_dery)
        else:
            dery = np.zeros([self.nx, self.ny])
            for i in range(self.nx):
                for j in range(self.ny):
                    start = max([0, j - 1])
                    end = min([start + 3, self.ny])
                    start = end - 3
                    for k in range(start, end):
                        dery[i][j] += field[i][k] * \
                            self.mat_dery[i][j][k - start]
        return dery

    def ddery(self, field):
        """
            Compute the second y-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        if use_fortran:
            ddery = dery_df2_f(field, self.mat_ddery)
        else:
            ddery = np.zeros([self.nx, self.ny])
            for i in range(self.nx):
                for j in range(self.ny):
                    start = max([0, j - 1])
                    end = min([start + 3, self.ny])
                    start = end - 3
                    for k in range(start, end):
                        ddery[i][j] += field[i][k] * \
                            self.mat_ddery[i][j][k - start]
        return ddery


class Grid_Upwind(Grid):
    """
        2D grid with finite difference derivative of second order
    """

    def __init__(self, _nx, _ny, _lx, _ly):
        Grid.__init__(self, _nx, _ny, _lx, _ly)
        self.velofield = np.zeros([self.nx, self.ny, 2])

    def derx(self, field):
        """
            Compute the first x-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        if use_fortran:
            derx = derx_upwind_f(field, self.velofield, self.dx)
        else:
            rdx = 1. / self.dx
            derx = np.zeros([self.nx, self.ny])
            for i in range(self.nx):
                for j in range(self.ny):
                    ap = max([self.velofield[i][j][0], 0.]) * 0.5
                    am = min([self.velofield[i][j][0], 0.]) * 0.5
                    if i > 0:
                        derx[i][j] += ap * \
                            (field[i][j] - field[i - 1][j]) * rdx
                    if i < self.nx - 1:
                        derx[i][j] += am * \
                            (field[i + 1][j] - field[i][j]) * rdx
        return derx

    def dery(self, field):
        """
            Compute the first y-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        if use_fortran:
            dery = dery_upwind_f(field, self.velofield, self.dy)
        else:
            rdy = 1. / self.dy
            dery = np.zeros([self.nx, self.ny])
            for i in range(self.nx):
                for j in range(self.ny):
                    ap = max([self.velofield[i][j][1], 0.]) * 0.5
                    am = min([self.velofield[i][j][1], 0.]) * 0.5
                    if j > 0:
                        dery[i][j] += ap * \
                            (field[i][j] - field[i][j - 1]) * rdy
                    if j < self.ny - 1:
                        dery[i][j] += am * \
                            (field[i][j + 1] - field[i][j]) * rdy
        return dery

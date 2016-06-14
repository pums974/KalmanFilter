#!/usr/bin/python2
# coding=utf-8
"""
    2D grid with finite difference derivative of second order
"""
import numpy as np


class Grid_DF2(object):
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


class Grid_Upwind(object):
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
        self.velofield = np.zeros([self.nx, self.ny, 2])

    def derx(self, field):
        """
            Compute the first x-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        derx = np.zeros([self.nx, self.ny])
        for i in range(self.nx):
            for j in range(self.ny):
                ap = max([self.velofield[i][j][0], 0.])
                am = min([self.velofield[i][j][0], 0.])
                if i > 0:
                    derx[i][j] += ap * (field[i][j] - field[i - 1][j]) / self.dx
                if i < self.nx - 1:
                    derx[i][j] += am * (field[i + 1][j] - field[i][j]) / self.dx
        return derx

    def dery(self, field):
        """
            Compute the first y-derivative of a field
        :param field: field to be derived
        :return: field derived
        """
        dery = np.zeros([self.nx, self.ny])
        for i in range(self.nx):
            for j in range(self.ny):
                ap = max([self.velofield[i][j][1], 0.])
                am = min([self.velofield[i][j][1], 0.])
                if j > 0:
                    dery[i][j] += ap * (field[i][j] - field[i][j - 1]) / self.dy
                if j < self.ny - 1:
                    dery[i][j] += am * (field[i][j + 1] - field[i][j]) / self.dy
        return dery

    @staticmethod
    def norm_l2(field):
        """
            Compute the L2 norm of a field :
            L1(field)^2 = sum_ij ( field_ij ^2 )
        :param field: field
        :return: norm
        """
        return np.sqrt(np.sum(np.square(field)))

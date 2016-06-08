#!/usr/bin/python2

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation


class Grid:
    def __init__(self, _nx, _ny, _Lx, _Ly):
        self.Lx = _Lx
        self.Ly = _Ly
        self.nx = _nx
        self.ny = _ny
        coordx = np.linspace(-self.Lx/2., self.Lx/2., _nx)
        coordy = np.linspace(-self.Ly/2., self.Ly/2., _ny)
        self.dx = self.Lx/(self.nx-1.)
        self.dy = self.Ly/(self.ny-1.)
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
        ddery = np.zeros([self.nx, self.ny])
        for i in range(self.nx):
            for j in range(self.ny):
                start = max([0, j - 1])
                end = min([start + 3, self.ny])
                start = end - 3
                for k in range(start, end):
                    ddery[i][j] += field[i][k] * self.mat_ddery[i][j][k - start]

        return ddery

    def norm(self, field):
        H1x = self.derx(field)
        H1y = self.dery(field)
        return np.sqrt(np.sum(np.square(field)) + np.sum(np.square(H1x)) + np.sum(np.square(H1y)))


class Reality:
    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _grid, _source_term, _noiselevel):
        self.source_term = _source_term
        self.noiselevel = _noiselevel
        self.grid = _grid
        self.field = np.zeros([self.grid.nx, self.grid.ny])

    def GetGrid(self):
        return self.grid

    def GetT(self):
        return self.field

    def GetTWithNoise(self):
        field_with_noise = np.zeros([self.grid.nx, self.grid.ny])
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                field_with_noise[i][j] = random.gauss(self.field[i][j], self.noiselevel)
        return field_with_noise

    def Compute(self):
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                self.field[i][j] = 1.-self.source_term * (
                    (self.grid.coordx[i][j])**2 + (self.grid.coordy[i][j])**2)/4.

    def plot(self, field):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        surf = ax.plot_surface(self.grid.coordx, self.grid.coordy, field, rstride=1,
                               cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


class Simulation:
    # --------------------------------PARAMETERS--------------------------------
    dt = 0.01
    cfl = 1./4.

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _grid, _source_term):
        def indx(i, j):
            return j + i * ny

        self.source_term = _source_term
        self.grid = _grid
        nx = _grid.nx
        ny = _grid.ny
        self.size = nx * ny
        if self.cfl:
            self.dt = min([self.grid.dx**2, self.grid.dy**2]) * self.cfl
        print "cfl = ", max([self.dt/(self.grid.dx**2), self.dt/(self.grid.dy**2)])
        print "dt = ", self.dt

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
            self.Mat[indx(nx-1, j)] = np.zeros([self.size])
            self.Mat[indx(0, j)][indx(0, j)] = 1.
            self.Mat[indx(nx-1, j)][indx(nx-1, j)] = 1.
            self.rhs[indx(0, j)] = 0.
            self.rhs[indx(nx-1, j)] = 0.
        for i in range(nx):
            self.Mat[indx(i, 0)] = np.zeros([self.size])
            self.Mat[indx(i, ny-1)] = np.zeros([self.size])
            self.Mat[indx(i, 0)][indx(i, 0)] = 1.
            self.Mat[indx(i, ny-1)][indx(i, ny-1)] = 1.
            self.rhs[indx(i, 0)] = 0.
            self.rhs[indx(i, ny-1)] = 0.
        self.field = np.zeros([self.size])

    def GetT(self):
        return np.reshape(self.field, [self.grid.nx, self.grid.ny])

    def SetT(self, _field):
        self.field = np.reshape(_field, [self.size])

    # Increment through the next time step of the simulation.
    def Step(self):

        new_field = self.Mat.dot(self.field)
        new_field += self.rhs

        _conv = np.sqrt(np.sum(np.square(self.field - new_field)))
        self.field = new_field

        return _conv

    def compute(self):
        i = 0
        conv = 1.
        err = 999.
        while conv > 1e-3:
            i += 1
            conv = self.Step()

    def data_gen(self):
        i = 0
        conv = 1.
        err = 999.
        while conv > 1e-3:
            i += 1
            conv = self.Step()
            T_sim = self.GetT()
            err = self.grid.norm(T_ref - T_sim)
            yield T_sim, i, err, conv

        # print i, conv
        print "Norme H1 de la simulation", err

    def animate(self):
        # For animation purpose
        def run(data):
            # update the data
            field, i, err, conv = data
            ax.clear()
            surf = ax.plot_surface(self.grid.coordx, self.grid.coordy, field, rstride=1,
                                   cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_title('It = ' + str(i) + ',\n conv = ' + str(conv) + ',\n err = ' + str(err))
            return surf,

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ani = animation.FuncAnimation(fig, run, self.data_gen, blit=False, interval=10,
                                      repeat=False)
        plt.show()


class Chaleur:
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

    def plot(self, field):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        surf = ax.plot_surface(self.grid.coordx, self.grid.coordy, field, rstride=1,
                               cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
# ------------------------ Initialize "reality"  ----------------------------

edp = Chaleur()
edp.reality.Compute()

T_ref = edp.reality.GetT()
H1_ref = edp.grid.norm(T_ref)
edp.reality.plot(T_ref)

# ------------------------ Operate Observation ----------------------------

T_mes = edp.reality.GetTWithNoise()
H1_mes = edp.grid.norm(T_ref - T_mes) / H1_ref
print "Norme H1 de la mesure", H1_mes
edp.reality.plot(T_mes)

# ------------------------ Compute simulation without Kalman ----------------------------

# Bad initial solution and boundary condition
edp.simulation.SetT(edp.reality.GetTWithNoise())
print "Simulation sans Kalman..."

if False:
    edp.simulation.compute()
    T_sim = edp.simulation.GetT()
    H1_sim = edp.grid.norm(T_ref - T_sim) / H1_ref
    print "Norme H1 de la simu", H1_sim
    edp.reality.plot(T_sim)
else:
    edp.simulation.animate()

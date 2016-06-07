#!/usr/bin/python2

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation


class Grid:
    Lx = 2.
    Ly = 3.

    def __init__(self, _nx, _ny):
        self.nx = _nx
        self.ny = _ny
        coordx = np.linspace(-self.Lx/2., self.Lx/2., _nx)
        coordy = np.linspace(-self.Ly/2., self.Ly/2., _ny)
        self.coordy, self.coordx = np.meshgrid(coordy, coordx)
        self.dx = self.Lx/(self.nx-1.)
        self.dy = self.Ly/(self.ny-1.)
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


class Reality:
    # --------------------------------PARAMETERS--------------------------------
    noiselevel = .2
    source_term = 2.

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _nx, _ny):
        self.grid = Grid(_nx, _ny)
        self.field = np.zeros([_nx, _ny])

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


class Simulation:
    # --------------------------------PARAMETERS--------------------------------
    source_term = 2.
    dt = 0.01

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _nx, _ny):
        self.grid = Grid(_nx, _ny)
        self.field = np.zeros([_nx, _ny])
        self.size = _nx * _ny
        # print "cfl = ", max([self.dt/(self.grid.dx**2), self.dt/(self.grid.dy**2)])
        self.dt = min([self.grid.dx**2, self.grid.dy**2])/4
        print "cfl = ", max([self.dt/(self.grid.dx**2), self.dt/(self.grid.dy**2)])
        # compute matrix
        # time
        self.Mat = np.eye(self.size, self.size)
        # Laplacien
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                self.field = np.zeros([_nx, _ny])
                self.field[i][j] = self.dt
                self.Mat[j + i * _ny] += np.reshape(self.grid.dderx(self.field), [self.size])
                self.Mat[j + i * _ny] += np.reshape(self.grid.ddery(self.field), [self.size])
        # rhs and boundary conditions
        self.rhs = np.zeros([self.grid.nx, self.grid.ny]) + self.source_term * self.dt
        self.Mat = self.Mat.transpose()
        for j in range(self.grid.ny):
            self.Mat[j +       0 * _ny] = np.zeros([self.size])
            self.Mat[j + (_nx-1) * _ny] = np.zeros([self.size])
            self.Mat[j + 0 * _ny][j + 0 * _ny] = 1.
            self.Mat[j + (_nx-1) * _ny][j + (_nx-1) * _ny] = 1.
            self.rhs[0][j] = 0.
            self.rhs[self.grid.nx - 1][j] = 0.
            # self.Mat[j + 0 * _ny][j + 0 * _ny] = 0.
            # self.Mat[j + (_nx - 1) * _ny][j + (_nx - 1) * _ny] = 0.
            # self.rhs[0][j] = bnd_field[0][j]
            # self.rhs[self.grid.nx - 1][j] = bnd_field[self.grid.nx - 1][j]
        for i in range(self.grid.nx):
            self.Mat[0     + i * _ny] = np.zeros([self.size])
            self.Mat[_ny-1 + i * _ny] = np.zeros([self.size])
            self.Mat[0     + i * _ny][0     + i * _ny] = 1.
            self.Mat[_ny-1 + i * _ny][_ny-1 + i * _ny] = 1.
            self.rhs[i][0] = 0.
            self.rhs[i][self.grid.ny - 1] = 0.
            # self.Mat[0 + i * _ny][0 + i * _ny] = 0.
            # self.Mat[_ny - 1 + i * _ny][_ny - 1 + i * _ny] = 0.
            # self.rhs[i][0] = bnd_field[i][0]
            # self.rhs[i][self.grid.ny - 1] = bnd_field[i][self.grid.ny - 1]
        self.field = np.zeros([_nx, _ny])

    def GetGrid(self):
        return self.grid

    def GetT(self):
        return self.field

    # Increment through the next time step of the simulation.
    def Step(self):

        vec = np.reshape(self.field, [self.size])
        vec = self.Mat.dot(vec)
        new_field = np.reshape(vec, [self.grid.nx, self.grid.ny])
        new_field += self.rhs

        _conv = np.sqrt(np.sum(np.square(self.field - new_field)))
        self.field = new_field

        return _conv

    def data_gen(self):
        i = 0
        conv = 1.
        err = 999.
        while conv > 1e-3:
            i += 1
            conv = self.Step()
            T_sim = self.GetT()
            H1 = T_ref - T_sim
            H1x = mesh.derx(H1)
            H1y = mesh.dery(H1)
            err = np.sqrt(np.sum(np.square(H1)) + np.sum(np.square(H1x)) + np.sum(np.square(H1y))) / H1_ref
            yield T_sim, i, err, conv

        # print i, conv
        print "Norme H1 de la simulation", err

# Implements a linear Kalman filter.
class KalmanFilterLinear:
    def __init__(self, _M, _X, _S, _Q, _R, _nx, _ny):
        self.size = _S.shape[0]
        self.fieldsize = [_nx, _ny]
        self.Id = np.eye(self.size)
        self.sim = Simulation(_nx, _ny)
        self.sim.field = np.reshape(X, self.fieldsize)

        self.Phi = self.sim.Mat  # State transition matrix.
        self.M = _M              # Observation matrix.
        self.X = _X              # Initial state estimate.
        self.S = _S              # Initial covariance estimate.
        self.Q = _Q              # Estimated error in process.
        self.R = _R              # Estimated error in measurements.

    def GetCurrentState(self):
        return self.X

    def Step(self, _Y):

        # -------------------------Prediction step-----------------------------
        # Simulation
        conv = self.sim.Step()
        X_new = np.reshape(self.sim.GetT(), self.size)
        # predict covariance estimate
        self.S = self.Phi.dot(self.S).dot(np.transpose(self.Phi)) + self.Q

        # ------------------------Observation step-----------------------------
        innovation_covariance = self.M.dot(self.S).dot(np.transpose(self.M)) + self.R
        innovation = _Y - self.M.dot(X_new)

        # ---------------------------Update step-------------------------------
        # compute Kalman filter
        K = self.S.dot(np.transpose(self.M)).dot(np.linalg.inv(innovation_covariance))
        # apply Kalman filter
        X_new = X_new + K.dot(innovation)
        self.S = (self.Id - K.dot(self.M)).dot(self.S)

        # compute L2 convergence to stationary solution
        conv = np.sqrt(np.sum(np.square(self.X - X_new)))

        # save fields
        self.X = X_new
        self.sim.field = np.reshape(self.X, self.fieldsize)
        return conv

    def data_gen(self):
        i = 0
        conv = 1.
        err = 999.
        while conv > 1e-2:
            i += 1
            Y = np.reshape(reality.GetTWithNoise(), [simulation.size])
            conv = self.Step(Y)
            X = self.GetCurrentState()
            T_kal = np.reshape(X, [nx, ny])
            H1 = T_ref - T_kal
            H1x = mesh.derx(H1)
            H1y = mesh.dery(H1)
            err = np.sqrt(np.sum(np.square(H1)) + np.sum(np.square(H1x)) + np.sum(np.square(H1y))) / H1_ref
            # if divmod(i, 10):
            yield T_kal, i, err, conv
        # print i, conv, err
        print "Norme H1 du Kalman", err


# For animation purpose
def run(data):
    # update the data
    field, i, err, conv = data
    ax.clear()
    surf = ax.plot_surface(mesh.coordx, mesh.coordy, field, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title('It = ' + str(i) + ',\n conv = ' + str(conv) + ',\n err = ' + str(err))
    return surf,
# ===========================REAL PROGRAM START================================

# Size of the problem
nx, ny = 10, 20

# ------------------------ Initialize arrays ----------------------------
T_ref = np.zeros([nx, ny])
T_mes = np.zeros([nx, ny])
T_bnd = np.zeros([nx, ny])
T_sim = np.zeros([nx, ny])
T_kal = np.zeros([nx, ny])

H1 = np.zeros([nx, ny])
H1x = np.zeros([nx, ny])
H1y = np.zeros([nx, ny])

# ------------------------ Initialize "reality"  ----------------------------

reality = Reality(nx, ny)
reality.Compute()
mesh = reality.GetGrid()
T_ref = reality.GetT()

H1 = T_ref
H1x = mesh.derx(H1)
H1y = mesh.dery(H1)
H1_ref = np.sqrt(np.sum(np.square(H1)) + np.sum(np.square(H1x)) + np.sum(np.square(H1y)))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
surf = ax.plot_surface(mesh.coordx, mesh.coordy, T_ref, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# ------------------------ Operate Observation ----------------------------

T_mes = reality.GetTWithNoise()
H1 = T_ref - T_mes
H1x = mesh.derx(H1)
H1y = mesh.dery(H1)
err = np.sqrt(np.sum(np.square(H1)) + np.sum(np.square(H1x)) + np.sum(np.square(H1y))) / H1_ref
print "Norme H1 de la mesure", err

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
surf = ax.plot_surface(mesh.coordx, mesh.coordy, T_mes, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# ------------------------ Compute simulation without Kalman ----------------------------

simulation = Simulation(nx, ny)
# Bad initial solution and boundary condition
T_bnd = reality.GetTWithNoise()
simulation.field = T_bnd


print "Simulation sans Kalman..."
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ani = animation.FuncAnimation(fig, run, simulation.data_gen, blit=False, interval=10,
                              repeat=False)
plt.show()

# ------------------------ Compute simulation with Kalman ----------------------------

M = np.eye(simulation.size)  # Observation matrix.
S = np.eye(simulation.size)        # Initial covariance estimate.
R = np.eye(simulation.size) * 0.2  # Estimated error in measurements.
Q = np.eye(simulation.size) * 0    # Estimated error in process.
X = np.zeros([simulation.size])
Y = np.zeros([simulation.size])

# Bad initial condition and boundary condition
X = np.reshape(reality.GetTWithNoise(), [simulation.size])
kf = KalmanFilterLinear(M, X, S, Q, R, nx, ny)

print "Simulation avec Kalman..."
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ani = animation.FuncAnimation(fig, run, kf.data_gen, blit=False, interval=10,
                              repeat=False)
plt.show()

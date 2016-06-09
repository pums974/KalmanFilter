#!/usr/bin/python2

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from kalman import KalmanFilter


class Reality:
    g = - 9.81
    power = 100.
    dir = 45.

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _Tfin, _It, _noiselevel):
        self.noiselevel = _noiselevel
        self.Tfin = _Tfin
        self.It = _It
        self.dt = _Tfin/_It
        self.field = np.zeros([4])
        self.init = np.array([0.,
                              self.power * math.cos(self.dir * math.pi / 180.),
                              0.,
                              self.power * math.sin(self.dir * math.pi / 180.)])

    def GetSol(self):
        return self.field

    def GetSolWithNoise(self):
        field_with_noise = np.zeros([4])
        for i in range(4):
            field_with_noise[i] = random.gauss(self.field[i], self.noiselevel)
        return field_with_noise

    def Compute(self):
        for i in range(self.It):
            time = i * self.dt
            self.field[0] = self.init[0] + self.init[1] * time
            self.field[1] = self.init[1]
            self.field[2] = self.init[2] + self.init[3] * time + 0.5 * self.g * time * time
            self.field[3] = self.init[3] + self.g * time
            yield i


class Simulation:
    # --------------------------------VARIABLES----------------------------------
    power = 100
    dir = 45

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _Tfin, _It, _noiselevel):
        self.noiselevel = _noiselevel
        self.Tfin = _Tfin
        self.It = _It
        self.dt = _Tfin/_It
        self.field = np.zeros([_It, 4])
        self.Mat = np.array([[1, self.dt, 0, 0],
                             [0,  1, 0,  0],
                             [0,  0, 1, self.dt],
                             [0,  0, 0,  1]])
        self.source_term = np.array([0,
                                     0,
                                     -0.5 * 9.81 * self.dt * self.dt,
                                     -9.81 * self.dt])

    def GetSol(self):         # Get current Solution
        return self.field

    def SetSol(self, field):  # Set current Solution
        self.field = field

    def Step(self):           # Increment through the next time step of the simulation.
        self.field = self.Mat.dot(self.field) + self.source_term


class KalmanWrapper:
    def __init__(self, _reality, _sim):
        self.reality = _reality
        self.kalsim = _sim
        self.It = _sim.It
        _M = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0]])  # Observation matrix.

        self.kalman = KalmanFilter(self.kalsim, _M)
        self.kalman.S = np.eye(self.kalman.size_s)  # Initial covariance estimate.
        self.kalman.R = np.eye(self.kalman.size_o) * 0.2  # Estimated error in measurements.
        self.kalman.Q = np.eye(self.kalman.size_s) * 0.  # Estimated error in process.

    def SetMes(self, field):
        self.kalman.Y = self.kalman.M.dot(field[..., np.newaxis])

    def GetSol(self):
        return self.kalman.X.flatten()

    def SetSol(self, field):
        self.kalman.X = field[..., np.newaxis]
        self.kalsim.SetSol(field)

    def Step(self):
        self.kalsim.Step()
        self.SetSol(self.kalsim.GetSol())
        self.kalman.Apply()
        self.kalsim.SetSol(self.GetSol())


class Drop:
    Tfin = 20.
    Iterations = 200
    noiselevel = 50

    def __init__(self):
        self.reality = Reality(self.Tfin, self.Iterations, self.noiselevel)
        self.simulation = Simulation(self.Tfin, self.Iterations, self.noiselevel)
        self.kalsim = Simulation(self.Tfin, self.Iterations, self.noiselevel)
        self.kalman = KalmanWrapper(self.reality, self.kalsim)

    def Compute(self, simu):
        for i in range(simu.It):
            if i > 0:
                simu.Step()
            yield i

    def plot(self, field):
        fig = plt.figure()
        ax = fig.gca()
        plt.xlabel('X position')
        plt.ylabel('Y position')
        # ax.set_xlim(0, 1500)
        # plt.title('Measurement of a Cannonball in Flight')
        plt.plot(field[:, 0], field[:, 2])
        # plt.legend(('reality', 'simulated', 'measured', 'kalman'))
        plt.show()

    def plotall(self, field_ref, field_mes, field_sim, field_kal):
        fig = plt.figure()
        ax = fig.gca()
        plt.xlabel('X position')
        plt.ylabel('Y position')
        ax.set_xlim(0, 1500)
        ax.set_ylim(-600, 400)
        # plt.title('Measurement of a Cannonball in Flight')
        plt.plot(field_ref[:, 0], field_ref[:, 2], '-',
                 field_mes[:, 0], field_mes[:, 2], ':',
                 field_sim[:, 0], field_sim[:, 2], '-',
                 field_kal[:, 0], field_kal[:, 2], '--')
        plt.legend(('reality', 'measured', 'simulated', 'kalman'))
        plt.show()

    def norm(self, field):
        return np.sqrt(np.sum(np.square(field)))


# ------------------------ Begin program ----------------------------

edp = Drop()
Sol_ref = np.zeros([edp.Iterations, 4])
Sol_mes = np.zeros([edp.Iterations, 4])
Sol_sim = np.zeros([edp.Iterations, 4])
Sol_kal = np.zeros([edp.Iterations, 4])

# ----------------- Compute reality and measurement --------------------
for it in edp.reality.Compute():
    Sol_ref[it, :] = edp.reality.GetSol()
    Sol_mes[it, :] = edp.reality.GetSolWithNoise()

Norm_ref = edp.norm(Sol_ref)
# edp.plot(Sol_ref)

Err_mes = edp.norm(Sol_ref - Sol_mes) / Norm_ref
print "Norme H1 de la mesure", Err_mes
# edp.plot(Sol_mes)

# ------------------------ Compute simulation without Kalman ----------------------------

# Bad initial solution
edp.simulation.SetSol(Sol_mes[0, :])

for it in edp.Compute(edp.simulation):
    Sol_sim[it, :] = edp.simulation.GetSol()

Err_sim = edp.norm(Sol_ref - Sol_sim) / Norm_ref
print "Norme H1 de la simu", Err_sim
# edp.plot(Sol_sim)

# ------------------------ Compute simulation with Kalman ----------------------------

# Bad initial solution
edp.kalsim.SetSol(Sol_mes[0, :])
edp.kalman.SetMes(Sol_mes[1, :])

it = 0
for it in edp.Compute(edp.kalman):
    Sol_kal[it, :] = edp.kalman.GetSol()
    if it < edp.Iterations-1:
        edp.kalman.SetMes(Sol_mes[it+1, :])
    # edp.plot(Sol_kal)

Err_kal = edp.norm(Sol_ref - Sol_kal) / Norm_ref
print "Norme H1 de la simu filtre", Err_kal
# edp.plot(Sol_kal)

# ------------------------ Final plot ----------------------------

edp.plotall(Sol_ref, Sol_mes, Sol_sim, Sol_kal)

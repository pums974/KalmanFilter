#!/usr/bin/python2

import math
import random
import numpy as np
import matplotlib.pyplot as plt


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
        self.field = np.zeros([_It, 4])
        self.field[0] = np.array([0.,
                                  self.power * math.cos(self.dir * math.pi / 180.),
                                  0.,
                                  self.power * math.sin(self.dir * math.pi / 180.)])

    def GetSol(self):
        return self.field

    def GetSolWithNoise(self):
        field_with_noise = np.zeros([self.It, 4])
        for i in range(self.It):
            for j in range(4):
                field_with_noise[i][j] = random.gauss(self.field[i][j], self.noiselevel)
        return field_with_noise

    def Compute(self):
        for i in range(self.It):
            t = i * self.dt
            self.field[i][0] = self.field[0][1] * t + 0.
            self.field[i][1] = self.field[0][1]
            self.field[i][2] = 0.5 * self.g * t *t + self.field[0][3] * t + 0.
            self.field[i][3] = self.g * t + self.field[0][3]


class Simulation:
    # --------------------------------VARIABLES----------------------------------
    power = 100
    dir = 45
    current_it = 0

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
        self.field[0] = np.array([0.,
                                  self.power * math.cos(self.dir * math.pi / 180.),
                                  0.,
                                  self.power * math.sin(self.dir * math.pi / 180.)])

    def GetSol(self):
        return self.field

    def SetSol(self, field):
        self.field = field

    # Increment through the next time step of the simulation.
    def Step(self):
        self.field[self.current_it + 1, :] = self.Mat.dot(self.field[self.current_it, :])\
                                           + self.source_term

    def compute(self):
        for i in range(self.It):
            if i > 0:
                self.Step()
                self.current_it += 1


class Drop:
    Tfin = 20.
    Iterations = 200
    noiselevel = 50

    def __init__(self):
        self.reality = Reality(self.Tfin, self.Iterations, self.noiselevel)
        self.simulation = Simulation(self.Tfin, self.Iterations, self.noiselevel)

    def plot(self, field):
        fig = plt.figure()
        ax = fig.gca()
        plt.xlabel('X position')
        plt.ylabel('Y position')
        ax.set_xlim(0, 1500)
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
        # plt.title('Measurement of a Cannonball in Flight')
        plt.plot(field_ref[:, 0], field_ref[:, 2], '-',
                 field_mes[:, 0], field_mes[:, 2], ':',
                 field_sim[:, 0], field_sim[:, 2], '-',
                 field_kal[:, 0], field_kal[:, 2], '--')
        plt.legend(('reality', 'measured', 'simulated', 'kalman'))
        plt.show()

    def norm(self, field):
        return np.sqrt(np.sum(np.square(field)))

# ------------------------ Initialize "reality"  ----------------------------

edp = Drop()
edp.reality.Compute()

Sol_ref = edp.reality.GetSol()
Norm_ref = edp.norm(Sol_ref)
# edp.plot(Sol_ref)

# ------------------------ Operate Observation ----------------------------

Sol_mes = edp.reality.GetSolWithNoise()
Err_mes = edp.norm(Sol_ref - Sol_mes) / Norm_ref
print "Norme H1 de la mesure", Err_mes
# edp.plot(Sol_mes)

# ------------------------ Compute simulation without Kalman ----------------------------

# Bad initial solution
edp.simulation.SetSol(edp.reality.GetSolWithNoise())
print "Simulation sans Kalman..."

if True:
    edp.simulation.compute()
    Sol_sim = edp.simulation.GetSol()
    Err_sim = edp.norm(Sol_ref - Sol_sim) / Norm_ref
    print "Norme H1 de la simu", Err_sim
    # edp.plot(Sol_sim)
else:
    edp.simulation.animate()
    Sol_sim = edp.simulation.GetSol()


edp.plotall(Sol_ref, Sol_mes, Sol_sim, Sol_sim)
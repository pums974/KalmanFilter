#!/usr/bin/python2

import math
import random
import numpy as np
import time
import matplotlib.pyplot as plt


# Implements a linear Kalman filter.
class KalmanFilterLinear:
    def __init__(self, _Phi, _B, _M, _X, _S, _Q, _R):
        self.Phi = _Phi  # State transition matrix.
        self.B = _B      # Control matrix.
        self.M = _M      # Observation matrix.
        self.X = _X      # Initial state estimate.
        self.S = _S      # Initial covariance estimate.
        self.Q = _Q      # Estimated error in process.
        self.R = _R      # Estimated error in measurements.
        self.Id = np.eye(self.S.shape[0])

    def GetCurrentState(self):
        return self.X

    def Step(self, _control_vector, _Y):

        # -------------------------Prediction step-----------------------------
        X_new = self.Phi.dot(self.X) + self.B.dot(_control_vector)
        self.S = self.Phi.dot(self.S).dot(np.transpose(self.Phi)) + self.Q

        # ------------------------Observation step-----------------------------
        innovation_covariance = self.M.dot(self.S).dot(np.transpose(self.M)) + self.R
        innovation = _Y - self.M.dot(X_new)

        # ---------------------------Update step-------------------------------
        K = self.S.dot(np.transpose(self.M)).dot(np.linalg.inv(innovation_covariance))
        self.X = X_new + K.dot(innovation)
        self.S = (self.Id - K.dot(self.M)).dot(self.S)


class Reality:
    # --------------------------------PARAMETERS--------------------------------
    power = 100
    dir = 45
    noiselevel = 50  # How much noise should we add to measurements?

    # The initial location and velocity of the cannonball.
    vars = np.array([[0],
                     [power * math.cos(dir * math.pi / 180)],
                     [0],
                     [power * math.sin(dir * math.pi / 180)]])

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _Mat):
        self.Mat = _Mat

    def GetX(self):
        return self.vars[0][0]

    def GetY(self):
        return self.vars[2][0]

    def GetXWithNoise(self):
        return random.gauss(self.GetX(), self.noiselevel)

    def GetYWithNoise(self):
        return random.gauss(self.GetY(), self.noiselevel)

    # Increment through the next time step of the simulation.
    def Step(self, _source_term):
        self.vars = self.Mat.dot(self.vars) + _source_term


class Simulation:
    # --------------------------------VARIABLES----------------------------------
    power = 100
    dir = 45
    # The initial location and velocity of the cannonball.
    vars = np.array([[0],
                     [power * math.cos(dir * math.pi / 180)],
                     [0],
                     [power * math.sin(dir * math.pi / 180)]])

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _Mat):
        self.Mat = _Mat

    def GetX(self):
        return self.vars[0][0]

    def GetY(self):
        return self.vars[2][0]

    def GetXVelocity(self):
        return self.vars[1][0]

    def GetYVelocity(self):
        return self.vars[3][0]

    # Increment through the next time step of the simulation.
    def Step(self, _source_term):
        self.vars = self.Mat.dot(self.vars) + _source_term

# ===========================REAL PROGRAM START================================

dt = 0.1          # How many seconds should elapse per iteration?
iterations = 200  # How many iterations should the simulation run for?

# These are arrays to store the data points we want to plot at the end.
x_ref = []
y_ref = []
x_mes = []
y_mes = []
x_sim = []
y_sim = []
x_kal = []
y_kal = []

# Equations : x(t+1) = Phi * x(t) + source_term
Phi_real = np.array([[1, dt, 0, 0],
                     [0,  1, 0,  0],
                     [0,  0, 1, dt],
                     [0,  0, 0,  1]])

# Let's make a proper cannon simulation.
reality = Reality(Phi_real)

# Error in modelisation
Error = 0.01 * dt * np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
Phi_sim = Phi_real + Error

# Let's make a erroneous cannon simulation.
simulation = Simulation(Phi_sim)

# Observation matrix is the identity matrix, since we can get direct
# measurements of all values in our example.
M = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])  # Observation matrix.
B = np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])  # Control matrix.
S = np.eye(4)        # Initial covariance estimate.
R = np.eye(2) * 0.2  # Estimated error in measurements.
Q = np.eye(4) * 0    # Estimated error in process.

# This is our guess of the initial state.
X = np.array([[simulation.GetX()],
              [simulation.GetXVelocity()],
              [simulation.GetY()],
              [simulation.GetYVelocity()]])

kf = KalmanFilterLinear(Phi_sim, B, M, X, S, Q, R)

# Iterate through the simulation.
for i in range(iterations):
    # print(i, reality.GetX(), reality.GetY())
    # if not initial value
    if i > 0:
        source_term = np.array([[0],
                                [0],
                                [-0.5 * 9.81 * dt * dt],
                                [-9.81 * dt]])

        # Iterate the cannon simulation to the next time step.
        reality.Step(source_term)

        # Iterate the cannon simulation to the next time step.
        simulation.Step(source_term)

        # Operate measurement
        Y = np.array([[reality.GetXWithNoise()],
                      [reality.GetYWithNoise()]])

        # Apply Kalman filter
        kf.Step(source_term, Y)

    # Store obtained values for plot
    X = kf.GetCurrentState()

    x_ref.append(reality.GetX())
    y_ref.append(reality.GetY())
    x_mes.append(reality.GetXWithNoise())
    y_mes.append(reality.GetYWithNoise())
    x_sim.append(simulation.GetX())
    y_sim.append(simulation.GetY())
    x_kal.append(X[0][0])
    y_kal.append(X[2][0])


err = 0
for i in range(iterations):
    err += (x_ref[i] - x_mes[i])**2
    err += (y_ref[i] - y_mes[i])**2
print "error in measurement", np.sqrt(err)
err = 0
for i in range(iterations):
    err += (x_ref[i] - x_sim[i])**2
    err += (y_ref[i] - y_sim[i])**2
print "error in simulation", np.sqrt(err)
err = 0
for i in range(iterations):
    err += (x_ref[i] - x_kal[i])**2
    err += (y_ref[i] - y_kal[i])**2
print "error final", np.sqrt(err)

# Plot all the results we got.
# plt.ion()
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Measurement of a Cannonball in Flight')
plt.plot(x_ref, y_ref, '-', x_sim, y_sim, '-', x_mes, y_mes, ':', x_kal, y_kal, '--')
plt.legend(('reality', 'simulated', 'measured', 'kalman'))
plt.show()




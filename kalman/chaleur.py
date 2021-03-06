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
import matplotlib.pyplot as plt
try:
    from kalman.kalman import KalmanFilter, KalmanFilterObservator
    from kalman.skeleton import *
    from kalman.grid import Grid_DF2
    from kalman.tools import gc_clean
    if sys.version_info < (3, ):
        from kalman.libs.fortran_libs_py2 import inv
    else:
        from kalman.libs.fortran_libs_py3 import inv
    
except:
    from kalman import KalmanFilter, KalmanFilterObservator
    from skeleton import *
    from grid import Grid_DF2
    from tools import gc_clean
    if sys.version_info < (3, ):
        from libs.fortran_libs_py2 import inv
    else:
        from libs.fortran_libs_py3 import inv
import math
import sys

if sys.version_info < (3, ):
    range = xrange


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
        wx = 2. *  math.pi / self.grid.Lx
        wy = 2. *  math.pi / self.grid.Ly
        self.initfield = np.zeros([self.grid.nx, self.grid.ny])
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                x = self.grid.coordx[i][j]
                y = self.grid.coordy[i][j]
                self.initfield[i][j] = math.sin(wx * x) \
                                     * math.sin(wy * y)
        gc_clean()

    def step(self):
        """
            Compute the analytical solution
        """
        wt = 2. *  math.pi
        self.field = np.zeros([self.grid.nx, self.grid.ny])
        self.It += 1
        time = self.It * self.dt
        self.field = self.initfield * math.cos(time * wt)


class Simulation(SkelSimulation):
    """
    This class contains everything for the simulation
    """
    # --------------------------------PARAMETERS--------------------------------
    cfl = 1. / 4.
    isimplicit = True

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _grid, _power, _noiselevel, _dt):
        self.grid = _grid
        # self.dt = min([self.grid.dx**2, self.grid.dy**2]) * self.cfl
        self.dt = _dt

        SkelSimulation.__init__(self, _noiselevel, self.dt, self.grid.shape)
        self.power = _power

        # compute matrix
        self.calcmat()
        self.calcmatbc()

        # rhs and boundary conditions
        self.field = np.zeros([self.size])
        self.calcbc(self.It,self.field)
        
        if self.isimplicit:
#            self.Mat = np.linalg.inv(self.Mat)
            self.Mat = inv(self.Mat)
            
        indx = self.grid.indx
        self.staticfield = np.zeros([self.size])
        wx = 2. *  math.pi / self.grid.Lx
        wy = 2. *  math.pi / self.grid.Ly
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                x = self.grid.coordx[i][j]
                y = self.grid.coordy[i][j]
                self.staticfield[indx(i, j)] =  math.sin(wx * x) * math.sin(wy * y)
            
        gc_clean()

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
        if self.isimplicit:
            coef = -self.dt
        else:
            coef = self.dt
        # Laplacien
        for i in range(nx):
            for j in range(ny):
                self.field = np.zeros([nx, ny])
                self.field[i, j] = coef
                self.Mat[indx(i, j)] += np.reshape(
                    self.grid.dderx(self.field), [self.size])
                self.Mat[indx(i, j)] += np.reshape(
                    self.grid.ddery(self.field), [self.size])
        self.Mat = self.Mat.transpose()

    def calcmatbc(self):
        """
            Insert boundary condition inside the matrix of the diffusion problem
        :return:
        """
        indx = self.grid.indx
        nx = self.grid.nx
        ny = self.grid.ny

        def dirichlet(_i, _j):
            """
                Impose a Dirichlet boundary condition
            :param _i:
            :param _j:
            :param val:
            :return:
            """
            self.Mat[indx(_i, _j)] = np.zeros([self.size])
            self.Mat[indx(_i, _j)][indx(_i, _j)] = 1.

        def neumann(_i, _j, _dir):
            """
                Impose a neumann boundary condition
            :param _i:
            :param _j:
            :param val:
            :return:
            """
            self.field = np.zeros([nx, ny])
            self.field[_i, _j] = 1.
            if dir == 'x':
            	self.Mat[indx(_i, _j)] = np.reshape(
                	    self.grid.derx(self.field), [self.size])
            if dir == 'y':
            	self.Mat[indx(_i, _j)] = np.reshape(
                	    self.grid.dery(self.field), [self.size])

        for i in range(nx):
            dirichlet(i, 0)
            dirichlet(i, ny - 1)
        for j in range(ny):
            dirichlet(0, j)
            dirichlet(nx - 1, j)

    def calcbc(self,It, field):
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
            field[indx(_i, _j)] = val

        def neumann(_i, _j, val, _dir):
            """
                Do NOT impose a Neumann boundary condition
                Impose a non homogeneous Dirichlet boundary condition
            :param _i:
            :param _j:
            :param val:
            :return:
            """
            c10 = 16 * 1. / (math.pi**2)
            c20 = math.pi / self.grid.Lx
            c30 = 0.5 * math.pi / self.grid.Ly
            c40 = math.pi * math.pi / (4. * self.grid.Ly**2)
            c50 = math.pi * math.pi / (self.grid.Lx**2)
            x = self.grid.coordx[_i][_j]
            y = self.grid.coordy[_i][_j]
            time = It * self.dt
            field[indx(_i, _j)]  = c10 * math.sin(c20 * x) \
                                       * math.sin(c30 * y) \
                                       * math.exp(-(c40 + c50) * time)

        for i in range(nx):
            dirichlet(i, 0, 0.)
            dirichlet(i, ny - 1, 0.)
        for j in range(ny):
            dirichlet(0, j, 0.)
            dirichlet(nx - 1, j, 0.)

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
        power = 0.  # np.random.normal(self.power, self.noiselevel, self.rhs.shape)
        self.It += 1

        indx = self.grid.indx
        time = self.It * self.dt
        wx = 2. *  math.pi / self.grid.Lx
        wy = 2. *  math.pi / self.grid.Ly
        wt = 2. *  math.pi
        coef = ((wx ** 2 + wy ** 2) *  math.cos(wt * time)
                -        wt         *  math.sin(wt * time))
        self.rhs = np.zeros([self.size])
        self.rhs = self.staticfield * coef
        self.rhs *= self.dt
        if self.isimplicit:
            self.rhs += self.field         
#            self.calcbc(self.It, self.rhs)
            self.field = self.Mat.dot(self.rhs)
        else:
            self.field = self.Mat.dot(self.field) + self.rhs
#            self.calcbc(self.It, self.field)


class KalmanWrapper(SkelKalmanObservatorWrapper):
    """
        This class is use around the simulation to apply the kalman filter
    """

    def __init__(self, _reality, _sim):
        SkelKalmanWrapper.__init__(self, _reality, _sim)
        self.size = self.kalsim.size
        _M = self.getwindow()  # Observation matrix.
        self.kalman = KalmanFilterObservator(self.kalsim, _M)
        indx = self.kalsim.grid.indx

#        # Initial covariance estimate.
#        self.kalman.S = np.eye(self.kalman.size_s) * \
#            0. ** 2

#        # Estimated error in measurements.
#        self.kalman.R = np.eye(self.kalman.size_o) * \
#            self.reality.noiselevel ** 2

        # Estimated error in process.
        # G = np.zeros([self.kalman.size_s, 1])
        # for i in range(self.kalsim.grid.nx):
        #     for j in range(self.kalsim.grid.ny):
        #         G[indx(i, j)] = self.kalsim.grid.dx ** 4 / 24. \
        #                       + self.kalsim.grid.dy ** 4 / 24. \
        #                       + self.kalsim.dt ** 2 / 2.
        #         # G[indx(i, j)] = self.kalsim.dt
        #
        # self.kalman.Q = G.dot(np.transpose(G)) * self.kalsim.noiselevel ** 2

#        self.kalman.Q = np.eye(self.kalman.size_s) * \
#            self.kalnoise ** 2

        self.kalman.S = np.empty([self.kalman.size_s])
        self.kalman.R = np.empty([self.kalman.size_o])
        self.kalman.Q = np.empty([self.kalman.size_s])
        self.kalman.R.fill(self.reality.noiselevel ** 2)
        self.kalman.Q.fill(self.reality.noiselevel ** 2)
            
        gc_clean()

    def getwindow(self):
        """
            Produce the observation matrix : designate what we conserve of the noisy field
        :return: observation matrix
        """
        indx = self.kalsim.grid.indx
        M = np.eye(self.kalsim.size)
        # ep = 1
        # size_o = 2 * ep * self.kalsim.grid.nx + 2 * ep * (self.kalsim.grid.ny - 2 * ep)
        # M = np.zeros([size_o, self.kalsim.size])
        # k = 0
        # for i in range(self.kalsim.grid.nx):
        #     for j in range(0, ep):
        #         M[k][indx(i, j)] = 1.
        #         k += 1
        #     for j in range(self.kalsim.grid.ny - ep, self.kalsim.grid.ny):
        #         M[k][indx(i, j)] = 1.
        #         k += 1
        # for j in range(ep, self.kalsim.grid.ny - ep):
        #     for i in range(0, ep):
        #         M[k][indx(i, j)] = 1.
        #         k += 1
        #     for i in range(self.kalsim.grid.nx - ep, self.kalsim.grid.nx):
        #         M[k][indx(i, j)] = 1.
        #         k += 1
        return M

    def getsol(self):
        """
            Extract the solution from kalman filter : has to reshape it
        :return: current solution
        """
        return np.reshape(self.kalman.X,
                          [self.kalsim.grid.nx, self.kalsim.grid.ny])

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
    name = "Chaleur"
    Lx = 2.
    Ly = 2.
    power = 1.
    
    noise_sim = 0.01
    noise_real = 0.01
    
    nx = 20
    ny = 20
    dt = 5e-3
    nIt = 150

    def __init__(self):
        EDP.__init__(self)
        self.reinit()
        gc_clean()
        print("cfl = ",
              max([self.dt / (self.grid.dx**2), self.dt / (self.grid.dy**2)]))
        print("dt = ", self.dt)
        print("Norme H1 |  reality |   simu   |  kalman")

    def reinit(self):
        """
            Reinit everything
        :return:
        """
        self.grid = Grid_DF2(self.nx, self.ny, self.Lx, self.Ly)
        self.grid.coordx += self.grid.Lx / 2.
        self.grid.coordy += self.grid.Ly / 2.
        
#        self.dt = min([self.grid.dx**2, self.grid.dy**2]) / 4.
        self.nIt = int(1./self.dt)
        self.simulation = Simulation(self.grid, 0., self.noise_sim, self.dt)
        self.kalsim = Simulation(self.grid, 0., self.noise_sim, self.dt)

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
        simu.err = 0.
        for i in EDP.compute(self, simu):
            simu.err += self.norm(simu.getsol() - self.reality.getsol())
            yield i

    def norm(self, field):
        """
            Compute the H1 norm of the field
        :param field: field
        :return: norm H1
        """
        return self.grid.norm_l2(field)

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

    def run_test_case(self, graphs, kalonly=False):
        """
            Run the test case
        :return:
        """
        self.reinit()

        Norm_ref = 1.
        Err_mes = -1.
        Err_sim = -1.
        Err_kal = -1.

        # ----------------- Compute reality and measurement -------------------
        if graphs:
            self.animate(self.reality)
        else:
            for it in self.compute(self.reality):
                pass

        if not kalonly:
            if graphs:
                self.animatewithnoise(self.reality)

        if not kalonly:
            # ------------------------ Compute simulation without Kalman ----------
            self.reality.reinit()
            # Initial solution
            self.simulation.setsol(self.reality.getsol())
            if graphs:
                self.animate(self.simulation)
            else:
                for it in self.compute(self.simulation):
                    pass

        # ------------------------ Compute simulation with Kalman -------------
        self.reality.reinit()
        # Initial solution
        self.kalman.setsol(self.reality.getsol())

        if graphs:
            self.animate(self.kalman)
        else:
            for it in self.compute(self.kalman):
                pass
                
        Sol_ref = self.reality.getsol()
        Norm_ref = self.norm(Sol_ref)
        if not kalonly:
            Sol_mes = self.reality.getsolwithnoise()
            Err_mes = self.norm(Sol_ref - Sol_mes) / Norm_ref
            Sol_sim = self.simulation.getsol()
            Err_sim = self.norm(Sol_ref - Sol_sim) / Norm_ref
            
        Sol_kal = self.kalman.getsol()
        Err_kal = self.norm(Sol_ref - Sol_kal) / Norm_ref

        # ------------------------ Final output ----------------------------

        print("%8.2e | %8.2e | %8.2e | %8.2e" %
              (Norm_ref, Err_mes, Err_sim, Err_kal))
        # Norm_ref = self.grid.norm_inf(Sol_ref)
        # Err_mes = self.grid.norm_inf(Sol_mes)
        # Err_sim = self.grid.norm_inf(Sol_sim)
        # Err_kal = self.grid.norm_inf(Sol_kal)
        # print("%8.2e | %8.2e | %8.2e | %8.2e" %
        #       (Norm_ref, Err_mes, Err_sim, Err_kal))
        return self.simulation.err/self.simulation.It

def sim():
   Chaleur().run_test_case(True)

def compure_order():
    edp = Chaleur()

    nx = edp.nx
    ny = edp.ny
    dt = edp.dt


    edp.nx = nx
    edp.ny = ny
    edp.dt = dt/100


    order = 0.
    moy = 0.
    for i in range(1,5):
        if i>1:
          old_err = err
          old_nx = edp.nx
          edp.nx = int(math.floor(1.2* edp.nx))
          edp.ny = int(math.floor(1.2 * edp.ny))
          edp.reinit()


        err = edp.run_test_case(False)

        if i>1:
            order = ( math.log(old_err) - math.log(err)) /\
                    ( math.log(edp.nx) - math.log(old_nx))
            moy+=order
        print("%8.2e | %8.2e | %8.2e | %8.2e" %
              (edp.grid.dx, edp.grid.dy, err, order))

    print("spatial order : ", moy/3)

    edp.nx = nx*2
    edp.ny = ny*2
    edp.dt = dt

    order = 0.
    moy = 0.
    for i in range(1,5):
        if i>1:
          old_err = err
          old_dt = edp.dt
          edp.dt = 0.8 * edp.dt
          edp.reinit()


        err = edp.run_test_case(False)

        if i>1:
            order = ( math.log(old_err) - math.log(err)) /\
                    ( math.log(old_dt) - math.log(edp.dt))
            moy+=order
        print("%8.2e | %8.2e | %8.2e | %8.2e" %
              (edp.dt, edp.dt, err, order))

    print("time order : ", moy/3)

    
if __name__ == "__main__":
    sim()
#    compure_order()
    

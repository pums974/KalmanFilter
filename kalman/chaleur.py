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
from numpy.polynomial import polynomial as P
#import matplotlib.pyplot as plt
#from matplotlib import animation
from multiprocessing import Pool, Process, Queue, Array
from Queue import Empty
try:
    from kalman.kalman import KalmanFilter, KalmanFilterObservator
    from kalman.skeleton import *
    from kalman.grid import Grid_DF2
    from kalman.tools import gc_clean
    if sys.version_info < (3, ):
        from kalman.libs.fortran_libs_py2 import test_prec,inv
    else:
        from kalman.libs.fortran_libs_py3 import test_prec,inv
    
except:
    from kalman import KalmanFilter, KalmanFilterObservator
    from skeleton import *
    from grid import Grid_DF2
    from tools import gc_clean
    if sys.version_info < (3, ):
        from libs.fortran_libs_py2 import test_prec,inv
    else:
        from libs.fortran_libs_py3 import test_prec,inv
import math
import sys
import time
import timeit

if sys.version_info < (3, ):
    range = xrange

class Namespace: pass


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
        c1 = 16 * _power / (math.pi**2)
        c2 = math.pi / self.grid.Lx
        c3 = 0.5 * math.pi / self.grid.Ly
        c4 = math.pi * math.pi / (4. * self.grid.Ly**2)
        c5 = math.pi * math.pi / (self.grid.Lx**2)
        add = np.zeros([self.grid.nx, self.grid.ny])
        self.nmode = 1
        self.mmode = 1
        self.initfield = np.zeros([self.grid.nx, self.grid.ny])
        for m in range(1, self.nmode + 1):
            for n in range(1, self.mmode + 1):
                c10 = c1 / ((2. * n - 1.) * (2. * m - 1.))
                c20 = c2 * (2. * n - 1.)
                c30 = c3 * (2. * m - 1.)
                for i in range(self.grid.nx):
                    for j in range(self.grid.ny):
                        x = self.grid.coordx[i][j]
                        y = self.grid.coordy[i][j]
                        add[i][j] = c10 * math.sin(c20 * x) \
                                        * math.sin(c30 * y)
                self.initfield += add
                # print(m, n, self.grid.norm_h1(add))
                # if self.grid.norm_h1(add) < 0.001:
                #     break
        gc_clean()

    def step(self):
        """
            Compute the analytical solution
        """
        self.field = np.zeros([self.grid.nx, self.grid.ny])
        self.It += 1
        time = self.It * self.dt
        c4 = math.pi * math.pi / (4. * self.grid.Ly**2)
        c5 = math.pi * math.pi / (self.grid.Lx**2)
        for m in range(1, self.nmode + 1):
            for n in range(1, self.mmode + 1):
                c40 = c4 * (2. * n - 1.)**2
                c50 = c5 * (2. * m - 1.)**2
                self.field = self.initfield * math.exp(-(c40 + c50) * time)


class Simulation(SkelSimulation):
    """
    This class contains everything for the simulation
    """
    # --------------------------------PARAMETERS--------------------------------
    cfl = 1. / 2.
    isimplicit = False

    # ---------------------------------METHODS-----------------------------------
    def __init__(self, _grid, _power, _noiselevel, _dt):
        self.grid = _grid
        # self.dt = min([self.grid.dx**2, self.grid.dy**2]) * self.cfl
        self.dt = _dt
#        if min([self.grid.dx**2, self.grid.dy**2]) * self.cfl < self.dt:
#            print("CFL not respected, dt must be < ", min([self.grid.dx**2, self.grid.dy**2]) * self.cfl )
#            exit()

        SkelSimulation.__init__(self, _noiselevel, self.dt, self.grid.shape)
        self.power = _power

        # compute matrix
        self.calcmat()
        self.calcmatbc()
        self.Mat = self.Mat.transpose()

        # rhs and boundary conditions
        self.field = np.zeros([self.size])
        self.calcbc()
        
        if self.isimplicit:
#            self.Mat = np.linalg.inv(self.Mat)
            self.Mat = inv(self.Mat)
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
            if self.isimplicit:
                neumann(i, ny - 1, 'y')
            else:
                dirichlet(i, ny - 1)
        for j in range(ny):
            dirichlet(0, j)
            dirichlet(nx - 1, j)

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

            self.field[indx(_i, _j)] = -val
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
            if self.isimplicit:
                dirichlet(i, ny - 1, 0.)
            else:
                neumann(i, ny - 1, 0., np.array([0, -1]))
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
        self.rhs = np.zeros([self.size]) + power
        if self.isimplicit:
            self.calcbc()
        SkelSimulation.step(self)
        if not self.isimplicit:        
            self.calcbc()


class KalmanWrapper(SkelKalmanObservatorWrapper):
    """
        This class is use around the simulation to apply the kalman filter
    """

    def __init__(self, _reality, _sim, _kalnoise):
        self.kalnoise = _kalnoise
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
        self.kalman.Q.fill(self.kalnoise ** 2)
            
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
    noise_sim = 0.

#    noise_real = 2e-6
    noise_real = 3e-4
    nx = 30
    ny = 30
    dt = 1e-6
    nIt = 50
    
#    noise_real = 3e-3
#    nx = 5
#    ny = 5
#    dt = 1e-2
#    nIt = 50
    
    def __init__(self):
        EDP.__init__(self)
        self.grid = Grid_DF2(self.nx, self.ny, self.Lx, self.Ly)
        self.grid.coordx += self.grid.Lx / 2.
        self.grid.coordy += self.grid.Ly / 2.
        self.reinit(self.noise_sim)
        gc_clean()

        # print("cfl = ",
        #       max([self.dt / (self.grid.dx**2), self.dt / (self.grid.dy**2)]))
        # print("dt = ", self.dt)
        # print("Norme H1 |  reality |   simu   |  kalman")

    def reinit(self, _kalnoise):
        """
            Reinit everything
        :return:
        """
        self.simulation = Simulation(self.grid, 0., self.noise_sim, self.dt)
        self.kalsim = Simulation(self.grid, 0., self.noise_sim, self.dt)

        self.dt = self.simulation.dt
        self.reality = Reality(self.grid, self.power, self.noise_real, self.dt)

        self.simulation.nIt = self.nIt
        self.kalsim.nIt = self.nIt
        self.reality.nIt = self.nIt

        self.kalman = KalmanWrapper(self.reality, self.kalsim, _kalnoise)

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

    def run_test_case(self, graphs, _kalnoise, kalonly):
        """
            Run the test case
        :return:
        """
        self.reinit(_kalnoise)

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

#        print("%8.2e | %8.2e | %8.2e | %8.2e" %
#              (Norm_ref, Err_mes, Err_sim, Err_kal))
        # Norm_ref = self.grid.norm_inf(Sol_ref)
        # Err_mes = self.grid.norm_inf(Sol_mes)
        # Err_sim = self.grid.norm_inf(Sol_sim)
        # Err_kal = self.grid.norm_inf(Sol_kal)
        # print("%8.2e | %8.2e | %8.2e | %8.2e" %
        #       (Norm_ref, Err_mes, Err_sim, Err_kal))
        return Norm_ref, Err_mes, Err_sim, Err_kal


def do_work(q, r, flag, rank):
    edp = Chaleur()
    while True:
        try:
            kalnoise = q.get(block=False)
            if flag[0] == 1:
                # print("Got a job")
                Norm_ref, Err_mes, Err_sim, Err_kal = edp.run_test_case(False, kalnoise, True)
                # print("job done")
                r.put(Err_kal)
            # else:
                # print("emptying the queue")
        except Empty:
            if flag[0] == 1:
                # print("ready to compute")
                flag[rank] = 1
            if flag[0] == 0:
                # print("queue empty")
                flag[rank] = 0
            if flag[0] == -1:
                break


def find_min():

    def fx():
        err = 0.
        minerr = 1e10
        maxerr = 0.
        # print("ready to compute ?")
        compute[0] = 1
        while max(compute) < 1:
            pass
        # print("giving job")
        for it in range(100):
            work_queue.put(ns.kalnoise)
        # print("getting results")
        for it in range(100000):
            err1 = res_queue.get()
            work_queue.put(ns.kalnoise)
            err += err1
            minerr = min(minerr, err1)
            maxerr = max(maxerr, err1)
            # print(abs(it * err1 / err-1.))
            # print(err/it, err1, err1 / err)
            if err1 / err < ns.eps:
                break
        # print("getting the rest")
        compute[0] = 0
        # print("waiting for process")
        while max(compute) > 0:
            pass
        while not res_queue.empty():
            err1 = res_queue.get()
            err += err1
            it += 1
        # print("done")
        err = err / it
        confidence = (maxerr - minerr) / it

        return err, confidence


    def animate():
        """
            Perform the optimisation and produce animation a the same time
        :param simu: the simulation to perform
        :param compute: the generator to compute time stns.eps
        """

        def plot_update(data):
            """
                Update the plot
            :return: surface to be plotted
            """
            _kalnoise, _err, _conf = data
            ns.kalnoises.append(_kalnoise)
            ns.errs.append(_err)
            ns.confs.append(_conf*1e5)
            ax.clear()
#            surf = ax.scatter(ns.kalnoises, ns.errs, s=ns.confs, marker='.', alpha=0.5)
            surf = ax.scatter(ns.kalnoises, ns.errs, marker='.')
            ax.set_xlim(np.min(ns.kalnoises)*0.6, np.max(ns.kalnoises)*1.1)
            ax.set_ylim(np.min(ns.errs)*0.8, np.max(ns.errs)*1.1)
            return surf,

        fig, ax = plt.subplots()
        ax.set_xlim(0.,1.)
        ax.set_ylim(0.,1.)

        _ = animation.FuncAnimation(fig,
                                    plot_update,
                                    next_val)
        plt.show()

    def next_val1():
        ns.best_kalnoise *= 0.8
        old_best_kalnoise = ns.best_kalnoise - ns.alpha
        for gtf in range(3):
            cntgtf = 0
            cntlost = 0

            ns.kalnoise = ns.best_kalnoise
            ns.alpha1 = abs(old_best_kalnoise - ns.best_kalnoise) * 0.5
            if ns.alpha1 > 1e-10:
                ns.alpha = ns.alpha1
            old_best_kalnoise = ns.best_kalnoise

            ns.eps = ns.eps * 0.5
            it = 0
            print("start        -1 %9.2e " % (abs(ns.kalnoise)), end=" ")
            err, conf = fx()
            print(" %9.2e  %9.2e " % (err, conf))
            old_dir = 1

            old_kalnoise = ns.kalnoise
            old_err = err
            ns.best_err = err
            for opt in range(100):
                yield ns.kalnoise, err, conf
                ns.kalnoise += ns.alpha
                print("continuing  %3i %9.2e " % (opt, abs(ns.kalnoise)), end=" ")
                err, conf = fx()
                print(" %9.2e  %9.2e " % (err, conf), end=" ")
                der = (err - old_err) / (ns.kalnoise - old_kalnoise)
#                print(" %9.2e " % (der), end=" ")
                if abs(der) < 1e-10:
                    print("Fin")
                    break
                if abs(der) > 1e+10:
                    print("Fin : ns.alpha too small")
                    break
                dir = -der / abs(der)
                if err < ns.best_err:
                    ns.best_kalnoise = ns.kalnoise
                    ns.best_err = err
                    print("", end=" * ")
                if abs(err - old_err) < conf:
                    print("stagnation")
                else:
                    if dir != old_dir:
                        # print("gone too far",ns.kalnoise,old_kalnoise, ns.alpha)
                        # print("gone too far", ns.kalnoise, err, old_kalnoise,old_err)
                        print("gone too far")
                        old_dir = dir
                        # ns.kalnoise = old_kalnoise - ns.alpha
                        ns.alpha *= - 0.5
                        cntgtf += 1
                        if cntgtf >= 5:
                            print("Fin : gtf")
                            break
                    else:
                        # cntgtf = 0
                        if err > ns.best_err:
                            cntlost += 1
                        if cntlost > 20:
                            print("Fin : lost")
                            break
                        print("")
                # print("")
                old_kalnoise = ns.kalnoise
                old_err = err

            print("Fin opt")
        print("Fin gtf")

    def next_val_big_graph():
        for ns.alpha in [1e-6, 1e-7, 1e-8]:
            start = 0.
            end = 5. * ns.alpha
            for it1 in range(3):
                ns.kalnoise = start
                err, conf = fx()
                output = open("data.dat",'a')
                output.write("%.15f  %.15f\n" % (ns.kalnoise, err))
                output.close()
                yield ns.kalnoise, err, conf
                if err < ns.best_err:
                    print("%8.2e  %8.2e" % (ns.kalnoise, err))
                    ns.best_kalnoise = ns.kalnoise
                    ns.best_err = err
                while ns.kalnoise < end:
                    ns.kalnoise += ns.alpha
                    err, conf = fx()
                    output = open("data.dat",'a')
                    output.write("%.15f  %.15f\n" % (ns.kalnoise, err))
                    output.close()
                    yield ns.kalnoise, err, conf
                    if err < ns.best_err:
                        print("%8.2e  %8.2e" % (ns.kalnoise, err))
                        ns.best_kalnoise = ns.kalnoise
                        ns.best_err = err
                start = ns.best_kalnoise * 0.2
                ns.alpha = ns.best_kalnoise * 0.1

    def next_val():
        start = guess * 0.2
        ns.alpha = guess * 0.1
        end = guess * 3
        for it1 in range(2):
            ns.kalnoise = start
            err, conf = fx()
            output = open("data.dat",'a')
            output.write("%.15f  %.15f\n" % (ns.kalnoise, err))
            output.close()
            yield ns.kalnoise, err, conf
            if err < ns.best_err:
                print("%8.2e  %8.2e" % (ns.kalnoise, err))
                ns.best_kalnoise = ns.kalnoise
                ns.best_err = err
#            while ns.kalnoise < ns.best_kalnoise * 1.8:
            while ns.kalnoise < end:
                ns.kalnoise += ns.alpha
                err, conf = fx()
                output = open("data.dat",'a')
                output.write("%.15f  %.15f\n" % (ns.kalnoise, err))
                output.close()
                yield ns.kalnoise, err, conf
                if err < ns.best_err:
                    print("%8.2e  %8.2e  %8.2e" % (ns.kalnoise, err, ns.ref/err))
                    ns.best_kalnoise = ns.kalnoise
                    ns.best_err = err
            start = ns.best_kalnoise * 0.2
            ns.alpha = ns.best_kalnoise * 0.1
            end = ns.best_kalnoise * 1.8
        

    edp = Chaleur()   
    print("dx       = %8.2e" % (edp.grid.dx))
    print("dy       = %8.2e" % (edp.grid.dy))
    print("dt       = %8.2e" % (edp.dt))
    print("nr       = %8.2e" % (edp.noise_real))
    print("ns       = %8.2e" % (edp.noise_sim))
    print("nit      = %8.2e" % (edp.nIt))
    print("cfl      = %8.2e" % (max([edp.dt / (edp.grid.dx**2), edp.dt / (edp.grid.dy**2)])))
    
    # cas explicite (pas sûr pour le temps)
    guess = 1.03e-2 * (edp.grid.dx**4 + edp.grid.dy** 4) + \
            2.69e-01* edp.dt**2
            
#    guess = guess /10
            
    print("alpha * (dx**4 + dy**4 ) + beta * dt**2 = %8.2e" % (guess))

    Norm_ref, Err_mes, Err_sim, Err_kal = edp.run_test_case(False, guess, False)
    print("Norme H1 |  reality |   simu   |  kalman")
    print("%8.2e | %8.2e | %8.2e | %8.2e" %
          (Norm_ref, Err_mes, Err_sim, Err_kal))

    nproc = 64
    work_queue = Queue()
    res_queue = Queue()
    compute = Array('i', nproc+1)
    for i in range(nproc+1):
        compute[i] = 0

    processes = [Process(target=do_work,
                         args=(work_queue,
                               res_queue,
                               compute, i+1))
                 for i in range(nproc)]

    for p in processes:
        p.start()
    print("status       it  kalnoise       err        conf    comment")

    ns = Namespace()

    ns.ref = Err_sim
    ns.best_kalnoise = guess
    ns.best_err = 1e10
    ns.eps = 1e-2
    ns.alpha = guess * 0.25


    ns.kalnoises = []
    ns.errs = []
    ns.confs = []
    output = open("data.dat",'w')
    output.close()
    results_kalnoise = np.zeros(5)
    results_err = np.zeros(5)
        
#    for global_test in range(3):
##        for optim in next_val():
##            pass
#        animate()

#        results_kalnoise[global_test] = ns.best_kalnoise
#        results_err[global_test] = ns.best_err

#    for _kalnoise, _err, _conf in next_val_big_graph():
    for _kalnoise, _err, _conf in next_val():
        ns.kalnoises.append(_kalnoise)
        ns.errs.append(_err)
        ns.confs.append(_conf)
#    animate()
#    ns.output.close()
    print("FIN")
#    print(results_kalnoise)
#    print(ns.best_err)
        
    ns.kalnoises = np.array(ns.kalnoises)
    ns.errs = np.array(ns.errs)
            
    x = ns.kalnoises[ns.kalnoises.argsort()]
    y = ns.errs[ns.kalnoises.argsort()]
    coef = np.polyfit(x, y, 4)[::-1]
    for order in range(4, -1, -1):
        if(order>1):
            print("%8.2e*x**%1i" % (coef[order], order), end=" + ")
        elif(order>0):
            print("%8.2e*x" % (coef[order]), end=" + ")
        else:
            print("%8.2e " % (coef[order]))
    polyder = P.polyder(coef)
    polyder2 = P.polyder(coef,2)
    roots = P.polyroots(polyder)
    ns.best_err = 1e100
    for root in roots:
        ns.kalnoise = abs(root.real)
        der2 = P.polyval(ns.kalnoise,polyder2)
        err = P.polyval(ns.kalnoise, coef)
        if err < ns.best_err:
            ns.best_err = err
            ns.best_kalnoise = ns.kalnoise
            
    ns.kalnoise = ns.best_kalnoise
    err = ns.best_err
    ns.eps = ns.eps / 10.
#    err, conf = fx()
    print("%8.2e %8.2e %8.2e" % (ns.kalnoise, err, ns.best_err))
    output = open("data.dat",'a')
    output.write("%.15f  %.15f\n" % (ns.best_kalnoise, err))
    output.close()

    compute[0] = -1
    for p in processes:
        p.join()

if __name__ == "__main__":
#    print(1.+1e-16 -1., test_prec(1., 1e-16, 1.))
#    print(1.+1e-15 -1., test_prec(1., 1e-15, 1.))
    find_min()
#    Chaleur().run_test_case(False, 2.4e-7, False)
#    print(timeit.timeit('Chaleur().run_test_case(False, 2.4e-7, False)',
#          setup="from __main__ import Chaleur",
#          number=10))
    exit()


    def fx():
        err = 0.
        minerr = 1e10
        maxerr = 0.
        # print("ready to compute ?")
        compute[0] = 1
        while max(compute) < 1:
            pass
        # print("giving job")
        for it in range(100):
            work_queue.put(ns.kalnoise)
        # print("getting results")
        for it in range(100000):
            err1 = res_queue.get()
            work_queue.put(ns.kalnoise)
            err += err1
            minerr = min(minerr, err1)
            maxerr = max(maxerr, err1)
            # print(abs(it * err1 / err-1.))
            # print(err/it, err1, err1 / err)
            if err1 / err < ns.eps:
                break
        # print("getting the rest")
        compute[0] = 0
        while not res_queue.empty():
            err1 = res_queue.get()
            err += err1
            it += 1
        # print("waiting for process")
        while max(compute) > 0:
            pass
        # print("done")
        err = err / it
        confidence = (maxerr - minerr) / it

        print(it)
        return err, confidence

    nproc = 64
    work_queue = Queue()
    res_queue = Queue()
    compute = Array('i', nproc+1)
    for i in range(nproc+1):
        compute[i] = 0

    processes = [Process(target=do_work,
                         args=(work_queue,
                               res_queue,
                               compute, i+1))
                 for i in range(nproc)]

    for p in processes:
        p.start()
        

    ns = Namespace()

    ns.kalnoise = 2.4e-7
    ns.eps = 1e-2
        
    print(timeit.timeit('fx()',
          setup="from __main__ import fx",
          number=10))
          
    compute[0] = -1
    for p in processes:
        p.join()


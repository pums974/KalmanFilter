#!/usr/bin/python2
# coding=utf-8
"""
    This allow plotting capabilities
"""
# let's try using multiprocessing instead of threading module:
import multiprocessing as mp
from queue import Empty
import ctypes
import time
import numpy as np
import sys

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from imp import reload


class Animator:
    """
       truc
    """
    def animator(self, gridx_, gridy_, test_, _data_queue):
        global gridx, gridy, test
        gridx = gridx_
        gridy = gridy_

        # import matplotlib
        # matplotlib.use('QT4Agg')
        # import matplotlib.pyplot as plt
        # from matplotlib import animation
        # from matplotlib import cm
        # from mpl_toolkits.mplot3d import Axes3D

        # plt.disconnect

        class Timer(object):
            """
            truc
            """

            def __init__(self, _data_queue):
                self.data_queue = _data_queue
                self.field, self.it, self.err, self.test = self.data_queue.get(block=True)

            def timer(self):
                while self.test:
                    yield self.field, self.it, self.err
                    self.field, self.it, self.err, self.test = self.data_queue.get(block=True)

        class My_Figure(object):
            """
            truc
            """

            def __init__(self):
                self.fig = plt.figure()
                self.ax = self.fig.gca(projection='3d')
                self.ax.set_xlim(-1, 1)
                self.ax.set_ylim(-1, 1)
                self.ax.set_zlim(-1, 1)

            def plot_update(self, truc):
                """
                    Update the plot
                :return: surface to be plotted
                """
                field, it, err = truc
                self.ax.clear()
                surf = self.ax.plot_surface(gridx, gridy, field, rstride=1,
                                            cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                self.ax.set_title('It = ' + str(it) + ',\n err = ' + str(err))

                self.ax.set_xlim(-1, 1)
                self.ax.set_ylim(-1, 1)
                self.ax.set_zlim(-1, 1)
                return surf,

            def show(self):
                plt.show()

        my_timer = Timer(_data_queue)
        my_figure = My_Figure()

        _ = animation.FuncAnimation(my_figure.fig, my_figure.plot_update, my_timer.timer, blit=False, interval=10,
                                    repeat=False)
        if test_.value < 1:
            test_.value = 1
        my_figure.show()

        if test_.value < 2:
            test_.value = 2
        while test_.value == 2:
            pass
        time.sleep(1)

    def animate_mt(self, grid, simu, compute):
        """
            Perform the simulation and produce animation a the same time
        :param simu: the simulation to perform
        :param compute: the generator to compute time steps
        """

        #  create shared arrays
        shared_gridx = mp.Array(ctypes.c_double, grid.size)
        shared_gridy = mp.Array(ctypes.c_double, grid.size)
        gridx = np.frombuffer(shared_gridx.get_obj())
        gridy = np.frombuffer(shared_gridy.get_obj())
        gridx = grid.coordx.copy()
        gridy = grid.coordy.copy()
        test = mp.Value('i', 0)
        data_queue = mp.Queue()
        data_queue.put((simu.getsol(), 0, 0.0, True))

        #  Instanciate the child
        child = mp.Process(target=self.animator, args=(gridx, gridy, test, data_queue))
        child.start()

        #  Do the computation
        for i in compute(simu):
            data_queue.put((simu.getsol(), i, simu.err, True))
            if test.value == 2:
                test.value = 3
                break

        data_queue.put((simu.getsol(), i, simu.err, False))
        if test.value < 3:
            test.value = 3

        #  Kill the child
        child.join()

        # emptying the queue
        while True:
            try:
                _ = data_queue.get(block=False)
            except Empty:
                if data_queue.empty():
                    if data_queue.qsize() == 0:
                        break

    def animate_seq(self, grid, simu, compute):
        """
            Perform the simulation and produce animation a the same time
        :param simu: the simulation to perform
        :param compute: the generator to compute time steps
        """
        # import matplotlib
        # import matplotlib.pyplot as plt
        # from matplotlib import animation
        # from matplotlib import cm
        # from mpl_toolkits.mplot3d import Axes3D

        def plot_update(i):
            """
                Update the plot
            :param i: iteration number
            :return: surface to be plotted
            """
            ax.clear()
            surf = ax.plot_surface(grid.coordx, grid.coordy, simu.getsol(), rstride=1,
                                   cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_title('It = ' + str(i) + ',\n err = ' + str(simu.err))
            return surf,

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        _ = animation.FuncAnimation(fig, plot_update, compute(simu), blit=False, interval=10,
                                    repeat=False)
        plt.show()


    def animatewithnoise(self, grid, simu, compute, norm):
        """
            Perform the simulation and produce animation a the same time
        :param simu: the simulation to perform
        :param compute: the generator to compute time steps
        :param norm: the norm to compute noise norm
        """

        # import matplotlib
        # import matplotlib.pyplot as plt
        # from matplotlib import animation
        # from matplotlib import cm
        # from mpl_toolkits.mplot3d import Axes3D

        def plot_update(i):
            """
                Update the plot
            :param i: iteration number
            :return: surface to be plotted
            """
            ax.clear()
            surf = ax.plot_surface(grid.coordx, grid.coordy, simu.getsolwithnoise(), rstride=1,
                                   cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            simu.err = norm(simu.getsol() - simu.getsolwithnoise())
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_title('It = ' + str(i) + ',\n err = ' + str(simu.err))
            return surf,

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        _ = animation.FuncAnimation(fig, plot_update, compute(simu), blit=False, interval=10,
                                    repeat=False)
        plt.show()

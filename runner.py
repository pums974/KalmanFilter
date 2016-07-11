
#!/usr/bin/python2
# coding=utf-8
"""
    Run all the tests cases :

    1) Drop will solve the problem
    a = - g - h v
    and use the Kalman filter to mix
    1st order simulation and noisy measurements
    In order to produce a trajectory closer to the analytic solution
    The result is one plot

    2) Chaleur will solve the problem
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

# import cProfile
import sys
# import pstats
# import os
import time

Drop = True
Chaleur = True
Convection = True
graphs = True
profiling = False
if profiling:
    import pprofile
if Drop:
    from kalman.drop import Drop
if Chaleur:
    from kalman.chaleur import Chaleur
if Convection:
    from kalman.convection import Convection

if sys.version_info < (3,):
    range = xrange


def test_cases():
    """
    Generator of all the considered test cases
    :return:
    """
    if Drop:
        yield Drop()
    if Chaleur:
        yield Chaleur()
    if Convection:
        yield Convection()


def run_a_test_case(edp):
    """
        Do we run the tests cases multiple time ?
    :return:
    """
    t1 = time.clock()
    for i in range(1):
        edp.run_test_case(graphs)
    t2 = time.clock()
    print("Elapsed " + str(t2 - t1) + "s")


def run_all_tests_cases():
    """
        Run all the test cases
    :return:
    """
    for edp in test_cases():
        if profiling:
            profiler = pprofile.Profile()
            with profiler:
                run_a_test_case(edp)
            # profiler.print_stats()
            with open("profile." + edp.name, 'w') as fich:
                profiler.callgrind(fich)
            # cProfile.run("edp.run_test_case(graphs)", "profile." + edp.name)
            # p = pstats.Stats("profile." + edp.name)
            # p.sort_stats('time').print_stats(10)
            # p.sort_stats('cumulative').print_stats(10)
            # os.remove("profile." + edp.name)
        else:
            run_a_test_case(edp)

if __name__ == "__main__":
    run_all_tests_cases()

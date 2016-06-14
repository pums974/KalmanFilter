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
from drop import Drop
from chaleur import Chaleur
from convection import Convection
import cProfile
import pstats
import os
import pprofile

def test_cases():
    # yield Drop()
    # yield Chaleur()
    yield Convection()

graphs = True

for edp in test_cases():
    edp.run_test_case(graphs)
    # profiler = pprofile.Profile()
    # with profiler:
    #     edp.run_test_case(graphs)
    # # profiler.print_stats()
    # with open("profile.stat", 'w') as file:
    #     profiler.callgrind(file)
    # cProfile.run("edp.run_test_case(graphs)", "profile.stat")
    # p = pstats.Stats('profile.stat')
    # p.sort_stats('time').print_stats(10)
    # p.sort_stats('cumulative').print_stats(10)
    # os.remove("profile.stat")

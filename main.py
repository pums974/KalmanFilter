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
import numpy as np


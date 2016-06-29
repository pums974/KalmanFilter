#!/usr/bin/python2
# coding=utf-8
"""
    Setup
"""
# from distutils.core import setup
#
# files = ["libs/*"]
#
# setup(name="Kalman",
#       version="1",
#       description="Kalman sandbox",
#       author="Alexandre Poux",
#       author_email="alexandre.poux@univ-poitiers.fr",
#       url="https://github.com/pums974/KalmanFilter",
#       packages=['kalman'],
#       package_data={'kalman': files},
#       scripts=["runner"],
#       long_description="""Really long text here."""
#       )

import numpy.distutils.core
import setuptools

files = ["libs/*"]

# setup fortran 90 extension
# ---------------------------------------------------------------------------
ext1 = numpy.distutils.core.Extension(
    name='fortran_libs',
    sources=['kalman/libs/fortran_libs.f90'], )

# call setup
# --------------------------------------------------------------------------
numpy.distutils.core.setup(name="Kalman",
                           version="1",
                           description="Kalman sandbox",
                           author="Alexandre Poux",
                           author_email="alexandre.poux@univ-poitiers.fr",
                           url="https://github.com/pums974/KalmanFilter",
                           packages=['kalman'],
                           package_data={'kalman': files},
                           ext_modules=[ext1],
                           scripts=["runner"],
                           long_description="""Really long text here.""")

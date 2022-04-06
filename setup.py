# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Spinning Up repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    py_modules=['spinup'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle==1.2.1',
        'gym[atari,box2d,classic_control]~=0.15.3',
        'ipython',
        'joblib',
        'matplotlib==3.1.1',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn==0.8.1',
        'tensorflow>=1.8.0,<2.0',
        'torch==1.3.1',
        'tqdm'
    ],
    description="Teaching tools for introducing people to deep RL.",
    author="Joshua Achiam",
)

# from setuptools import setup

# setup(name="gym_reachability",
#       version="0.0.1",
#       install_requires=["gym", "numpy", "math", "pyglet", "Box2D", "sys", "mujoco_py"])

#!/usr/bin/env python3
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['drl_utils'],
    package_dir={'drl_utils': 'drl_utils'}
)

setup(**d)

#!/usr/bin/env python

from setuptools import setup

__author__ = 'anonymous'
__version__ = '1.0'

setup(
    name='risk',
    version=__version__,
    description='Library for A Risk-Sensitive Approach to Policy Optimization',
    long_description=open('README.md').read(),
    author=__author__,
    author_email='anonymous',
    license='BSD',
    packages=['risk'],
    keywords='deep reinforcement learning, risk-sensitive RL, constrained RL',
    classifiers=[],
    install_requires=['numpy', 'torch', 'torchvision', 'gym[atari]', 'scipy', 'tensorboard', 'mpi4py', 'matplotlib',
                      'seaborn']
)

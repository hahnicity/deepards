#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(name='deepards',
      version="2.0",
      description='Deep Learning For ARDS detection with Ventilator Waveform Data',
      packages=find_packages(),
      install_requires=[
          'dtwco',
          'prettytable',
          'ucdpvanalysis>=1.5',
          'imbalanced-learn==0.4.3',
          'pyyaml',
      ],
      entry_points={
      },
      )

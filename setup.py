#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(name='deepards',
      version="1.0",
      description='Deep Learning For ARDS detection with Ventilator Waveform Data',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'prettytable',
          'scipy',
          'scikit-learn<0.21.0',
          'ventmap',
          'ucdpvanalysis>=1.5',
          'imblearn==0.4.3',
      ],
      entry_points={
      },
      )

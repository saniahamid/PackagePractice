# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 21:11:24 2021

@author: hamid
"""

from setuptools import setup

setup(name = 'mlp',
      packages = ['mlp'],
      version = '0.01dev1',
      entry_points={
          'console_scripts': ['mlp-cli=mlp.cli:main']}
      )
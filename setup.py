import os
from setuptools import setup
from setuptools import find_packages

setup(name='NCL',
      author='Ta-Chu Kao and Kris Jensen',
      version='0.0.1',
      description='Natural Gradient Continual Learning',
      license='MIT',
      install_requires=['numpy==1.19.2', 'jax==0.2.11'],
      packages=find_packages())

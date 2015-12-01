#!/usr/bin/env python

from setuptools import setup, find_packages

__author__ = 'Eugenio Marinetto'

setup(
    name='ImageJTools',
    version="1.0.0",
    packages=find_packages(),
    install_requires=['ctk_cli','numpy', 'vtk', 'PythonTools','scipy', 'math', 'SimpleITK'],
    author='Eugenio Marinetto',
    author_email='marinetto@jhu.edu',
    description='',
)

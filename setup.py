#!/usr/bin/env python

from setuptools import setup, find_packages

__author__ = 'Eugenio Marinetto'

setup(
    name='TrabeculaeTools',
    version="1.0",
    packages=find_packages(),
    install_requires=['vtk','scipy','SimpleITK','pyimagej','matplotlib', 'numpy', 'scipy'],
    author='Eugenio Marinetto',
    author_email='marinetto@jhu.edu',
    description='Tools and Scripts for BoneJ analysis',
    url='http://github.com/nenetto/TrabeculaeTools',
    dependency_links = ['http://github.com/nenetto/pyimagej']
)

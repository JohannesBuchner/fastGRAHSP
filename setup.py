#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

try:
    with open('README.rst') as readme_file:
        readme = readme_file.read()

    with open('HISTORY.rst') as history_file:
        history = history_file.read()
except IOError:
    with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme_file:
        readme = readme_file.read()

    with open(os.path.join(os.path.dirname(__file__), 'HISTORY.rst')) as history_file:
        history = history_file.read()

requirements = ['numpy', 'scipy', 'onnxruntime']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    keywords=['neural network emulator', 'astronomy', 'galaxies', 'quasars'],
    name='fastGRAHSP',
    author="Johannes Buchner",
    author_email='johannes.buchner.acad@gmx.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    description="Fast neural network simulator for fluxes of galaxies and quasars.",
    package_data={'fastGRAHSP': ['fastGRAHSP/*.onnx', 'fastGRAHSP/*.json']},
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    packages=['fastGRAHSP'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require=dict(plot=['matplotlib', 'scipy']),
    url='https://github.com/JohannesBuchner/fastGRAHSP',
    version='1.0.3',
)

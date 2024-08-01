#!/usr/bin/env python

"""The setup script."""

import platform
import subprocess
from setuptools import setup, find_packages
from codecs import open
from os import path

subprocess.check_call(['pip', 'install', 'Cython'])

__version__ = "0.1.3"

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs

with open(
    path.join(here, "requirements.txt"), encoding="utf-8"
) as f:
    all_reqs = f.read().split("\n")

install_requires = [
    x.strip() for x in all_reqs if "git+" not in x
]
dependency_links = [
    x.strip().replace("git+", "")
    for x in all_reqs
    if x.startswith("git+")
]

setup(
    author="T. Moudiki",
    author_email='thierry.moudiki@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Unified interface for Gradient Boosted Decision Trees",
    entry_points={
        'console_scripts': [
            'unifiedbooster=unifiedbooster.cli:main',
        ],
    },
    install_requires=install_requires,
    license="BSD license",
    long_description="Unified interface for Gradient Boosted Decision Trees",
    include_package_data=True,
    keywords='unifiedbooster',
    name='unifiedbooster',
    packages=find_packages(include=['unifiedbooster', 'unifiedbooster.*']),
    test_suite='tests',
    url='https://github.com/thierrymoudiki/unifiedbooster',
    version=__version__,
    zip_safe=False,
)

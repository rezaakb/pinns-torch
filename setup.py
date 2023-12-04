#!/usr/bin/env python

from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pinnstorch",
    version="0.1.2",
    description="An implementation of PINNs in PyTorch using Lightning and Hydra.",
    author="Reza Akbarian Bafghi",
    author_email="reza.akbarianbafghi@coloardo.edu",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/rezaakb/pinns-torch",
    license='BSD-3-Clause',
    install_requires=["hydra-core", "scipy", "pyDOE", "matplotlib", "rootutils", "tqdm", "rich"],
    packages=find_packages(include='pinnstorch.*'),
    # use this to customize global commands available in the terminal after installing the package
    include_package_data=True,
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
],
)

#!/usr/bin/env python

from setuptools import find_namespace_packages, setup, find_packages

setup(
    name="pinnstorch",
    version="0.0.1",
    description="An implementation of PINNs in pytorch using Lightning and Hydra.",
    author="Reza Akbarian Bafghi",
    author_email="reza.akbarianbafghi@coloardo.edu",
    url="https://github.com/rezaakb/pinns-torch",
    install_requires=["lightning", "hydra-core", "scipy", "pyDOE", "matplotlib", "tqdm"],
    packages=find_packages(include='pinnstorch.*'),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = pinnstorch.train:main",
        ]
    },
    include_package_data=True,
)

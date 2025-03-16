#!/usr/bin/env python
"""
Setup script for the DeltaRCM package.
"""

from setuptools import setup, find_packages

setup(
    name="deltarcm",
    version="0.1.0",
    description="A reduced-complexity model for river delta formation",
    author="DeltaRCM Team",
    author_email="example@example.com",
    url="https://github.com/your-username/deltarcm",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
)
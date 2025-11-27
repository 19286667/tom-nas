#!/usr/bin/env python
"""
ToM-NAS: Theory of Mind Neural Architecture Search

A comprehensive framework for evolving neural architectures capable of
Theory of Mind reasoning.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tom-nas",
    version="1.0.0",
    author="ToM-NAS Research Team",
    description="Theory of Mind Neural Architecture Search Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/19286667/tom-nas",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tom-train=train:main",
            "tom-coevolve=train_coevolution:main",
            "tom-experiment=experiment_runner:main",
        ],
    },
)

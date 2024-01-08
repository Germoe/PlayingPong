# PlayingPong
A Reinforcement Learning Approach to Playing Pong

- [PlayingPong](#playingpong)
  - [Introduction](#introduction)
  - [Setup](#setup)
    - [Dependency management](#dependency-management)
      - [Install Conda Environment from File](#install-conda-environment-from-file)
      - [Use Conda Environment](#use-conda-environment)
      - [Optional Kernel Installation](#optional-kernel-installation)

## Introduction

## Setup

### Dependency management
This project uses `conda` for environment management and `poetry` to manage dependencies. This is useful especially in the ML context as `conda` is more powerful in terms of installing useable dependencies outside the python context (e.g. CUDA), while `poetry` allows for a very clean within-python package management. For a useful discussion on the topic see [this](https://stackoverflow.com/questions/70851048/does-it-make-sense-to-use-conda-poetry) stackoverflow post.

For example: An issue I ran into when trying to install "gymansium[atari]" was that `poetry` was unable to install the dependency `ale-py=0.8.1` as it was not available for the M1 chip on PyPI. However, `conda` with `miniforge` was able to install it from the `conda-forge` channel. For more information on M1 compatibility see [this](https://naolin.medium.com/conda-on-m1-mac-with-miniforge-bbc4e3924f2b).

#### Install Conda Environment from File

To install the conda environment from the `environment.yml` file, run the following command:
```bash
conda env create -f environment.yml
```

#### Use Conda Environment

To activate the conda environment, run the following command:
```bash
conda activate PlayingPong
```

To deactivate the conda environment, run the following command:
```bash
conda deactivate
```

*Note: Do not activate Poetry's environment. The explicit environment is solely managed by conda.*

#### Optional Kernel Installation

To use the poetry environment as a kernel in Jupyter, run the following command in an active environment shell:
```bash
ipython kernel install --name "PlayingPong" --user
```




Remove kernel with:
```bash
jupyter kernelspec uninstall playingpong
```

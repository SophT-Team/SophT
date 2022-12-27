# SophT
Scalable One-stop Platform for Hydroelastic Things (SOPHT).

This repository corresponds to the development of flow-structure
interaction simulator (2D and 3D) using immersed boundary method, while capturing Cosserat rod dynamics
using `pyelastica`.

## Installation

Below are steps of how to install `sopht`. We mainly use `poetry` to manage
the project, although most of the important commands will be provided in `Makefile`.

1. Clone!

First **create the fork repository and clone** to your local machine.

2. Virtual python workspace: `conda`, `pyenv`, or `venv`.

We recommend using python version above 3.10.

```bash
conda create --name sopht-env
conda activate sopht-env
conda install pip
```

3. Setup [`poetry`](https://python-poetry.org) and `dependencies`!

```bash
make poetry-download
make install
make pre-commit-install
```

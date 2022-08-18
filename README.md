# SophT-Examples
Scalable One-stop Platform for Hydroelastic Things (SOPHT) example cases.

This respository corresponds to the development of flow-structure
interaction cases (2D and 3D), using `sopht-backend` and `pyelastica`.

## Installation

Below are steps of how to install `sopht-examples`. We mainly use `poetry` to manage
the project, although most of the important commands will be provided in `Makefile`.

1. Clone!

First **create the fork repository and clone** to your local machine.

2. Virtual python workspace: `conda`, `pyenv`, or `venv`.

We recommend using python version above 3.8.0.

```bash
conda create --name sopht-examples-env
conda activate sopht-examples-env
conda install python==3.10
```

3. Setup [`poetry`](https://python-poetry.org) and `dependencies`!

```bash
make poetry-download
make install
make pre-commit-install
```

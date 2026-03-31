<div align='center'>
<h1> SophT </h1>

[![CI][badge-CI]][link-CI] [![DOI][badge-doi]][link-doi]
 </div>

Scalable One-stop Platform for Hydroelastic Things (SOPHT).

This repository corresponds to the development of flow-structure interaction simulator (2D and 3D) using immersed boundary method, while capturing Cosserat rod dynamics using `pyelastica`.

## Installation

A Python version of at least `3.10` is required for installation.

`sopht` can be installed in one of two ways:
1. direct installation from GitHub, and
2. cloning the repository and building it manually.

### 1. Direct installation
The first approach is straightforward with `pip`:
```sh
pip install "git+https://github.com/SophT-Team/SophT.git"
```
or optionally with additional dependencies required by the examples:
```sh
pip install "sopht[examples]@git+https://github.com/SophT-Team/SophT.git"
```

### 2. Manual installation
To install `sopht` manually, first clone the repository
```sh
git clone https://github.com/SophT-Team/SophT.git
```
The user is free to choose any Python package manager or build-system for the following steps, although we recommand using [UV](https://docs.astral.sh/uv/) as the package manager for which we have prepared a `uv.lock` file.
We provide several dependency groups and optional dependencies for various development purposes:
```sh
# Dependency groups
# uv sync --group dev
# uv sync --group lint
# uv sync --group tests

# Optional dependencies
# uv sync --extra examples

# Install all groups
uv sync
```

### 3. (Optional) Install `ffmpeg`
`sopht` examples use `ffmpeg` for data animation. To install `ffmpeg`, please visit visit their [website](https://ffmpeg.org/) for instructions.

## Citation

We ask that any publications which use SophT cite as following:

```
@software{yashraj_bhosale_2023_7658908,
  author       = {Yashraj Bhosale and
                  Arman Tekinalp and
                  Songyuan Cui and
                  Fan Kiat Chan and
                  Mattia Gazzola},
  title        = {{Scalable One-stop Platform for Hydroelastic Things
                   (SOPHT)}},
  month        = feb,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v.0.0.1},
  doi          = {10.5281/zenodo.7658908},
  url          = {https://doi.org/10.5281/zenodo.7658908}
}
```

[badge-doi]: https://zenodo.org/badge/498451510.svg
[badge-CI]: https://github.com/SophT-Team/SophT/workflows/CI/badge.svg

[link-doi]: https://zenodo.org/badge/latestdoi/498451510
[link-CI]: https://github.com/SophT-Team/SophT/actions

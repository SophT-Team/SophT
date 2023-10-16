<div align='center'>
<h1> SophT </h1>

[![CI][badge-CI]][link-CI] [![DOI][badge-doi]][link-doi]
 </div>

Scalable One-stop Platform for Hydroelastic Things (SOPHT).

This repository corresponds to the development of flow-structure
interaction simulator (2D and 3D) using immersed boundary method, while capturing Cosserat rod dynamics
using `pyelastica`.

## Installation

Below are steps of how to install `sopht`. We mainly use `poetry` to manage
the project, although most of the important commands will be provided in `Makefile`.

1. Clone!

First **create the fork repository and clone** to your local machine.

2. Set up virtual python workspace: `conda`.

```bash
conda create -n sopht-env
conda activate sopht-env
conda install python=3.10
```
3. (MacOS) System-wide installed dependencies

On MacOS (especially M-series with ARM64 architecture), we require a Homebrew installed
`fftw` library and a working `clang++` compiler with `OpenMP` support. If these requirements
are not met, we recommend
```bash
brew install llvm
brew install fftw
```

4. Set up `dependencies`!

```bash
make poetry-install
make install
make pre-commit-install
```

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

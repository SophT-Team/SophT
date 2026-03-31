# SophT Examples

This directory contains number of examples of `sopht`.
Each of the example cases is stored in separate subdirectories, containing case descriptions, run file, and all other data/script necessary to run.

## Installing Requirements
In order to run examples, you will need to install additional dependencies.

```bash
uv sync --extra examples
```

In particular, if you wish to run either of the `OctopusArmCase` (2d or 3d), Python 3.10 must be used to install the dependency `coomm` manually.

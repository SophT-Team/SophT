import ctypes
import functools
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pystencils as ps
from pystencils.jit.cpu.compiler_info import AppleClangInfo

# IMPORTANT: macOS Only
# pystencils and pyFFTW tend to load two separate copies of OpenMP
# (1) pyFFTW is bundled with one from LLVM-lite
# (2) pystencils find one that is on the system (e.g. Homebrew)
# On macOS this can create some conflicts due to cross-contamination of symbols
# The workaround here is to find the dylb for the macOS-loaded OpenMP, and explicitly
# request that pystencils kernels link to it during compilation.
#
# TODO: Check for alternatives from build options, e.g. build pyFFTW
# from system OMP and FFTW so that it does not rely on bundled OMP.


def _loaded_libomp_images() -> list[str]:
    """LLVM OpenMP runtimes already loaded in this process, in load order (macOS)."""
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libsystem._dyld_image_count.restype = ctypes.c_uint32  # noqa: SLF001
    libsystem._dyld_get_image_name.argtypes = [ctypes.c_uint32]  # noqa: SLF001
    libsystem._dyld_get_image_name.restype = ctypes.c_char_p  # noqa: SLF001
    names = (libsystem._dyld_get_image_name(i) for i in range(libsystem._dyld_image_count()))  # noqa: SLF001
    return [n.decode() for n in names if n and Path(n.decode()).name == "libomp.dylib"]


@functools.cache
def _cpu_jit_linking_libomp(libomp_path: str) -> ps.jit.CpuJit:
    """
    Create a pystencils jit config that appends a specified omp path to
    compiler flags. Cached so it is not run repeatedly.
    """

    class _SharedLibompAppleClangInfo(AppleClangInfo):
        def linker_flags(self) -> list[str]:
            flags = [f for f in super().linker_flags() if not f.endswith("libomp.dylib")]
            return [*flags, libomp_path]

    return ps.jit.CpuJit(compiler_info=_SharedLibompAppleClangInfo())


def get_pyst_dtype(real_t: type) -> str:
    """Return the pystencils data type based on real dtype."""
    if real_t == np.float32:
        return "float32"
    if real_t == np.float64:
        return "float64"
    msg = "Invalid real type"
    raise ValueError(msg)


def get_grid_info(
    grid_size: tuple[int, ...] | None,
    dimension: int,
    field_type: Literal["scalar", "vector"],
) -> str:
    """Return the pystencils field shape spec"""
    if dimension not in (2, 3):
        msg = f"dimension must be 2 or 3, but got {dimension}"
        raise ValueError(msg)

    if field_type not in ("scalar", "vector"):
        msg = f"field_type must be 'scalar' or 'vector', but got {field_type}"
        raise ValueError(msg)

    isscalar = field_type == "scalar"

    if grid_size is None:
        return f"{dimension if isscalar else dimension + 1}D"

    if len(grid_size) != dimension:
        msg = f"grid_size expected to be {dimension}D, but got {grid_size}"
        raise ValueError(msg)

    shape = grid_size if isscalar else (dimension, *grid_size)
    return ", ".join(str(i) for i in shape)


def get_pyst_kernel_config(
    real_t: type,
    num_threads: int | bool,
    iteration_slice: tuple | None = None,
) -> ps.CreateKernelConfig:
    """Returns the pystencils kernel config based on the data dtype and number of threads

    OpenMP is disabled if:
    (1) num_threads is False
    (2) num_threads < 1

    If num_threads is True, omp is left as default
    If num_threads > 1, use num_threads for this kernel
    """

    pyst_dtype = get_pyst_dtype(real_t)  # "float32" or "float64"
    kernel_config = ps.CreateKernelConfig(
        target=ps.Target.CurrentCPU,
        default_dtype=pyst_dtype,
        iteration_slice=iteration_slice,
    )

    # OpenMP options
    use_omp = num_threads is True or (isinstance(num_threads, int) and num_threads > 1)
    kernel_config.cpu.openmp.enable = use_omp
    if use_omp and num_threads is not True:
        kernel_config.cpu.openmp.schedule = "static"
        kernel_config.cpu.openmp.num_threads = num_threads

    # SIMD (Doesn't work right now and may not matter anyway)
    # kernel_config.cpu.vectorize.enable = True

    # macOS only: link pystencil kernels with whichever OpenMP that is properly loaded
    if sys.platform == "darwin" and (loaded := _loaded_libomp_images()):
        kernel_config.jit = _cpu_jit_linking_libomp(loaded[0])

    return kernel_config

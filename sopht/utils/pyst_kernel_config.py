import numpy as np
import pystencils as ps


def get_pyst_dtype(real_t: type) -> str:
    """Return the pystencils data type based on real dtype."""
    if real_t == np.float32:
        return "float32"
    if real_t == np.float64:
        return "float64"
    msg = "Invalid real type"
    raise ValueError(msg)


def get_pyst_kernel_config(
    real_t: type, num_threads: int, iteration_slice: tuple | None = None
) -> ps.CreateKernelConfig:
    """Returns the pystencils kernel config based on the data
    dtype and number of threads"""
    pyst_dtype = get_pyst_dtype(real_t) # "float32" or "float64"
    kernel_config = ps.CreateKernelConfig(
        target=ps.Target.CurrentCPU,
        default_dtype=pyst_dtype,
        iteration_slice=iteration_slice,
    )

    # OpenMP options
    kernel_config.cpu.openmp.enable = True
    kernel_config.cpu.openmp.schedule = "static"
    kernel_config.cpu.openmp.num_threads = num_threads

    # SIMD
    # kernel_config.cpu.vectorize.enable = True

    return kernel_config

import numpy as np
import pystencils as ps


def get_pyst_dtype(real_t: type) -> str:
    """Return the pystencils data type based on real dtype."""
    if real_t == np.float32:
        return "float32"
    elif real_t == np.float64:
        return "float64"
    else:
        raise ValueError("Invalid real type")


def get_pyst_kernel_config(
    real_t: type, num_threads: int, iteration_slice: tuple = None
) -> ps.CreateKernelConfig:
    """Returns the pystencils kernel config based on the data
    dtype and number of threads"""
    pyst_dtype = get_pyst_dtype(real_t)
    # TODO check out more options here!
    kernel_config = ps.CreateKernelConfig(
        data_type=pyst_dtype,
        default_number_float=pyst_dtype,
        cpu_openmp=num_threads,
        iteration_slice=iteration_slice,
    )
    return kernel_config

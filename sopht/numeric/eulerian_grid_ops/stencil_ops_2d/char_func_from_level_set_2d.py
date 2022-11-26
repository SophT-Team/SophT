"""Kernels for computing characteristic function from level set field in 2D."""
import numpy as np

import pystencils as ps

import sympy as sp

from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_2d(
    blend_width,
    real_t,
    num_threads=False,
    fixed_grid_size=False,
):
    """Level set --> Characteristic function 2D kernel generator.

    Generate function that computes characteristic function field
    from the level set field, via a smooth sine Heaviside function.
    """
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
    )
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    sine_prefactor = np.pi / blend_width

    @ps.kernel
    def _char_func_from_level_set_via_sine_heaviside_stencil_2d():
        char_func_field, level_set_field = ps.fields(
            f"char_func_field, level_set_field : {pyst_dtype}[{grid_info}]"
        )
        char_func_field[0, 0] @= (1 if (level_set_field[0, 0] > blend_width) else 0) + (
            0
            if abs(level_set_field[0, 0]) > blend_width
            else 0.5
            * (
                1
                + level_set_field[0, 0] / blend_width
                + sp.sin(sine_prefactor * level_set_field[0, 0]) / np.pi
            )
        )

    char_func_from_level_set_via_sine_heaviside_pyst_kernel_2d = ps.create_kernel(
        _char_func_from_level_set_via_sine_heaviside_stencil_2d,
        config=kernel_config,
    ).compile()
    return char_func_from_level_set_via_sine_heaviside_pyst_kernel_2d

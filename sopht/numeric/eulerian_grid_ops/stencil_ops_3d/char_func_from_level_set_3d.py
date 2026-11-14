"""Kernels for computing characteristic function from level set field in 3D."""

from collections.abc import Callable

import numpy as np
import pystencils as ps
import sympy as sp

import sopht.utils as spu


def gen_char_func_from_level_set_via_sine_heaviside_pyst_kernel_3d(
    blend_width: float,
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int, int] | None = None,
) -> Callable:
    """Level set --> Characteristic function 3D kernel generator.

    Generate function that computes characteristic function field
    from the level set field, via a smooth sine Heaviside function.
    """
    grid_info = spu.get_grid_info(grid_size, 3, "scalar")
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    sine_prefactor = np.pi / blend_width

    char_func_field, level_set_field = ps.fields(
        f"char_func_field, level_set_field : {pyst_dtype}[{grid_info}]"
    )

    _char_func_field = sp.Piecewise(
        (1, level_set_field.center > blend_width), (0, True)
    ) + sp.Piecewise(
        (0, sp.Abs(level_set_field.center) > blend_width),
        (
            0.5
            * (
                1
                + level_set_field.center / blend_width
                + sp.sin(sine_prefactor * level_set_field.center) / np.pi
            ),
            True,
        ),
    )

    return ps.create_kernel(
        ps.Assignment(char_func_field.center, _char_func_field),
        config=kernel_config,
    ).compile()

"""Kernels for penalising field boundary in 2D."""
import numpy as np

import pystencils as ps

import sympy as sp

from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_penalise_field_boundary_pyst_kernel_2d(
    width,
    dx,
    x_grid_field,
    y_grid_field,
    real_t,
    num_threads=False,
    fixed_grid_size=False,
):
    # TODO expand docs
    """2D penalise field boundary kernel generator."""
    assert width >= 0 and isinstance(width, int), "invalid zone width"

    if width == 0:
        # bypass option to prevent penalisation, done this way since by
        # default to avoid artifacts one must use penalisation...
        def penalise_field_boundary_pyst_kernel_2d(field):
            pass

    else:
        pyst_dtype = get_pyst_dtype(real_t)
        grid_info = (
            f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
        )
        x_grid_field_start = x_grid_field[0, 0]
        y_grid_field_start = y_grid_field[0, 0]
        x_grid_field_end = x_grid_field[0, -1]
        y_grid_field_end = y_grid_field[-1, 0]

        sine_prefactor = (np.pi / 2) / (width * dx)

        x_front_boundary_slice = ps.make_slice[:, :width]
        x_front_boundary_kernel_config = get_pyst_kernel_config(
            real_t,
            num_threads,
            iteration_slice=x_front_boundary_slice,
        )
        x_back_boundary_slice = ps.make_slice[:, -width:]
        x_back_boundary_kernel_config = get_pyst_kernel_config(
            real_t,
            num_threads,
            iteration_slice=x_back_boundary_slice,
        )

        @ps.kernel
        def penalise_field_x_front_boundary_stencil_2d():
            field, x_grid_field = ps.fields(
                f"field, x_grid_field : {pyst_dtype}[{grid_info}]"
            )
            field[0, 0] @= field[0, 0] * sp.sin(
                sine_prefactor * (x_grid_field[0, 0] - x_grid_field_start)
            )

        penalise_field_x_front_boundary_kernel_2d = ps.create_kernel(
            penalise_field_x_front_boundary_stencil_2d,
            config=x_front_boundary_kernel_config,
        ).compile()

        @ps.kernel
        def penalise_field_x_back_boundary_stencil_2d():
            field, x_grid_field = ps.fields(
                f"field, x_grid_field : {pyst_dtype}[{grid_info}]"
            )
            field[0, 0] @= field[0, 0] * sp.sin(
                sine_prefactor * (x_grid_field_end - x_grid_field[0, 0])
            )

        penalise_field_x_back_boundary_kernel_2d = ps.create_kernel(
            penalise_field_x_back_boundary_stencil_2d,
            config=x_back_boundary_kernel_config,
        ).compile()

        y_front_boundary_slice = ps.make_slice[:width, :]
        y_front_boundary_kernel_config = get_pyst_kernel_config(
            real_t,
            num_threads,
            iteration_slice=y_front_boundary_slice,
        )
        y_back_boundary_slice = ps.make_slice[-width:, :]
        y_back_boundary_kernel_config = get_pyst_kernel_config(
            real_t,
            num_threads,
            iteration_slice=y_back_boundary_slice,
        )

        @ps.kernel
        def penalise_field_y_front_boundary_stencil_2d():
            field, y_grid_field = ps.fields(
                f"field, y_grid_field : {pyst_dtype}[{grid_info}]"
            )
            field[0, 0] @= field[0, 0] * sp.sin(
                sine_prefactor * (y_grid_field[0, 0] - y_grid_field_start)
            )

        penalise_field_y_front_boundary_kernel_2d = ps.create_kernel(
            penalise_field_y_front_boundary_stencil_2d,
            config=y_front_boundary_kernel_config,
        ).compile()

        @ps.kernel
        def penalise_field_y_back_boundary_stencil_2d():
            field, y_grid_field = ps.fields(
                f"field, y_grid_field : {pyst_dtype}[{grid_info}]"
            )
            field[0, 0] @= field[0, 0] * sp.sin(
                sine_prefactor * (y_grid_field_end - y_grid_field[0, 0])
            )

        penalise_field_y_back_boundary_kernel_2d = ps.create_kernel(
            penalise_field_y_back_boundary_stencil_2d,
            config=y_back_boundary_kernel_config,
        ).compile()

        def penalise_field_boundary_pyst_kernel_2d(field):
            """2D penalise field boundary kernel.

            Penalises field on the boundaries in a sine wave fashion
            in the given width in X and Y direction
            field: field to be penalised
            """
            # first along X
            # these parts involve broadcasting hence couldn't be pystencilized
            field[:, :width] = field[:, (width - 1) : width]
            field[:, -width:] = field[:, -width : (-width + 1)]
            penalise_field_x_front_boundary_kernel_2d(
                field=field, x_grid_field=x_grid_field
            )
            penalise_field_x_back_boundary_kernel_2d(
                field=field, x_grid_field=x_grid_field
            )

            # then along Y
            # these parts involve broadcasting hence couldn't be pystencilized
            field[:width, :] = field[(width - 1) : width, :]
            field[-width:, :] = field[-width : (-width + 1), :]
            penalise_field_y_front_boundary_kernel_2d(
                field=field, y_grid_field=y_grid_field
            )
            penalise_field_y_back_boundary_kernel_2d(
                field=field, y_grid_field=y_grid_field
            )

    return penalise_field_boundary_pyst_kernel_2d

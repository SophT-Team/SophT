"""Kernels for computing advection flux in 2D."""
import numpy as np

import pystencils as ps

import sympy as sp

from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_advection_flux_conservative_eno3_pyst_kernel_2d(
    real_t, num_threads=False, fixed_grid_size=False
):
    # TODO expand docs
    """2D conservative ENO3 advection flux kernel generator."""
    pyst_dtype = "float32" if real_t == np.float32 else "float64"
    kernel_config = ps.CreateKernelConfig(
        data_type=pyst_dtype, default_number_float=pyst_dtype, cpu_openmp=num_threads
    )
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
    )

    @ps.kernel
    def _advection_flux_x_front_conservative_eno3_stencil_2d():
        advection_flux, field, velocity_x = ps.fields(
            f"advection_flux, field, velocity_x : {pyst_dtype}[{grid_info}]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0] @= advection_flux[0, 0] + inv_dx * (
            (
                (1 / 3) * field[0, 1] * velocity_x[0, 1]
                + (5 / 6) * field[0, 0] * velocity_x[0, 0]
                - (1 / 6) * field[0, -1] * velocity_x[0, -1]
            )
            if velocity_x[0, 0] > -velocity_x[0, 1]
            else (
                (1 / 3) * field[0, 0] * velocity_x[0, 0]
                + (5 / 6) * field[0, 1] * velocity_x[0, 1]
                - (1 / 6) * field[0, 2] * velocity_x[0, 2]
            )
        )

    _advection_flux_x_front_conservative_eno3_kernel_2d = ps.create_kernel(
        _advection_flux_x_front_conservative_eno3_stencil_2d, config=kernel_config
    ).compile()

    @ps.kernel
    def _advection_flux_x_back_conservative_eno3_stencil_2d():
        advection_flux, field, velocity_x = ps.fields(
            # f"advection_flux, field, velocity_x : {pyst_dtype}[{grid_info}]"
            # This switch is done to avoid the weird slowdown happening here.
            f"advection_flux, field, velocity_x : {pyst_dtype}[2D]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0] @= advection_flux[0, 0] - inv_dx * (
            (
                (1 / 3) * field[0, 0] * velocity_x[0, 0]
                + (5 / 6) * field[0, -1] * velocity_x[0, -1]
                - (1 / 6) * field[0, -2] * velocity_x[0, -2]
            )
            if velocity_x[0, 0] > -velocity_x[0, -1]
            else (
                (1 / 3) * field[0, -1] * velocity_x[0, -1]
                + (5 / 6) * field[0, 0] * velocity_x[0, 0]
                - (1 / 6) * field[0, 1] * velocity_x[0, 1]
            )
        )

    _advection_flux_x_back_conservative_eno3_kernel_2d = ps.create_kernel(
        _advection_flux_x_back_conservative_eno3_stencil_2d, config=kernel_config
    ).compile()

    @ps.kernel
    def _advection_flux_y_front_conservative_eno3_stencil_2d():
        advection_flux, field, velocity_y = ps.fields(
            f"advection_flux, field, velocity_y : {pyst_dtype}[{grid_info}]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0] @= advection_flux[0, 0] + inv_dx * (
            (
                (1 / 3) * field[1, 0] * velocity_y[1, 0]
                + (5 / 6) * field[0, 0] * velocity_y[0, 0]
                - (1 / 6) * field[-1, 0] * velocity_y[-1, 0]
            )
            if velocity_y[0, 0] > -velocity_y[1, 0]
            else (
                (1 / 3) * field[0, 0] * velocity_y[0, 0]
                + (5 / 6) * field[1, 0] * velocity_y[1, 0]
                - (1 / 6) * field[2, 0] * velocity_y[2, 0]
            )
        )

    _advection_flux_y_front_conservative_eno3_kernel_2d = ps.create_kernel(
        _advection_flux_y_front_conservative_eno3_stencil_2d, config=kernel_config
    ).compile()

    @ps.kernel
    def _advection_flux_y_back_conservative_eno3_stencil_2d():
        advection_flux, field, velocity_y = ps.fields(
            f"advection_flux, field, velocity_y : {pyst_dtype}[{grid_info}]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0] @= advection_flux[0, 0] - inv_dx * (
            (
                (1 / 3) * field[0, 0] * velocity_y[0, 0]
                + (5 / 6) * field[-1, 0] * velocity_y[-1, 0]
                - (1 / 6) * field[-2, 0] * velocity_y[-2, 0]
            )
            if velocity_y[0, 0] > -velocity_y[-1, 0]
            else (
                (1 / 3) * field[-1, 0] * velocity_y[-1, 0]
                + (5 / 6) * field[0, 0] * velocity_y[0, 0]
                - (1 / 6) * field[1, 0] * velocity_y[1, 0]
            )
        )

    _advection_flux_y_back_conservative_eno3_kernel_2d = ps.create_kernel(
        _advection_flux_y_back_conservative_eno3_stencil_2d, config=kernel_config
    ).compile()

    def advection_flux_conservative_eno3_pyst_kernel_2d(
        advection_flux, field, velocity, inv_dx
    ):
        # TODO expand docs
        """2D conservative ENO3 advection flux kernel."""
        _advection_flux_x_front_conservative_eno3_kernel_2d(
            advection_flux=advection_flux,
            field=field,
            velocity_x=velocity[0],
            inv_dx=inv_dx,
        )
        _advection_flux_x_back_conservative_eno3_kernel_2d(
            advection_flux=advection_flux,
            field=field,
            velocity_x=velocity[0],
            inv_dx=inv_dx,
        )
        _advection_flux_y_front_conservative_eno3_kernel_2d(
            advection_flux=advection_flux,
            field=field,
            velocity_y=velocity[1],
            inv_dx=inv_dx,
        )
        _advection_flux_y_back_conservative_eno3_kernel_2d(
            advection_flux=advection_flux,
            field=field,
            velocity_y=velocity[1],
            inv_dx=inv_dx,
        )

    return advection_flux_conservative_eno3_pyst_kernel_2d

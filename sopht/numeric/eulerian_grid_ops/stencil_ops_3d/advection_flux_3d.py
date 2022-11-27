"""Kernels for computing advection flux in 3D."""
import pystencils as ps

import sympy as sp

from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_advection_flux_conservative_eno3_pyst_kernel_3d(
    real_t, num_threads=False, fixed_grid_size=False
):
    # TODO expand docs
    """3D conservative ENO3 advection flux kernel generator."""
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if fixed_grid_size
        else "3D"
    )

    @ps.kernel
    def _advection_flux_x_front_conservative_eno3_stencil_3d():
        advection_flux, field, velocity_x = ps.fields(
            f"advection_flux, field, velocity_x : {pyst_dtype}[{grid_info}]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0, 0] @= advection_flux[0, 0, 0] + inv_dx * (
            (
                (1 / 3) * field[0, 0, 1] * velocity_x[0, 0, 1]
                + (5 / 6) * field[0, 0, 0] * velocity_x[0, 0, 0]
                - (1 / 6) * field[0, 0, -1] * velocity_x[0, 0, -1]
            )
            if velocity_x[0, 0, 0] > -velocity_x[0, 0, 1]
            else (
                (1 / 3) * field[0, 0, 0] * velocity_x[0, 0, 0]
                + (5 / 6) * field[0, 0, 1] * velocity_x[0, 0, 1]
                - (1 / 6) * field[0, 0, 2] * velocity_x[0, 0, 2]
            )
        )

    _advection_flux_x_front_conservative_eno3_kernel_3d = ps.create_kernel(
        _advection_flux_x_front_conservative_eno3_stencil_3d, config=kernel_config
    ).compile()

    @ps.kernel
    def _advection_flux_x_back_conservative_eno3_stencil_3d():
        advection_flux, field, velocity_x = ps.fields(
            # f"advection_flux, field, velocity_x : {pyst_dtype}[{grid_info}]"
            # This switch is done to avoid the weird slowdown happening here.
            f"advection_flux, field, velocity_x : {pyst_dtype}[3D]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0, 0] @= advection_flux[0, 0, 0] - inv_dx * (
            (
                (1 / 3) * field[0, 0, 0] * velocity_x[0, 0, 0]
                + (5 / 6) * field[0, 0, -1] * velocity_x[0, 0, -1]
                - (1 / 6) * field[0, 0, -2] * velocity_x[0, 0, -2]
            )
            if velocity_x[0, 0, 0] > -velocity_x[0, 0, -1]
            else (
                (1 / 3) * field[0, 0, -1] * velocity_x[0, 0, -1]
                + (5 / 6) * field[0, 0, 0] * velocity_x[0, 0, 0]
                - (1 / 6) * field[0, 0, 1] * velocity_x[0, 0, 1]
            )
        )

    _advection_flux_x_back_conservative_eno3_kernel_3d = ps.create_kernel(
        _advection_flux_x_back_conservative_eno3_stencil_3d, config=kernel_config
    ).compile()

    @ps.kernel
    def _advection_flux_y_front_conservative_eno3_stencil_3d():
        advection_flux, field, velocity_y = ps.fields(
            f"advection_flux, field, velocity_y : {pyst_dtype}[{grid_info}]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0, 0] @= advection_flux[0, 0, 0] + inv_dx * (
            (
                (1 / 3) * field[0, 1, 0] * velocity_y[0, 1, 0]
                + (5 / 6) * field[0, 0, 0] * velocity_y[0, 0, 0]
                - (1 / 6) * field[0, -1, 0] * velocity_y[0, -1, 0]
            )
            if velocity_y[0, 0, 0] > -velocity_y[0, 1, 0]
            else (
                (1 / 3) * field[0, 0, 0] * velocity_y[0, 0, 0]
                + (5 / 6) * field[0, 1, 0] * velocity_y[0, 1, 0]
                - (1 / 6) * field[0, 2, 0] * velocity_y[0, 2, 0]
            )
        )

    _advection_flux_y_front_conservative_eno3_kernel_3d = ps.create_kernel(
        _advection_flux_y_front_conservative_eno3_stencil_3d, config=kernel_config
    ).compile()

    @ps.kernel
    def _advection_flux_y_back_conservative_eno3_stencil_3d():
        advection_flux, field, velocity_y = ps.fields(
            f"advection_flux, field, velocity_y : {pyst_dtype}[{grid_info}]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0, 0] @= advection_flux[0, 0, 0] - inv_dx * (
            (
                (1 / 3) * field[0, 0, 0] * velocity_y[0, 0, 0]
                + (5 / 6) * field[0, -1, 0] * velocity_y[0, -1, 0]
                - (1 / 6) * field[0, -2, 0] * velocity_y[0, -2, 0]
            )
            if velocity_y[0, 0, 0] > -velocity_y[0, -1, 0]
            else (
                (1 / 3) * field[0, -1, 0] * velocity_y[0, -1, 0]
                + (5 / 6) * field[0, 0, 0] * velocity_y[0, 0, 0]
                - (1 / 6) * field[0, 1, 0] * velocity_y[0, 1, 0]
            )
        )

    _advection_flux_y_back_conservative_eno3_kernel_3d = ps.create_kernel(
        _advection_flux_y_back_conservative_eno3_stencil_3d, config=kernel_config
    ).compile()

    @ps.kernel
    def _advection_flux_z_front_conservative_eno3_stencil_3d():
        advection_flux, field, velocity_z = ps.fields(
            f"advection_flux, field, velocity_z : {pyst_dtype}[{grid_info}]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0, 0] @= advection_flux[0, 0, 0] + inv_dx * (
            (
                (1 / 3) * field[1, 0, 0] * velocity_z[1, 0, 0]
                + (5 / 6) * field[0, 0, 0] * velocity_z[0, 0, 0]
                - (1 / 6) * field[-1, 0, 0] * velocity_z[-1, 0, 0]
            )
            if velocity_z[0, 0, 0] > -velocity_z[1, 0, 0]
            else (
                (1 / 3) * field[0, 0, 0] * velocity_z[0, 0, 0]
                + (5 / 6) * field[1, 0, 0] * velocity_z[1, 0, 0]
                - (1 / 6) * field[2, 0, 0] * velocity_z[2, 0, 0]
            )
        )

    _advection_flux_z_front_conservative_eno3_kernel_3d = ps.create_kernel(
        _advection_flux_z_front_conservative_eno3_stencil_3d, config=kernel_config
    ).compile()

    @ps.kernel
    def _advection_flux_z_back_conservative_eno3_stencil_3d():
        advection_flux, field, velocity_z = ps.fields(
            f"advection_flux, field, velocity_z : {pyst_dtype}[{grid_info}]"
        )
        inv_dx = sp.symbols("inv_dx")
        # TODO SHOULD HAVE CODEGEN FOR THIS!
        advection_flux[0, 0, 0] @= advection_flux[0, 0, 0] - inv_dx * (
            (
                (1 / 3) * field[0, 0, 0] * velocity_z[0, 0, 0]
                + (5 / 6) * field[-1, 0, 0] * velocity_z[-1, 0, 0]
                - (1 / 6) * field[-2, 0, 0] * velocity_z[-2, 0, 0]
            )
            if velocity_z[0, 0, 0] > -velocity_z[-1, 0, 0]
            else (
                (1 / 3) * field[-1, 0, 0] * velocity_z[-1, 0, 0]
                + (5 / 6) * field[0, 0, 0] * velocity_z[0, 0, 0]
                - (1 / 6) * field[1, 0, 0] * velocity_z[1, 0, 0]
            )
        )

    _advection_flux_z_back_conservative_eno3_kernel_3d = ps.create_kernel(
        _advection_flux_z_back_conservative_eno3_stencil_3d, config=kernel_config
    ).compile()

    def advection_flux_conservative_eno3_pyst_kernel_3d(
        advection_flux, field, velocity, inv_dx
    ):
        # TODO expand docs
        """3D conservative ENO3 advection flux kernel.

        Computes 3D conservative advection flux using
        3rd order ENO stencil.
        """
        # Assumes velocity field is (3, n, n)
        _advection_flux_x_front_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_x=velocity[0],
            inv_dx=inv_dx,
        )
        _advection_flux_x_back_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_x=velocity[0],
            inv_dx=inv_dx,
        )
        _advection_flux_y_front_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_y=velocity[1],
            inv_dx=inv_dx,
        )
        _advection_flux_y_back_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_y=velocity[1],
            inv_dx=inv_dx,
        )
        _advection_flux_z_front_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_z=velocity[2],
            inv_dx=inv_dx,
        )
        _advection_flux_z_back_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_z=velocity[2],
            inv_dx=inv_dx,
        )

    return advection_flux_conservative_eno3_pyst_kernel_3d

"""Kernels for computing advection flux in 3D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.utils as spu
from typing import Callable


def gen_advection_flux_conservative_eno3_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
) -> Callable:
    # TODO expand docs
    """3D conservative ENO3 advection flux kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
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
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()

    def advection_flux_conservative_eno3_pyst_kernel_3d(
        advection_flux: np.ndarray,
        field: np.ndarray,
        velocity: np.ndarray,
        inv_dx: float,
    ) -> None:
        # TODO expand docs
        """3D conservative ENO3 advection flux kernel.

        Computes 3D conservative advection flux using
        3rd order ENO stencil.
        """
        # Assumes velocity field is (3, n, n)
        _advection_flux_x_front_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_x=velocity[x_axis_idx],
            inv_dx=inv_dx,
        )
        _advection_flux_x_back_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_x=velocity[x_axis_idx],
            inv_dx=inv_dx,
        )
        _advection_flux_y_front_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_y=velocity[y_axis_idx],
            inv_dx=inv_dx,
        )
        _advection_flux_y_back_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_y=velocity[y_axis_idx],
            inv_dx=inv_dx,
        )
        _advection_flux_z_front_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_z=velocity[z_axis_idx],
            inv_dx=inv_dx,
        )
        _advection_flux_z_back_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_z=velocity[z_axis_idx],
            inv_dx=inv_dx,
        )

    return advection_flux_conservative_eno3_pyst_kernel_3d

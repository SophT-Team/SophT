"""Kernels for computing advection flux in 3D."""

from collections.abc import Callable

import numpy as np
import pystencils as ps
import sympy as sp

import sopht.utils as spu


def gen_advection_flux_conservative_eno3_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int, int] | None = None,
) -> Callable:
    # TODO: expand docs
    """3D conservative ENO3 advection flux kernel generator"""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = spu.get_grid_info(grid_size, 3, "scalar")

    advection_flux, field, velocity_x, velocity_y, velocity_z = ps.fields(
        f"advection_flux, field, velocity_x, velocity_y, velocity_z : {pyst_dtype}[{grid_info}]"
    )
    inv_dx = sp.Symbol("inv_dx")

    def face_flux(velocity: ps.Field, axis: int, offset: int) -> sp.Expr:
        """ENO3 conservative flux at the face between cells offset and offset + 1 along axis."""

        def flux(k: int) -> sp.Expr:
            return field.center.neighbor(axis, k) * velocity.center.neighbor(axis, k)

        def vel(k: int) -> sp.Expr:
            return velocity.center.neighbor(axis, k)

        i = offset
        left_biased = (1 / 3) * flux(i + 1) + (5 / 6) * flux(i) - (1 / 6) * flux(i - 1)
        right_biased = (1 / 3) * flux(i) + (5 / 6) * flux(i + 1) - (1 / 6) * flux(i + 2)
        is_positive = sp.Piecewise((1, vel(i) > -vel(i + 1)), (0, True))
        return right_biased + (left_biased - right_biased) * is_positive

    # field is in z-y-x convention: axis 2 is x, axis 1 is y, axis 0 is z
    x_axis, y_axis, z_axis = 2, 1, 0
    flux_divergence = (
        face_flux(velocity_x, x_axis, 0)
        - face_flux(velocity_x, x_axis, -1)
        + face_flux(velocity_y, y_axis, 0)
        - face_flux(velocity_y, y_axis, -1)
        + face_flux(velocity_z, z_axis, 0)
        - face_flux(velocity_z, z_axis, -1)
    )
    eqn = ps.AddAugmentedAssignment(advection_flux.center, inv_dx * flux_divergence)
    _advection_flux_conservative_eno3_kernel_3d = ps.create_kernel(
        eqn, config=kernel_config
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
        # TODO: expand docs
        """3D conservative ENO3 advection flux kernel.

        Computes 3D conservative advection flux using
        3rd order ENO stencil.
        """
        # Assumes velocity field is (3, n, n)
        _advection_flux_conservative_eno3_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity_x=velocity[x_axis_idx],
            velocity_y=velocity[y_axis_idx],
            velocity_z=velocity[z_axis_idx],
            inv_dx=inv_dx,
        )

    return advection_flux_conservative_eno3_pyst_kernel_3d

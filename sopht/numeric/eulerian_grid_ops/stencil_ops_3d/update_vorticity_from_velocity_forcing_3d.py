"""Kernels for updating vorticity based on velocity forcing in 3D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.utils as spu
from typing import Callable


def gen_update_vorticity_from_velocity_forcing_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
) -> Callable:
    # TODO expand docs
    """Update vorticity based on velocity forcing in 3D kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _update_vorticity_from_velocity_forcing_x_comp_stencil_3d():
        (
            vorticity_field_x,
            velocity_forcing_field_y,
            velocity_forcing_field_z,
        ) = ps.fields(
            f"vorticity_field_x, velocity_forcing_field_y, "
            f"velocity_forcing_field_z : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        # curl_x = df_z / dy - df_y / dz
        vorticity_field_x[0, 0, 0] @= vorticity_field_x[0, 0, 0] + prefactor * (
            velocity_forcing_field_z[0, 1, 0]
            - velocity_forcing_field_z[0, -1, 0]
            - velocity_forcing_field_y[1, 0, 0]
            + velocity_forcing_field_y[-1, 0, 0]
        )

    _update_vorticity_from_velocity_forcing_x_comp_kernel_3d = ps.create_kernel(
        _update_vorticity_from_velocity_forcing_x_comp_stencil_3d, config=kernel_config
    ).compile()

    @ps.kernel
    def _update_vorticity_from_velocity_forcing_y_comp_stencil_3d():
        (
            vorticity_field_y,
            velocity_forcing_field_x,
            velocity_forcing_field_z,
        ) = ps.fields(
            f"vorticity_field_y, velocity_forcing_field_x,"
            f" velocity_forcing_field_z : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        # curl_y = df_x / dz - df_z / dx
        vorticity_field_y[0, 0, 0] @= vorticity_field_y[0, 0, 0] + prefactor * (
            velocity_forcing_field_x[1, 0, 0]
            - velocity_forcing_field_x[-1, 0, 0]
            - velocity_forcing_field_z[0, 0, 1]
            + velocity_forcing_field_z[0, 0, -1]
        )

    _update_vorticity_from_velocity_forcing_y_comp_kernel_3d = ps.create_kernel(
        _update_vorticity_from_velocity_forcing_y_comp_stencil_3d, config=kernel_config
    ).compile()

    @ps.kernel
    def _update_vorticity_from_velocity_forcing_z_comp_stencil_3d():
        (
            vorticity_field_z,
            velocity_forcing_field_x,
            velocity_forcing_field_y,
        ) = ps.fields(
            f"vorticity_field_z, velocity_forcing_field_x,"
            f" velocity_forcing_field_y : {pyst_dtype}[{grid_info}]"
        )
        ""
        prefactor = sp.symbols("prefactor")
        # curl_z = df_y / dx - df_x / dy
        vorticity_field_z[0, 0, 0] @= vorticity_field_z[0, 0, 0] + prefactor * (
            velocity_forcing_field_y[0, 0, 1]
            - velocity_forcing_field_y[0, 0, -1]
            - velocity_forcing_field_x[0, 1, 0]
            + velocity_forcing_field_x[0, -1, 0]
        )

    _update_vorticity_from_velocity_forcing_z_comp_kernel_3d = ps.create_kernel(
        _update_vorticity_from_velocity_forcing_z_comp_stencil_3d, config=kernel_config
    ).compile()
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()

    def update_vorticity_from_velocity_forcing_pyst_kernel_3d(
        vorticity_field: np.ndarray,
        velocity_forcing_field: np.ndarray,
        prefactor: float,
    ) -> None:
        """Kernel for updating vorticity based on velocity forcing in 3D.

        Updates vorticity_field based on velocity_forcing_field
        vorticity_field += prefactor * curl(velocity_forcing_field)
        prefactor: grid spacing factored out, along with any other multiplier
        Assumes velocity_forcing_field is (3, n, n).
        """
        _update_vorticity_from_velocity_forcing_x_comp_kernel_3d(
            vorticity_field_x=vorticity_field[x_axis_idx],
            velocity_forcing_field_y=velocity_forcing_field[y_axis_idx],
            velocity_forcing_field_z=velocity_forcing_field[z_axis_idx],
            prefactor=prefactor,
        )
        _update_vorticity_from_velocity_forcing_y_comp_kernel_3d(
            vorticity_field_y=vorticity_field[y_axis_idx],
            velocity_forcing_field_x=velocity_forcing_field[x_axis_idx],
            velocity_forcing_field_z=velocity_forcing_field[z_axis_idx],
            prefactor=prefactor,
        )
        _update_vorticity_from_velocity_forcing_z_comp_kernel_3d(
            vorticity_field_z=vorticity_field[z_axis_idx],
            velocity_forcing_field_x=velocity_forcing_field[x_axis_idx],
            velocity_forcing_field_y=velocity_forcing_field[y_axis_idx],
            prefactor=prefactor,
        )

    return update_vorticity_from_velocity_forcing_pyst_kernel_3d


def gen_update_vorticity_from_penalised_velocity_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
) -> Callable:
    # TODO expand docs
    """Update vorticity based on penalised velocity in 3D kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _update_vorticity_from_penalised_velocity_x_comp_stencil_3d():
        (vorticity_field_x, velocity_field_y, velocity_field_z,) = ps.fields(
            f"vorticity_field_x, velocity_field_y, "
            f"velocity_field_z : {pyst_dtype}[{grid_info}]"
        )
        (penalised_velocity_field_y, penalised_velocity_field_z,) = ps.fields(
            f"penalised_velocity_field_y, "
            f"penalised_velocity_field_z : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        # curl_x = df_z / dy - df_y / dz
        vorticity_field_x[0, 0, 0] @= vorticity_field_x[0, 0, 0] + prefactor * (
            penalised_velocity_field_z[0, 1, 0]
            - velocity_field_z[0, 1, 0]
            - penalised_velocity_field_z[0, -1, 0]
            + velocity_field_z[0, -1, 0]
            - penalised_velocity_field_y[1, 0, 0]
            + velocity_field_y[1, 0, 0]
            + penalised_velocity_field_y[-1, 0, 0]
            - velocity_field_y[-1, 0, 0]
        )

    _update_vorticity_from_penalised_velocity_x_comp_kernel_3d = ps.create_kernel(
        _update_vorticity_from_penalised_velocity_x_comp_stencil_3d,
        config=kernel_config,
    ).compile()

    @ps.kernel
    def _update_vorticity_from_penalised_velocity_y_comp_stencil_3d():
        (vorticity_field_y, velocity_field_x, velocity_field_z,) = ps.fields(
            f"vorticity_field_y, velocity_field_x,"
            f" velocity_field_z : {pyst_dtype}[{grid_info}]"
        )
        (penalised_velocity_field_x, penalised_velocity_field_z,) = ps.fields(
            f"penalised_velocity_field_x, "
            f"penalised_velocity_field_z : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        # curl_y = df_x / dz - df_z / dx
        vorticity_field_y[0, 0, 0] @= vorticity_field_y[0, 0, 0] + prefactor * (
            penalised_velocity_field_x[1, 0, 0]
            - velocity_field_x[1, 0, 0]
            - penalised_velocity_field_x[-1, 0, 0]
            + velocity_field_x[-1, 0, 0]
            - penalised_velocity_field_z[0, 0, 1]
            + velocity_field_z[0, 0, 1]
            + penalised_velocity_field_z[0, 0, -1]
            - velocity_field_z[0, 0, -1]
        )

    _update_vorticity_from_penalised_velocity_y_comp_kernel_3d = ps.create_kernel(
        _update_vorticity_from_penalised_velocity_y_comp_stencil_3d,
        config=kernel_config,
    ).compile()

    @ps.kernel
    def _update_vorticity_from_penalised_velocity_z_comp_stencil_3d():
        (vorticity_field_z, velocity_field_x, velocity_field_y,) = ps.fields(
            f"vorticity_field_z, velocity_field_x,"
            f" velocity_field_y : {pyst_dtype}[{grid_info}]"
        )
        (penalised_velocity_field_x, penalised_velocity_field_y,) = ps.fields(
            f"penalised_velocity_field_x, "
            f"penalised_velocity_field_y : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        # curl_z = df_y / dx - df_x / dy
        vorticity_field_z[0, 0, 0] @= vorticity_field_z[0, 0, 0] + prefactor * (
            penalised_velocity_field_y[0, 0, 1]
            - velocity_field_y[0, 0, 1]
            - penalised_velocity_field_y[0, 0, -1]
            + velocity_field_y[0, 0, -1]
            - penalised_velocity_field_x[0, 1, 0]
            + velocity_field_x[0, 1, 0]
            + penalised_velocity_field_x[0, -1, 0]
            - velocity_field_x[0, -1, 0]
        )

    _update_vorticity_from_penalised_velocity_z_comp_kernel_3d = ps.create_kernel(
        _update_vorticity_from_penalised_velocity_z_comp_stencil_3d,
        config=kernel_config,
    ).compile()
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()

    def update_vorticity_from_penalised_velocity_kernel_3d(
        vorticity_field: np.ndarray,
        penalised_velocity_field: np.ndarray,
        velocity_field: np.ndarray,
        prefactor: float,
    ) -> None:
        """Update vorticity based on penalised velocity kernel in 3D.

        Updates vorticity_field based on velocity_field and penalised_velocity_field
        vorticity_field += prefactor * curl(penalised_velocity_field - velocity_field)
        prefactor: grid spacing factored out, along with any other multiplier
        Assumes velocity_field is (3, n, n).
        """
        _update_vorticity_from_penalised_velocity_x_comp_kernel_3d(
            vorticity_field_x=vorticity_field[x_axis_idx],
            penalised_velocity_field_y=penalised_velocity_field[y_axis_idx],
            penalised_velocity_field_z=penalised_velocity_field[z_axis_idx],
            velocity_field_y=velocity_field[y_axis_idx],
            velocity_field_z=velocity_field[z_axis_idx],
            prefactor=prefactor,
        )
        _update_vorticity_from_penalised_velocity_y_comp_kernel_3d(
            vorticity_field_y=vorticity_field[y_axis_idx],
            penalised_velocity_field_x=penalised_velocity_field[x_axis_idx],
            penalised_velocity_field_z=penalised_velocity_field[z_axis_idx],
            velocity_field_x=velocity_field[x_axis_idx],
            velocity_field_z=velocity_field[z_axis_idx],
            prefactor=prefactor,
        )
        _update_vorticity_from_penalised_velocity_z_comp_kernel_3d(
            vorticity_field_z=vorticity_field[z_axis_idx],
            penalised_velocity_field_x=penalised_velocity_field[x_axis_idx],
            penalised_velocity_field_y=penalised_velocity_field[y_axis_idx],
            velocity_field_x=velocity_field[x_axis_idx],
            velocity_field_y=velocity_field[y_axis_idx],
            prefactor=prefactor,
        )

    return update_vorticity_from_penalised_velocity_kernel_3d

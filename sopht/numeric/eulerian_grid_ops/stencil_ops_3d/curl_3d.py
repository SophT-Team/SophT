"""Kernels for computing curl in 3D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu
from typing import Callable


def gen_curl_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
    reset_ghost_zone: bool = True,
) -> Callable:
    # TODO expand docs
    """3D Curl kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _curl_x_comp_stencil_3d():
        curl_x, field_y, field_z = ps.fields(
            f"curl_x, field_y, field_z : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        # curl_x = df_z / dy - df_y / dz
        curl_x[0, 0, 0] @= prefactor * (
            field_z[0, 1, 0] - field_z[0, -1, 0] - field_y[1, 0, 0] + field_y[-1, 0, 0]
        )

    _curl_x_comp_3d = ps.create_kernel(
        _curl_x_comp_stencil_3d, config=kernel_config
    ).compile()

    @ps.kernel
    def _curl_y_comp_stencil_3d():
        curl_y, field_x, field_z = ps.fields(
            f"curl_y, field_x, field_z : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        # curl_y = df_x / dz - df_z / dx
        curl_y[0, 0, 0] @= prefactor * (
            field_x[1, 0, 0] - field_x[-1, 0, 0] - field_z[0, 0, 1] + field_z[0, 0, -1]
        )

    _curl_y_comp_3d = ps.create_kernel(
        _curl_y_comp_stencil_3d, config=kernel_config
    ).compile()

    @ps.kernel
    def _curl_z_comp_stencil_3d():
        curl_z, field_x, field_y = ps.fields(
            f"curl_z, field_x, field_y : {pyst_dtype}[{grid_info}]"
        )
        ""
        prefactor = sp.symbols("prefactor")
        # curl_z = df_y / dx - df_x / dy
        curl_z[0, 0, 0] @= prefactor * (
            field_y[0, 0, 1] - field_y[0, 0, -1] - field_x[0, 1, 0] + field_x[0, -1, 0]
        )

    _curl_z_comp_3d = ps.create_kernel(
        _curl_z_comp_stencil_3d, config=kernel_config
    ).compile()
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()

    def curl_pyst_kernel_3d(
        curl: np.ndarray, field: np.ndarray, prefactor: float
    ) -> None:
        """Curl in 3D.

        Computes curl (3D vector field) essentially vector
        Laplacian for a 3D vector field
        assumes shape of fields (3, n, n, n)
        # Assumes field is (3, n, n, n) and dx = dy = dz
        """
        # curl_x = df_z / dy - df_y / dz
        _curl_x_comp_3d(
            curl_x=curl[x_axis_idx],
            field_z=field[z_axis_idx],
            field_y=field[y_axis_idx],
            prefactor=prefactor,
        )
        # curl_y = df_x / dz - df_z / dx
        _curl_y_comp_3d(
            curl_y=curl[y_axis_idx],
            field_x=field[x_axis_idx],
            field_z=field[z_axis_idx],
            prefactor=prefactor,
        )
        # curl_z = df_y / dx - df_x / dy
        _curl_z_comp_3d(
            curl_z=curl[z_axis_idx],
            field_y=field[y_axis_idx],
            field_x=field[x_axis_idx],
            prefactor=prefactor,
        )

    match reset_ghost_zone:
        case False:
            return curl_pyst_kernel_3d
        case True:
            # to set boundary zone = 0
            boundary_width = 1
            set_fixed_val_at_boundaries_3d = spne.gen_set_fixed_val_at_boundaries_pyst_kernel_3d(
                real_t=real_t,
                width=boundary_width,
                # complexity of this operation is O(N^2), hence setting serial version
                num_threads=False,
                field_type="vector",
            )

            def curl_with_ghost_zone_reset_pyst_kernel_3d(
                curl: np.ndarray, field: np.ndarray, prefactor: float
            ) -> None:
                """Curl in 3D, with resetting of ghost zone

                Computes curl (3D vector field) essentially vector
                Laplacian for a 3D vector field
                assumes shape of fields (3, n, n, n)
                # Assumes field is (3, n, n, n) and dx = dy = dz
                """
                curl_pyst_kernel_3d(curl, field, prefactor)
                # set boundary unaffected points to 0
                # TODO need one sided corrections?
                set_fixed_val_at_boundaries_3d(vector_field=curl, fixed_vals=[0, 0, 0])

            return curl_with_ghost_zone_reset_pyst_kernel_3d

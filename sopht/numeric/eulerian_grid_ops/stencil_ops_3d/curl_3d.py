"""Kernels for computing curl in 3D."""
import pystencils as ps
import sympy as sp
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_set_fixed_val_at_boundaries_pyst_kernel_3d,
)
from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_curl_pyst_kernel_3d(
    real_t, num_threads=False, fixed_grid_size=False, reset_ghost_zone=True
):
    # TODO expand docs
    """3D Curl kernel generator."""
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if fixed_grid_size
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

    def curl_pyst_kernel_3d(curl, field, prefactor):
        """Curl in 3D.

        Computes curl (3D vector field) essentially vector
        Laplacian for a 3D vector field
        assumes shape of fields (3, n, n, n)
        # Assumes field is (3, n, n, n) and dx = dy = dz
        """
        # curl_x = df_z / dy - df_y / dz
        _curl_x_comp_3d(
            curl_x=curl[0], field_z=field[2], field_y=field[1], prefactor=prefactor
        )
        # curl_y = df_x / dz - df_z / dx
        _curl_y_comp_3d(
            curl_y=curl[1], field_x=field[0], field_z=field[2], prefactor=prefactor
        )
        # curl_z = df_y / dx - df_x / dy
        _curl_z_comp_3d(
            curl_z=curl[2], field_y=field[1], field_x=field[0], prefactor=prefactor
        )

    if not reset_ghost_zone:
        return curl_pyst_kernel_3d
    else:
        # to set boundary zone = 0
        boundary_width = 1
        set_fixed_val_at_boundaries_3d = gen_set_fixed_val_at_boundaries_pyst_kernel_3d(
            real_t=real_t,
            width=boundary_width,
            # complexity of this operation is O(N^2), hence setting serial version
            num_threads=False,
            field_type="vector",
        )

        def curl_with_ghost_zone_reset_pyst_kernel_3d(curl, field, prefactor):
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

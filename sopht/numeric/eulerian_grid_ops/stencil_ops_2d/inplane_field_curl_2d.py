"""Kernels for computing curl of inplane field in 2D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.utils as spu


def gen_inplane_field_curl_pyst_kernel_2d(
    real_t, num_threads=False, fixed_grid_size=False
):
    """2D Inplane field curl kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "2D"
    )

    @ps.kernel
    def _inplane_field_curl_stencil_2d():
        curl, field_x, field_y = ps.fields(
            f"curl, field_x, field_y : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        curl[0, 0] @= (
            field_y[0, 1] - field_y[0, -1] - field_x[1, 0] + field_x[-1, 0]
        ) * prefactor

    _inplane_field_curl_pyst_kernel_2d = ps.create_kernel(
        _inplane_field_curl_stencil_2d, config=kernel_config
    ).compile()
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()

    def inplane_field_curl_pyst_kernel_2d(
        curl: np.ndarray, field: np.ndarray, prefactor: float
    ) -> None:
        """Inplane field curl in 2D.

        Computes curl of inplane 2D vector field (field_x, field_y)
        into scalar 2D outplane field (curl).
        Used for velocity ---> vorticity
        Assumes field is (2, n, n)
        """
        _inplane_field_curl_pyst_kernel_2d(
            curl=curl,
            field_x=field[x_axis_idx],
            field_y=field[y_axis_idx],
            prefactor=prefactor,
        )

    return inplane_field_curl_pyst_kernel_2d

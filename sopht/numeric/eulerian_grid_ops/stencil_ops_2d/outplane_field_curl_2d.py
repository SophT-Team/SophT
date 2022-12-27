"""Kernels for computing curl of outplane field in 2D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu
from typing import Callable


def gen_outplane_field_curl_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int] | bool = False,
    reset_ghost_zone: bool = True,
) -> Callable:
    """2D Outplane field curl kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "2D"
    )

    @ps.kernel
    def _outplane_field_curl_x_stencil_2d():
        curl_x, field = ps.fields(f"curl_x, field : {pyst_dtype}[{grid_info}]")
        prefactor = sp.symbols("prefactor")
        # curl_x = d (field) / dy
        curl_x[0, 0] @= (field[1, 0] - field[-1, 0]) * prefactor

    _outplane_field_curl_x_pyst_kernel_2d = ps.create_kernel(
        _outplane_field_curl_x_stencil_2d, config=kernel_config
    ).compile()

    @ps.kernel
    def _outplane_field_curl_y_stencil_2d():
        curl_y, field = ps.fields(f"curl_y, field : {pyst_dtype}[{grid_info}]")
        prefactor = sp.symbols("prefactor")
        # curl_y = -d (field) / dx
        curl_y[0, 0] @= (field[0, -1] - field[0, 1]) * prefactor

    _outplane_field_curl_y_pyst_kernel_2d = ps.create_kernel(
        _outplane_field_curl_y_stencil_2d, config=kernel_config
    ).compile()
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()

    def outplane_field_curl_pyst_kernel_2d(
        curl: np.ndarray, field: np.ndarray, prefactor: float
    ) -> None:
        """Outplane field curl in 2D.

        Computes curl of outplane 2D vector field (field)
        into vector 2D inplane field (curl_x, curl_y).
        Used for psi ---> velocity
        Assumes curl field is (2, n, n).
        """
        _outplane_field_curl_x_pyst_kernel_2d(
            curl_x=curl[x_axis_idx], field=field, prefactor=prefactor
        )
        _outplane_field_curl_y_pyst_kernel_2d(
            curl_y=curl[y_axis_idx], field=field, prefactor=prefactor
        )

    match reset_ghost_zone:
        case False:
            return outplane_field_curl_pyst_kernel_2d
        case True:
            # to set boundary zone = 0
            boundary_width = 1
            set_fixed_val_at_boundaries_2d = spne.gen_set_fixed_val_at_boundaries_pyst_kernel_2d(
                real_t=real_t,
                width=boundary_width,
                # complexity of this operation is O(N), hence setting serial version
                num_threads=False,
                field_type="vector",
            )

            def outplane_field_curl_with_ghost_zone_reset_pyst_kernel_2d(
                curl: np.ndarray, field: np.ndarray, prefactor: float
            ) -> None:
                """Outplane field curl in 2D, with resetting of ghost zone.

                Computes curl of outplane 2D vector field (field)
                into vector 2D inplane field (curl_x, curl_y).
                Used for psi ---> velocity
                Assumes curl field is (2, n, n).
                """
                outplane_field_curl_pyst_kernel_2d(curl, field, prefactor)

                # set boundary unaffected points to 0
                # TODO need one sided corrections?
                set_fixed_val_at_boundaries_2d(vector_field=curl, fixed_vals=[0, 0])

            return outplane_field_curl_with_ghost_zone_reset_pyst_kernel_2d

"""Kernels for computing vorticity stretching flux in 3D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu
from typing import Callable


def gen_vorticity_stretching_flux_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
) -> Callable:
    # TODO expand docs
    """3D Vorticity stretching flux kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _vorticity_stretching_flux_single_comp_stencil_3d():
        vorticity_stretching_flux_field_comp, velocity_field_comp = ps.fields(
            f"vorticity_stretching_flux_field_comp, "
            f"velocity_field_comp : {pyst_dtype}[{grid_info}]"
        )
        vorticity_field_x, vorticity_field_y, vorticity_field_z = ps.fields(
            f"vorticity_field_x, vorticity_field_y, "
            f"vorticity_field_z : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        vorticity_stretching_flux_field_comp[0, 0, 0] @= prefactor * (
            vorticity_field_x[0, 0, 0]
            * (velocity_field_comp[0, 0, 1] - velocity_field_comp[0, 0, -1])
            + vorticity_field_y[0, 0, 0]
            * (velocity_field_comp[0, 1, 0] - velocity_field_comp[0, -1, 0])
            + vorticity_field_z[0, 0, 0]
            * (velocity_field_comp[1, 0, 0] - velocity_field_comp[-1, 0, 0])
        )

    _vorticity_stretching_flux_single_comp_kernel_3d = ps.create_kernel(
        _vorticity_stretching_flux_single_comp_stencil_3d,
        config=kernel_config,
    ).compile()

    # to set boundary zone = 0
    boundary_width = 1
    set_fixed_val_at_boundaries_3d = (
        spne.gen_set_fixed_val_at_boundaries_pyst_kernel_3d(
            real_t=real_t,
            width=boundary_width,
            num_threads=num_threads,
            field_type="vector",
        )
    )
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()

    def vorticity_stretching_flux_pyst_kernel_3d(
        vorticity_stretching_flux_field: np.ndarray,
        vorticity_field: np.ndarray,
        velocity_field: np.ndarray,
        prefactor: float,
    ) -> None:
        """Vorticity stretching flux kernel in 3D.

        Computes the vorticity stretching flux in 3D, for a 3D
        vorticity_field (3, n, n, n) and velocity_field (3, n, n, n), and
        stores result in vorticity_stretching_flux_field (3, n, n, n).
        """
        _vorticity_stretching_flux_single_comp_kernel_3d(
            vorticity_stretching_flux_field_comp=vorticity_stretching_flux_field[
                x_axis_idx
            ],
            velocity_field_comp=velocity_field[x_axis_idx],
            vorticity_field_x=vorticity_field[x_axis_idx],
            vorticity_field_y=vorticity_field[y_axis_idx],
            vorticity_field_z=vorticity_field[z_axis_idx],
            prefactor=prefactor,
        )
        _vorticity_stretching_flux_single_comp_kernel_3d(
            vorticity_stretching_flux_field_comp=vorticity_stretching_flux_field[
                y_axis_idx
            ],
            velocity_field_comp=velocity_field[y_axis_idx],
            vorticity_field_x=vorticity_field[x_axis_idx],
            vorticity_field_y=vorticity_field[y_axis_idx],
            vorticity_field_z=vorticity_field[z_axis_idx],
            prefactor=prefactor,
        )
        _vorticity_stretching_flux_single_comp_kernel_3d(
            vorticity_stretching_flux_field_comp=vorticity_stretching_flux_field[
                z_axis_idx
            ],
            velocity_field_comp=velocity_field[z_axis_idx],
            vorticity_field_x=vorticity_field[x_axis_idx],
            vorticity_field_y=vorticity_field[y_axis_idx],
            vorticity_field_z=vorticity_field[z_axis_idx],
            prefactor=prefactor,
        )

        # set boundary unaffected points to 0
        # TODO need one sided corrections?
        set_fixed_val_at_boundaries_3d(
            vector_field=vorticity_stretching_flux_field, fixed_vals=[0, 0, 0]
        )

    return vorticity_stretching_flux_pyst_kernel_3d

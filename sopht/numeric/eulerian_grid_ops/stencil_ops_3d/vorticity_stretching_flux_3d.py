"""Kernels for computing vorticity stretching flux in 3D."""
import pystencils as ps
import sympy as sp
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_set_fixed_val_at_boundaries_pyst_kernel_3d,
)
from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config
from typing import Union, Tuple, Type


def gen_vorticity_stretching_flux_pyst_kernel_3d(
    real_t: Type,
    num_threads: Union[bool, int] = False,
    fixed_grid_size: Union[Tuple, int] = False,
    reset_ghost_zone: bool = True,
):
    # TODO expand docs
    """3D Vorticity stretching flux kernel generator."""
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if fixed_grid_size
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

    def vorticity_stretching_flux_pyst_kernel_3d(
        vorticity_stretching_flux_field, vorticity_field, velocity_field, prefactor
    ):
        """Vorticity stretching flux kernel in 3D.

        Computes the vorticity stretching flux in 3D, for a 3D
        vorticity_field (3, n, n, n) and velocity_field (3, n, n, n), and
        stores result in vorticity_stretching_flux_field (3, n, n, n).
        """
        _vorticity_stretching_flux_single_comp_kernel_3d(
            vorticity_stretching_flux_field_comp=vorticity_stretching_flux_field[0],
            velocity_field_comp=velocity_field[0],
            vorticity_field_x=vorticity_field[0],
            vorticity_field_y=vorticity_field[1],
            vorticity_field_z=vorticity_field[2],
            prefactor=prefactor,
        )
        _vorticity_stretching_flux_single_comp_kernel_3d(
            vorticity_stretching_flux_field_comp=vorticity_stretching_flux_field[1],
            velocity_field_comp=velocity_field[1],
            vorticity_field_x=vorticity_field[0],
            vorticity_field_y=vorticity_field[1],
            vorticity_field_z=vorticity_field[2],
            prefactor=prefactor,
        )
        _vorticity_stretching_flux_single_comp_kernel_3d(
            vorticity_stretching_flux_field_comp=vorticity_stretching_flux_field[2],
            velocity_field_comp=velocity_field[2],
            vorticity_field_x=vorticity_field[0],
            vorticity_field_y=vorticity_field[1],
            vorticity_field_z=vorticity_field[2],
            prefactor=prefactor,
        )

    if not reset_ghost_zone:
        return vorticity_stretching_flux_pyst_kernel_3d
    else:
        # to set boundary zone = 0
        boundary_width = 1
        set_fixed_val_at_boundaries_3d = gen_set_fixed_val_at_boundaries_pyst_kernel_3d(
            real_t=real_t,
            width=boundary_width,
            num_threads=num_threads,
            field_type="vector",
        )

        def vorticity_stretching_flux_with_ghost_zone_reset_pyst_kernel_3d(
            vorticity_stretching_flux_field, vorticity_field, velocity_field, prefactor
        ):
            vorticity_stretching_flux_pyst_kernel_3d(
                vorticity_stretching_flux_field=vorticity_stretching_flux_field,
                vorticity_field=vorticity_field,
                velocity_field=velocity_field,
                prefactor=prefactor,
            )
            # set boundary unaffected points to 0
            # TODO need one sided corrections?
            set_fixed_val_at_boundaries_3d(
                vector_field=vorticity_stretching_flux_field, fixed_vals=[0, 0, 0]
            )

        return vorticity_stretching_flux_with_ghost_zone_reset_pyst_kernel_3d

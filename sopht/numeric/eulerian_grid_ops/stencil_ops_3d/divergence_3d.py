"""Kernels for computing divergence in 3D."""
import pystencils as ps
import sympy as sp
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_set_fixed_val_at_boundaries_pyst_kernel_3d,
)
from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_divergence_pyst_kernel_3d(
    real_t, num_threads=False, fixed_grid_size=False, reset_ghost_zone=True
):
    # TODO expand docs
    """3D divergence kernel generator."""
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if fixed_grid_size
        else "3D"
    )

    @ps.kernel
    def _divergence_stencil_3d():
        divergence, field_x, field_y, field_z = ps.fields(
            f"divergence, field_x, field_y, field_z : {pyst_dtype}[{grid_info}]"
        )
        inv_dx = sp.symbols("inv_dx")
        divergence[0, 0, 0] @= (
            0.5
            * inv_dx
            * (
                field_x[0, 0, 1]
                - field_x[0, 0, -1]
                + field_y[0, 1, 0]
                - field_y[0, -1, 0]
                + field_z[1, 0, 0]
                - field_z[-1, 0, 0]
            )
        )

    _divergence_kernel_3d = ps.create_kernel(
        _divergence_stencil_3d, config=kernel_config
    ).compile()

    def divergence_pyst_kernel_3d(divergence, field, inv_dx):
        """Divergence in 3D.

        Computes divergence (3D scalar field) for a 3D vector field
        Assumes field is (3, n, n, n) and dx = dy = dz
        """
        _divergence_kernel_3d(
            divergence=divergence,
            field_x=field[0],
            field_y=field[1],
            field_z=field[2],
            inv_dx=inv_dx,
        )

    if not reset_ghost_zone:
        return divergence_pyst_kernel_3d
    else:
        # to set boundary zone = 0
        boundary_width = 1
        set_fixed_val_at_boundaries_3d = gen_set_fixed_val_at_boundaries_pyst_kernel_3d(
            real_t=real_t,
            width=boundary_width,
            # complexity of this operation is O(N^2), hence setting serial version
            num_threads=False,
            field_type="scalar",
        )

        def divergence_with_ghost_zone_reset_pyst_kernel_3d(divergence, field, inv_dx):
            """Divergence in 3D, with resetting of ghost zone

            Computes divergence (3D scalar field) for a 3D vector field
            Assumes field is (3, n, n, n) and dx = dy = dz
            """
            divergence_pyst_kernel_3d(divergence, field, inv_dx)

            # set boundary unaffected points to 0
            # TODO need one sided corrections?
            set_fixed_val_at_boundaries_3d(field=divergence, fixed_val=0)

        return divergence_with_ghost_zone_reset_pyst_kernel_3d

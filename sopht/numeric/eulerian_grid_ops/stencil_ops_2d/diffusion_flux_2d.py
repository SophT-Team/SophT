"""Kernels for computing diffusion flux in 2D."""
import pystencils as ps
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_set_fixed_val_at_boundaries_pyst_kernel_2d,
)
from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config
import sympy as sp


def gen_diffusion_flux_pyst_kernel_2d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    reset_ghost_zone=True,
):
    # TODO expand docs
    """2D Diffusion flux kernel generator."""
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
    )

    @ps.kernel
    def _diffusion_stencil_2d():
        diffusion_flux, field = ps.fields(
            f"diffusion_flux, field : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        diffusion_flux[0, 0] @= prefactor * (
            field[1, 0] + field[-1, 0] + field[0, 1] + field[0, -1] - 4 * field[0, 0]
        )

    diffusion_flux_kernel_2d = ps.create_kernel(
        _diffusion_stencil_2d, config=kernel_config
    ).compile()

    if not reset_ghost_zone:
        return diffusion_flux_kernel_2d
    else:
        # to set boundary zone = 0
        boundary_width = 1
        set_fixed_val_at_boundaries_2d = gen_set_fixed_val_at_boundaries_pyst_kernel_2d(
            real_t=real_t,
            width=boundary_width,
            # complexity of this operation is O(N), hence setting serial version
            num_threads=False,
            field_type="scalar",
        )

        def diffusion_flux_with_ghost_zone_reset_pyst_kernel_2d(
            diffusion_flux, field, prefactor
        ):
            """Diffusion flux in 2D, with resetting of ghost zone.

            Computes diffusion flux of 2D scalar field (field)
            into scalar 2D field (diffusion_flux).
            """
            diffusion_flux_kernel_2d(
                diffusion_flux=diffusion_flux, field=field, prefactor=prefactor
            )

            # set boundary unaffected points to 0
            # TODO need one sided corrections?
            set_fixed_val_at_boundaries_2d(field=diffusion_flux, fixed_val=0)

        return diffusion_flux_with_ghost_zone_reset_pyst_kernel_2d

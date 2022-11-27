"""Kernels for computing diffusion flux in 3D."""
import pystencils as ps
import sympy as sp
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_set_fixed_val_at_boundaries_pyst_kernel_3d,
)
from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_diffusion_flux_pyst_kernel_3d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
    reset_ghost_zone=True,
):
    # TODO expand docs
    """3D Diffusion flux kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if fixed_grid_size
        else "3D"
    )

    @ps.kernel
    def _diffusion_stencil_3d():
        diffusion_flux, field = ps.fields(
            f"diffusion_flux, field : {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        diffusion_flux[0, 0, 0] @= prefactor * (
            field[1, 0, 0]
            + field[-1, 0, 0]
            + field[0, 1, 0]
            + field[0, -1, 0]
            + field[0, 0, 1]
            + field[0, 0, -1]
            - 6 * field[0, 0, 0]
        )

    diffusion_kernel_3d = ps.create_kernel(
        _diffusion_stencil_3d, config=kernel_config
    ).compile()
    if not reset_ghost_zone:
        diffusion_flux_pyst_kernel_3d = diffusion_kernel_3d
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

        def diffusion_flux_with_ghost_zone_reset_pyst_kernel_3d(
            diffusion_flux, field, prefactor
        ):
            """Diffusion flux in 3D, with resetting of ghost zone.

            Computes diffusion flux of 3D scalar field (field)
            into scalar 3D field (diffusion_flux).
            """
            diffusion_kernel_3d(
                diffusion_flux=diffusion_flux, field=field, prefactor=prefactor
            )

            # set boundary unaffected points to 0
            # TODO need one sided corrections?
            set_fixed_val_at_boundaries_3d(field=diffusion_flux, fixed_val=0)

        diffusion_flux_pyst_kernel_3d = (
            diffusion_flux_with_ghost_zone_reset_pyst_kernel_3d
        )

    if field_type == "scalar":
        return diffusion_flux_pyst_kernel_3d
    elif field_type == "vector":

        def vector_field_diffusion_flux_pyst_kernel_3d(
            vector_field_diffusion_flux, vector_field, prefactor
        ):
            """Vector field diffusion flux in 3D.

            Computes diffusion flux (3D vector field) essentially vector
            Laplacian for a 3D vector field
            assumes shape of fields (3, n, n, n)
            """
            diffusion_flux_pyst_kernel_3d(
                diffusion_flux=vector_field_diffusion_flux[0],
                field=vector_field[0],
                prefactor=prefactor,
            )
            diffusion_flux_pyst_kernel_3d(
                diffusion_flux=vector_field_diffusion_flux[1],
                field=vector_field[1],
                prefactor=prefactor,
            )
            diffusion_flux_pyst_kernel_3d(
                diffusion_flux=vector_field_diffusion_flux[2],
                field=vector_field[2],
                prefactor=prefactor,
            )

        return vector_field_diffusion_flux_pyst_kernel_3d

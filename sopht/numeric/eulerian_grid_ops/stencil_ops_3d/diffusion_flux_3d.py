"""Kernels for computing diffusion flux in 3D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu
from typing import Callable, Literal


def gen_diffusion_flux_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
    reset_ghost_zone: bool = True,
) -> Callable:
    # TODO expand docs
    """3D Diffusion flux kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
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
    match reset_ghost_zone:
        case False:
            diffusion_flux_pyst_kernel_3d = diffusion_kernel_3d
        case _:  # True
            # to set boundary zone = 0
            boundary_width = 1
            set_fixed_val_at_boundaries_3d = spne.gen_set_fixed_val_at_boundaries_pyst_kernel_3d(
                real_t=real_t,
                width=boundary_width,
                # complexity of this operation is O(N^2), hence setting serial version
                num_threads=False,
                field_type="scalar",
            )

            def diffusion_flux_with_ghost_zone_reset_pyst_kernel_3d(
                diffusion_flux: np.ndarray, field: np.ndarray, prefactor: float
            ) -> None:
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

    match field_type:
        case "scalar":
            return diffusion_flux_pyst_kernel_3d
        case "vector":
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()
            z_axis_idx = spu.VectorField.z_axis_idx()

            def vector_field_diffusion_flux_pyst_kernel_3d(
                vector_field_diffusion_flux: np.ndarray,
                vector_field: np.ndarray,
                prefactor: float,
            ) -> None:
                """Vector field diffusion flux in 3D.

                Computes diffusion flux (3D vector field) essentially vector
                Laplacian for a 3D vector field
                assumes shape of fields (3, n, n, n)
                """
                diffusion_flux_pyst_kernel_3d(
                    diffusion_flux=vector_field_diffusion_flux[x_axis_idx],
                    field=vector_field[x_axis_idx],
                    prefactor=prefactor,
                )
                diffusion_flux_pyst_kernel_3d(
                    diffusion_flux=vector_field_diffusion_flux[y_axis_idx],
                    field=vector_field[y_axis_idx],
                    prefactor=prefactor,
                )
                diffusion_flux_pyst_kernel_3d(
                    diffusion_flux=vector_field_diffusion_flux[z_axis_idx],
                    field=vector_field[z_axis_idx],
                    prefactor=prefactor,
                )

            return vector_field_diffusion_flux_pyst_kernel_3d
        case _:
            raise ValueError("Invalid field type")

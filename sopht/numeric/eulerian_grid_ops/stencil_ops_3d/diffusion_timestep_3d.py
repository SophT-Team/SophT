"""Kernels for performing diffusion timestep in 3D."""
import numpy as np
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu
from typing import Callable, Literal


def gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO expand docs
    """3D Diffusion euler forward timestep kernel generator."""
    elementwise_sum_pyst_kernel_3d = spne.gen_elementwise_sum_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    diffusion_flux_kernel_3d = spne.gen_diffusion_flux_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )

    def diffusion_timestep_euler_forward_pyst_kernel_3d(
        field: np.ndarray, diffusion_flux: np.ndarray, nu_dt_by_dx2: float
    ) -> None:
        """3D Diffusion Euler forward timestep (scalar field).

        Performs an inplace diffusion timestep in 3D using Euler forward,
        for a 3D field (n, n, n).
        """
        diffusion_flux_kernel_3d(
            diffusion_flux=diffusion_flux,
            field=field,
            prefactor=nu_dt_by_dx2,
        )
        elementwise_sum_pyst_kernel_3d(
            sum_field=field, field_1=field, field_2=diffusion_flux
        )

    match field_type:
        case "scalar":
            return diffusion_timestep_euler_forward_pyst_kernel_3d
        case "vector":
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()
            z_axis_idx = spu.VectorField.z_axis_idx()

            def vector_field_diffusion_timestep_euler_forward_pyst_kernel_3d(
                vector_field: np.ndarray,
                diffusion_flux: np.ndarray,
                nu_dt_by_dx2: float,
            ) -> None:
                """3D Diffusion Euler forward timestep (vector field).

                Performs an inplace diffusion timestep in 3D using Euler forward,
                for a 3D vector field (3, n, n, n).
                """
                diffusion_timestep_euler_forward_pyst_kernel_3d(
                    field=vector_field[x_axis_idx],
                    diffusion_flux=diffusion_flux,
                    nu_dt_by_dx2=nu_dt_by_dx2,
                )
                diffusion_timestep_euler_forward_pyst_kernel_3d(
                    field=vector_field[y_axis_idx],
                    diffusion_flux=diffusion_flux,
                    nu_dt_by_dx2=nu_dt_by_dx2,
                )
                diffusion_timestep_euler_forward_pyst_kernel_3d(
                    field=vector_field[z_axis_idx],
                    diffusion_flux=diffusion_flux,
                    nu_dt_by_dx2=nu_dt_by_dx2,
                )

            return vector_field_diffusion_timestep_euler_forward_pyst_kernel_3d

"""Kernels for performing diffusion timestep in 2D."""
import numpy as np

import sopht.numeric.eulerian_grid_ops as spne
from typing import Callable


def gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int] | bool = False,
) -> Callable:
    # TODO expand docs
    """2D Diffusion Euler forward timestep generator."""
    elementwise_sum_pyst_kernel_2d = spne.gen_elementwise_sum_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    diffusion_flux_kernel_2d = spne.gen_diffusion_flux_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )

    def diffusion_timestep_euler_forward_pyst_kernel_2d(
        field: np.ndarray, diffusion_flux: np.ndarray, nu_dt_by_dx2: float
    ) -> None:
        """2D Diffusion Euler forward timestep.

        Performs an inplace diffusion timestep in 2D using Euler forward,
        for a 2D field (n, n).
        """
        diffusion_flux_kernel_2d(
            diffusion_flux=diffusion_flux,
            field=field,
            prefactor=nu_dt_by_dx2,
        )
        elementwise_sum_pyst_kernel_2d(
            sum_field=field, field_1=field, field_2=diffusion_flux
        )

    return diffusion_timestep_euler_forward_pyst_kernel_2d

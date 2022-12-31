"""Kernels for performing advection timestep in 2D."""
import numpy as np

import sopht.numeric.eulerian_grid_ops as spne
from typing import Callable


def gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int] | bool = False,
) -> Callable:
    # TODO expand docs
    """2D Advection (ENO3 stencil) Euler forward timestep generator."""
    elementwise_sum_pyst_kernel_2d = spne.gen_elementwise_sum_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    set_fixed_val_pyst_kernel_2d = spne.gen_set_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    advection_flux_conservative_eno3_pyst_kernel_2d = (
        spne.gen_advection_flux_conservative_eno3_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=fixed_grid_size,
            num_threads=num_threads,
        )
    )

    def advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
        field: np.ndarray,
        advection_flux: np.ndarray,
        velocity: np.ndarray,
        dt_by_dx: float,
    ) -> None:
        """2D Advection (ENO3 stencil) Euler forward timestep.

        Performs an inplace advection timestep (using ENO3 stencil)
        in 2D using Euler forward, for a 2D field (n, n).
        """
        set_fixed_val_pyst_kernel_2d(field=advection_flux, fixed_val=0)
        advection_flux_conservative_eno3_pyst_kernel_2d(
            advection_flux=advection_flux,
            field=field,
            velocity=velocity,
            inv_dx=-dt_by_dx,
        )
        elementwise_sum_pyst_kernel_2d(
            sum_field=field, field_1=field, field_2=advection_flux
        )

    return advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d

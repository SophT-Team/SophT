"""Kernels for performing diffusion timestep in 2D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.diffusion_flux_2d import (
    gen_diffusion_flux_pyst_kernel_2d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_elementwise_sum_pyst_kernel_2d,
)


def gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
):
    # TODO expand docs
    """2D Diffusion Euler forward timestep generator."""
    elementwise_sum_pyst_kernel_2d = gen_elementwise_sum_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    diffusion_flux_kernel_2d = gen_diffusion_flux_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )

    def diffusion_timestep_euler_forward_pyst_kernel_2d(
        field, diffusion_flux, nu_dt_by_dx2
    ):
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

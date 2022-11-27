"""Kernels for performing advection timestep in 2D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.advection_flux_2d import (
    gen_advection_flux_conservative_eno3_pyst_kernel_2d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_elementwise_sum_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
)


def gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
    real_t, num_threads=False, fixed_grid_size=False
):
    # TODO expand docs
    """2D Advection (ENO3 stencil) Euler forward timestep generator."""
    elementwise_sum_pyst_kernel_2d = gen_elementwise_sum_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    set_fixed_val_pyst_kernel_2d = gen_set_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    advection_flux_conservative_eno3_pyst_kernel_2d = (
        gen_advection_flux_conservative_eno3_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=fixed_grid_size,
            num_threads=num_threads,
        )
    )

    def advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
        field, advection_flux, velocity, dt_by_dx
    ):
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

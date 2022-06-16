from sopht.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d,
    gen_diffusion_timestep_euler_forward_pyst_kernel_3d,
)


def gen_advection_diffusion_timestep_kernel_3d(
    real_t, grid_size, dx, nu, num_threads, field_type="scalar"
):
    """generates kernel for advection-diffusion timestep in 3D."""
    # note currently only has base Euler Forward timestepper version.

    diffusion_timestep = gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=grid_size,
        num_threads=num_threads,
        field_type=field_type,
    )
    advection_timestep = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=grid_size,
            num_threads=num_threads,
            field_type=field_type,
        )
    )

    def advection_and_diffusion_timestep_kernel_3d(
        field, velocity_field, flux_buffer, dt
    ):
        advection_timestep(
            field,
            advection_flux=flux_buffer,
            velocity=velocity_field,
            dt_by_dx=(dt / dx),
        )
        diffusion_timestep(
            field,
            diffusion_flux=flux_buffer,
            nu_dt_by_dx2=(nu * dt / dx / dx),
        )

    return advection_and_diffusion_timestep_kernel_3d

from sopht.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d,
    gen_diffusion_timestep_euler_forward_pyst_kernel_2d,
    gen_outplane_field_curl_pyst_kernel_2d,
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d,
    UnboundedPoissonSolverPYFFTW2D,
)


def gen_compute_velocity_from_vorticity_kernel_2d(real_t, grid_size, dx, num_threads):
    """generates kernel that computes velocity from vorticity."""

    # compile kernels
    unbounded_poisson_solver = UnboundedPoissonSolverPYFFTW2D(
        grid_size_y=grid_size[0],
        grid_size_x=grid_size[1],
        dx=dx,
        real_t=real_t,
        num_threads=num_threads,
    )
    unbounded_poisson_solve = unbounded_poisson_solver.solve
    outplane_field_curl = gen_outplane_field_curl_pyst_kernel_2d(
        real_t=real_t,
        num_threads=num_threads,
        fixed_grid_size=grid_size,
    )

    def compute_velocity_from_vorticity_kernel_2d(
        velocity_field,
        vorticity_field,
        stream_func_field,
    ):
        unbounded_poisson_solve(
            solution_field=stream_func_field, rhs_field=vorticity_field
        )
        outplane_field_curl(
            curl=velocity_field, field=stream_func_field, prefactor=real_t(0.5 / dx)
        )

    return compute_velocity_from_vorticity_kernel_2d


def gen_advection_diffusion_timestep_kernel_2d(real_t, grid_size, dx, nu, num_threads):
    """generates kernel for advection-diffusion timestep in 2D."""
    # note currently only has base Euler Forward timestepper version.

    diffusion_timestep = gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=grid_size,
        num_threads=num_threads,
    )
    advection_timestep = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=grid_size,
            num_threads=num_threads,
        )
    )

    def advection_and_diffusion_timestep_kernel_2d(
        field, velocity_field, flux_buffer, dt
    ):
        advection_timestep(
            field=field,
            advection_flux=flux_buffer,
            velocity=velocity_field,
            dt_by_dx=real_t(dt / dx),
        )
        diffusion_timestep(
            field=field,
            diffusion_flux=flux_buffer,
            nu_dt_by_dx2=real_t(nu * dt / dx / dx),
        )

    return advection_and_diffusion_timestep_kernel_2d


def gen_advection_diffusion_with_forcing_timestep_kernel_2d(
    real_t, grid_size, dx, nu, num_threads
):
    """generates kernel for advection-diffusion with velocity forcing timestep in 2D."""
    # note currently only has base Euler Forward timestepper version.

    diffusion_timestep = gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=grid_size,
        num_threads=num_threads,
    )
    advection_timestep = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=grid_size,
            num_threads=num_threads,
        )
    )
    update_vorticity_from_velocity_forcing = (
        gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=grid_size,
            num_threads=num_threads,
        )
    )

    def advection_and_diffusion_with_forcing_timestep_kernel_2d(
        eul_grid_forcing_field,
        field,
        velocity_field,
        flux_buffer,
        dt,
        forcing_prefactor,
    ):
        advection_timestep(
            field=field,
            advection_flux=flux_buffer,
            velocity=velocity_field,
            dt_by_dx=real_t(dt / dx),
        )
        update_vorticity_from_velocity_forcing(
            vorticity_field=field,
            velocity_forcing_field=eul_grid_forcing_field,
            prefactor=real_t(forcing_prefactor * 0.5 / dx),
        )
        diffusion_timestep(
            field=field,
            diffusion_flux=flux_buffer,
            nu_dt_by_dx2=real_t(nu * dt / dx / dx),
        )

    return advection_and_diffusion_with_forcing_timestep_kernel_2d

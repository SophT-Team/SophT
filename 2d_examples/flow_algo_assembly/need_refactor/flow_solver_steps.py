from sopht.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d,
    gen_diffusion_timestep_euler_forward_pyst_kernel_2d,
    gen_penalise_field_boundary_pyst_kernel_2d,
    gen_outplane_field_curl_pyst_kernel_2d,
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d,
    UnboundedPoissonSolverPYFFTW2D,
)


def gen_compute_velocity_from_vorticity_kernel_2d(real_t, grid_size, dx, num_threads):
    """Generate kernel that computes velocity from vorticity."""

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


def gen_advection_diffusion_euler_forward_timestep_kernel_2d(
    real_t,
    grid_size,
    dx,
    nu,
    num_threads,
    **kwargs,
):
    """Generates kernel for advection-diffusion Euler forward timestep in 2D."""

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

    def advection_and_diffusion_euler_forward_timestep_kernel_2d(
        field, velocity_field, flux_buffer, dt, **kwargs
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

    return advection_and_diffusion_euler_forward_timestep_kernel_2d


def pre_append_update_vorticity_from_velocity_forcing_2d(kernel_generator):
    def modified_kernel_generator(*args, **kwargs):
        kernel = kernel_generator(*args, **kwargs)
        update_vorticity_from_velocity_forcing = (
            gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d(
                real_t=kwargs["real_t"],
                fixed_grid_size=kwargs["grid_size"],
                num_threads=kwargs["num_threads"],
            )
        )
        dx = kwargs["dx"]

        def modified_kernel(*args, **kwargs):
            update_vorticity_from_velocity_forcing(
                vorticity_field=kwargs["field"],
                velocity_forcing_field=kwargs["eul_grid_forcing_field"],
                prefactor=kwargs["forcing_prefactor"] / (2 * dx),
            )
            kernel(*args, **kwargs)

        return modified_kernel

    return modified_kernel_generator


def post_append_compute_velocity_from_vorticity_2d(kernel_generator):
    def modified_kernel_generator(*args, **kwargs):
        kernel = kernel_generator(*args, **kwargs)
        compute_velocity_from_vorticity = gen_compute_velocity_from_vorticity_kernel_2d(
            real_t=kwargs["real_t"],
            grid_size=kwargs["grid_size"],
            dx=kwargs["dx"],
            num_threads=kwargs["num_threads"],
        )

        def modified_kernel(*args, **kwargs):
            kernel(*args, **kwargs)
            compute_velocity_from_vorticity(
                velocity_field=kwargs["velocity_field"],
                vorticity_field=kwargs["vorticity_field"],
                stream_func_field=kwargs["stream_func_field"],
            )

        return modified_kernel

    return modified_kernel_generator


def post_append_penalise_field_towards_boundary_2d(kernel_generator):
    def modified_kernel_generator(*args, **kwargs):
        kernel = kernel_generator(*args, **kwargs)
        penalise_field_towards_boundary = gen_penalise_field_boundary_pyst_kernel_2d(
            width=kwargs["penalty_zone_width"],
            dx=kwargs["dx"],
            x_grid_field=kwargs["x_grid"],
            y_grid_field=kwargs["y_grid"],
            real_t=kwargs["real_t"],
            num_threads=kwargs["num_threads"],
            fixed_grid_size=kwargs["grid_size"],
        )

        def modified_kernel(*args, **kwargs):
            kernel(*args, **kwargs)
            penalise_field_towards_boundary(field=kwargs["field_to_penalise"])

        return modified_kernel

    return modified_kernel_generator


def gen_advection_diffusion_euler_forward_timestep_with_forcing_kernel_2d(
    real_t,
    dx,
    nu,
    grid_size,
    num_threads,
    **kwargs,
):
    """Generate advection-diffusion Euler forward timestep with forcing kernel in 2D."""
    kernel_generator = pre_append_update_vorticity_from_velocity_forcing_2d(
        gen_advection_diffusion_euler_forward_timestep_kernel_2d
    )
    return kernel_generator(
        real_t=real_t, dx=dx, nu=nu, grid_size=grid_size, num_threads=num_threads
    )


def gen_full_flow_timestep_kernel_2d(real_t, dx, nu, grid_size, num_threads):
    """
    Generates kernel for full flow timestep
    (advection-diffusion, followed by velocity recovery)
    in 2D.
    """
    kernel_generator = post_append_compute_velocity_from_vorticity_2d(
        gen_advection_diffusion_euler_forward_timestep_kernel_2d
    )
    return kernel_generator(
        real_t=real_t, dx=dx, nu=nu, grid_size=grid_size, num_threads=num_threads
    )


def gen_full_flow_timestep_with_forcing_kernel_2d(
    real_t, dx, nu, grid_size, num_threads
):
    """
    Generate kernel for full flow timestep
    (advection-diffusion with velocity forcing, followed by velocity recovery)
    in 2D.
    """
    kernel_generator = post_append_compute_velocity_from_vorticity_2d(
        gen_advection_diffusion_euler_forward_timestep_with_forcing_kernel_2d
    )
    return kernel_generator(
        real_t=real_t, dx=dx, nu=nu, grid_size=grid_size, num_threads=num_threads
    )


def gen_full_flow_timestep_with_forcing_and_boundary_penalisation_kernel_2d(
    real_t, dx, nu, grid_size, num_threads, penalty_zone_width, x_grid, y_grid
):
    """
    Generate kernel for full flow timestep
    (advection-diffusion with velocity forcing, followed by vorticity boundary
    penalisation and velocity recovery)
    in 2D.
    """
    kernel_generator = post_append_compute_velocity_from_vorticity_2d(
        post_append_penalise_field_towards_boundary_2d(
            gen_advection_diffusion_euler_forward_timestep_with_forcing_kernel_2d
        )
    )
    return kernel_generator(
        real_t=real_t,
        dx=dx,
        nu=nu,
        grid_size=grid_size,
        num_threads=num_threads,
        penalty_zone_width=penalty_zone_width,
        x_grid=x_grid,
        y_grid=y_grid,
    )

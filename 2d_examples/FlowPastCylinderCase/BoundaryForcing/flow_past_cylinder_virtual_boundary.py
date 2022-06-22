from flow_algo_assembly.flow_solver_steps import (
    gen_compute_velocity_from_vorticity_kernel_2d,
    gen_advection_diffusion_with_forcing_timestep_kernel_2d,
)
from flow_algo_assembly.timestep_limits import compute_advection_diffusion_timestep

import numpy as np

import os

from plot_utils.lab_cmap import lab_cmap

from sopht.numeric.eulerian_grid_ops import (
    gen_add_fixed_val_pyst_kernel_2d,
    gen_penalise_field_boundary_pyst_kernel_2d,
)
from sopht.numeric.immersed_boundary_ops import VirtualBoundaryForcing
from sopht.utils.precision import get_real_t


def flow_past_cylinder_boundary_forcing_case(
    grid_size_x,
    grid_size_y,
    num_threads=4,
    precision="single",
    save_snap=False,
    save_diagnostic=False,
):
    """
    This example considers a simple flow past cylinder using immersed
    boundary forcing.
    """
    real_t = get_real_t(precision)
    # Initialize 2D domain
    grid_size_y_by_x = grid_size_y / grid_size_x
    dx = real_t(1.0 / grid_size_x)
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, grid_size_x).astype(real_t)
    y = np.linspace(
        eul_grid_shift, grid_size_y_by_x - eul_grid_shift, grid_size_y
    ).astype(real_t)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flow parameters
    U_inf = real_t(1.0)
    cyl_radius = real_t(0.03)
    Re = 200
    nu = cyl_radius * U_inf / Re

    # Initialize flow field
    vorticity_field = np.zeros_like(x_grid)
    velocity_free_stream = np.zeros(2)
    velocity_free_stream[0] = U_inf
    velocity_field = np.zeros((2, grid_size_y, grid_size_x), dtype=real_t)
    # we use the same buffer for advection, diffusion and velocity recovery
    buffer_scalar_field = np.zeros_like(vorticity_field)
    buffer_vector_field = np.zeros_like(velocity_field)

    # Initialize virtual forcing stuff for fixed cylinder
    X_cm = real_t(2.5) * cyl_radius
    Y_cm = real_t(0.5) * grid_size_y_by_x
    num_lag_nodes = 50
    dtheta = 2.0 * np.pi / num_lag_nodes
    theta = np.linspace(0 + dtheta / 2.0, 2.0 * np.pi - dtheta / 2.0, num_lag_nodes)
    ds = cyl_radius * real_t(dtheta)
    lag_grid_position_field = np.zeros((2, num_lag_nodes), dtype=real_t)
    lag_grid_position_field[0, :] = X_cm + cyl_radius * np.cos(theta).astype(real_t)
    lag_grid_position_field[1, :] = Y_cm + cyl_radius * np.sin(theta).astype(real_t)
    lag_grid_velocity_field = np.zeros_like(lag_grid_position_field)

    # Compile kernels
    # Flow kernels
    penalise_field_towards_boundary_pyst_kernel = (
        gen_penalise_field_boundary_pyst_kernel_2d(
            width=2,
            dx=dx,
            x_grid_field=x_grid,
            y_grid_field=y_grid,
            real_t=real_t,
            num_threads=num_threads,
            fixed_grid_size=(grid_size_y, grid_size_x),
        )
    )
    compute_velocity_from_vorticity_kernel_2d = (
        gen_compute_velocity_from_vorticity_kernel_2d(
            real_t, (grid_size_y, grid_size_x), dx, num_threads
        )
    )
    add_fixed_val = gen_add_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(grid_size_y, grid_size_x),
        num_threads=num_threads,
        field_type="vector",
    )
    advection_and_diffusion_with_forcing_timestep = (
        gen_advection_diffusion_with_forcing_timestep_kernel_2d(
            real_t, (grid_size_y, grid_size_x), dx, nu, num_threads
        )
    )

    # Virtual boundary forcing kernels
    virtual_boundary_stiffness_coeff = real_t(-1e4 * ds)
    virtual_boundary_damping_coeff = real_t(-1e1 * ds)
    interp_kernel_width = 2
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
        grid_dim=2,
        dx=dx,
        eul_grid_coord_shift=eul_grid_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        enable_eul_grid_forcing_reset=True,
        num_threads=num_threads,
    )
    compute_feedback_force = virtual_boundary_forcing.compute_interaction_forcing

    CFL = real_t(0.1)
    # iterate
    timescale = cyl_radius / U_inf
    t_end_hat = real_t(200.0)  # non-dimensional end time
    t_end = t_end_hat * timescale  # dimensional end time
    t = real_t(0.0)
    if save_snap:
        foto_timer = 0.0
        foto_timer_limit = t_end / 50
        import matplotlib.pyplot as plt

        plt.style.use("seaborn")
    if save_diagnostic:
        data_timer = 0.0
        data_timer_limit = 0.25 * timescale
        time = []
        drag_coeffs = []

    while t < t_end:

        # penalise vorticity towards boundary
        penalise_field_towards_boundary_pyst_kernel(
            field=vorticity_field,
            x_grid_field=x_grid,
            y_grid_field=y_grid,
        )

        # compute velocity from vorticity
        compute_velocity_from_vorticity_kernel_2d(
            velocity_field=velocity_field,
            vorticity_field=vorticity_field,
            stream_func_field=buffer_scalar_field,
        )

        # add freestream
        add_fixed_val(
            sum_field=velocity_field,
            vector_field=velocity_field,
            fixed_vals=velocity_free_stream,
        )

        # compute timestep
        dt = compute_advection_diffusion_timestep(
            velocity_field=velocity_field, CFL=CFL, nu=nu, dx=dx
        )

        # Plot solution
        if save_snap:
            if foto_timer >= foto_timer_limit or foto_timer == 0:
                foto_timer = 0.0
                fig = plt.figure(frameon=True, dpi=150)
                ax = fig.add_subplot(111)
                plt.contourf(
                    x_grid,
                    y_grid,
                    vorticity_field,
                    levels=np.linspace(-25, 25, 100),
                    extend="both",
                    cmap=lab_cmap,
                )
                plt.colorbar()
                plt.plot(
                    lag_grid_position_field[0],
                    lag_grid_position_field[1],
                    linewidth=2,
                    color="k",
                )
                ax.set_aspect(aspect=1)
                ax.set_title(f"Vorticity, t_hat: {t / timescale:.2f}")
                plt.savefig(
                    "snap_" + str("%0.4d" % (t * 100)) + ".png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.clf()
                plt.close("all")
                print(
                    f"t_hat: {t / timescale:.2f}, max_vort: {np.amax(vorticity_field):.4f}"
                )
            foto_timer += dt

        if save_diagnostic:
            if data_timer >= data_timer_limit or data_timer == 0:
                data_timer = 0.0
                time.append(t / timescale)

                # calculate drag
                F = np.sum(virtual_boundary_forcing.lag_grid_forcing_field[0, ...])
                drag_coeff = np.fabs(F) / U_inf / U_inf / cyl_radius
                drag_coeffs.append(drag_coeff)
                print(f"Cd: {drag_coeff}")
                # velocity deviation at forcing points
                velocity_deviation = np.fabs(
                    lag_grid_velocity_field
                    - virtual_boundary_forcing.lag_grid_flow_velocity_field
                )
                print(f"Velocity deviation: " f"{np.amax(velocity_deviation) / U_inf}")
            data_timer += dt
        t = t + dt

        # compute flow forcing
        eul_grid_forcing_field = buffer_vector_field.view()
        compute_feedback_force(
            eul_grid_forcing_field=eul_grid_forcing_field,
            eul_grid_velocity_field=velocity_field,
            lag_grid_position_field=lag_grid_position_field,
            lag_grid_velocity_field=lag_grid_velocity_field,
            dt=dt,
        )

        # advect and diffuse vorticity with forcing
        advection_and_diffusion_with_forcing_timestep(
            eul_grid_forcing_field=eul_grid_forcing_field,
            field=vorticity_field,
            velocity_field=velocity_field,
            flux_buffer=buffer_scalar_field,
            dt=dt,
            forcing_prefactor=dt,
        )

    # compile video
    if save_snap:
        os.system("rm -f flow.mp4")
        os.system(
            "ffmpeg -r 10 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
            "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
            " flow.mp4"
        )
        os.system("rm -f snap*.png")

    if save_diagnostic:
        np.savetxt(
            "drag_vs_time.csv",
            np.c_[np.array(time), np.array(drag_coeffs)],
            delimiter=",",
        )
        plt.figure()
        plt.plot(np.array(time), np.array(drag_coeffs))
        plt.ylim([0.7, 1.7])
        plt.xlabel("Non-dimensional time")
        plt.ylabel("Drag coefficient, Cd")
        plt.savefig("drag_vs_time.png")


if __name__ == "__main__":
    grid_size_x = 512
    grid_size_y = 256
    flow_past_cylinder_boundary_forcing_case(
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        save_snap=True,
        save_diagnostic=True,
    )

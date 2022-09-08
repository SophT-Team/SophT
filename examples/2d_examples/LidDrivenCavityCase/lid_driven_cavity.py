import matplotlib.pyplot as plt
import numpy as np
import os
from sopht.numeric.immersed_boundary_ops import VirtualBoundaryForcing
from sopht.utils.precision import get_real_t
import sopht_simulator as sps


def lid_driven_cavity_case(
    grid_size,
    Re=100,
    num_threads=4,
    coupling_stiffness=-7.5e2,
    coupling_damping=-3e-1,
    precision="single",
    save_diagnostic=False,
):
    """
    This example considers a lid driven cavity flow using immersed
    boundary forcing.
    """
    real_t = get_real_t(precision)
    # Flow parameters
    U = 1.0
    ldc_side_length = 0.6
    nu = ldc_side_length * U / Re
    CFL = 0.1
    x_range = 1.0

    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=(grid_size, grid_size),
        x_range=x_range,
        kinematic_viscosity=nu,
        CFL=CFL,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        num_threads=num_threads,
    )

    # Initialize virtual forcing grid for ldc boundaries
    # TODO refactor as 4 fixed osserat rods
    num_ldc_sides = 4
    num_lag_nodes_per_side = 100
    num_lag_nodes = num_ldc_sides * num_lag_nodes_per_side
    ds = ldc_side_length / num_lag_nodes_per_side
    lag_grid_position_field = np.zeros((2, num_lag_nodes), dtype=real_t)
    side_coordinates_range = np.linspace(
        0.5 - 0.5 * ldc_side_length + 0.5 * ds,
        0.5 + 0.5 * ldc_side_length - 0.5 * ds,
        num_lag_nodes_per_side,
    )
    # top boundary
    lag_grid_position_field[0, :num_lag_nodes_per_side] = side_coordinates_range
    lag_grid_position_field[1, :num_lag_nodes_per_side] = 0.5 + 0.5 * ldc_side_length
    # right boundary
    lag_grid_position_field[0, num_lag_nodes_per_side : 2 * num_lag_nodes_per_side] = (
        0.5 + 0.5 * ldc_side_length
    )
    lag_grid_position_field[
        1, num_lag_nodes_per_side : 2 * num_lag_nodes_per_side
    ] = side_coordinates_range
    # bottom boundary
    lag_grid_position_field[
        0, 2 * num_lag_nodes_per_side : 3 * num_lag_nodes_per_side
    ] = side_coordinates_range
    lag_grid_position_field[
        1, 2 * num_lag_nodes_per_side : 3 * num_lag_nodes_per_side
    ] = (0.5 - 0.5 * ldc_side_length)
    # left boundary
    lag_grid_position_field[
        0, 3 * num_lag_nodes_per_side : 4 * num_lag_nodes_per_side
    ] = (0.5 - 0.5 * ldc_side_length)
    lag_grid_position_field[
        1, 3 * num_lag_nodes_per_side : 4 * num_lag_nodes_per_side
    ] = side_coordinates_range

    lag_grid_velocity_field = np.zeros_like(lag_grid_position_field)
    lag_grid_velocity_field[0, :num_lag_nodes_per_side] = U
    ldc_mask = (np.fabs(flow_sim.x_grid - 0.5) < 0.5 * ldc_side_length) * (
        np.fabs(flow_sim.y_grid - 0.5) < 0.5 * ldc_side_length
    )

    # Virtual boundary forcing kernels
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        grid_dim=2,
        dx=flow_sim.dx,
        num_lag_nodes=num_lag_nodes,
        real_t=real_t,
        enable_eul_grid_forcing_reset=False,
        num_threads=num_threads,
    )
    compute_flow_interaction = virtual_boundary_forcing.compute_interaction_forcing

    # iterate
    timescale = ldc_side_length / U
    t_end_hat = 3.0  # non-dimensional end time
    t_end = t_end_hat * timescale  # dimensional end time
    t = 0.0
    foto_timer = 0.0
    foto_timer_limit = t_end / 50

    # create fig for plotting flow fields
    fig, ax = sps.create_figure_and_axes()

    while t < t_end:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(f"U velocity, t_hat: {t / timescale:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.x_grid,
                flow_sim.y_grid,
                ldc_mask * flow_sim.velocity_field[0],
                levels=np.linspace(-0.5, 0.5, 100),
                extend="both",
                cmap=sps.lab_cmap,
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.scatter(
                lag_grid_position_field[0],
                lag_grid_position_field[1],
                s=10,
                color="k",
            )
            sps.save_and_clear_fig(
                fig, ax, cbar, file_name="snap_" + str("%0.4d" % (t * 100)) + ".png"
            )
            print(
                f"time: {t:.2f} ({(t/t_end*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}"
            )

        dt = flow_sim.compute_stable_timestep()

        # compute flow forcing and timestep forcing
        virtual_boundary_forcing.time_step(dt=dt)
        compute_flow_interaction(
            eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
            eul_grid_velocity_field=flow_sim.velocity_field,
            lag_grid_position_field=lag_grid_position_field,
            lag_grid_velocity_field=lag_grid_velocity_field,
        )

        # timestep the flow
        flow_sim.time_step(dt)

        # update time
        t = t + dt
        foto_timer += dt

    # compile video
    os.system("rm -f flow.mp4")
    os.system(
        "ffmpeg -r 10 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
        "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
        " flow.mp4"
    )
    os.system("rm -f snap*.png")

    if save_diagnostic:
        plt.figure()
        plt.plot(
            (flow_sim.y_grid[..., grid_size // 2] - (0.5 - 0.5 * ldc_side_length))
            / ldc_side_length,
            (ldc_mask * flow_sim.velocity_field)[0, ..., grid_size // 2],
        )
        plt.xlim([0.0, 1.0])
        plt.xlabel("Y")
        plt.ylabel("U(Y) at X = 0.5")
        plt.savefig("U_variation_with_Y.png")


if __name__ == "__main__":
    grid_size = 256
    lid_driven_cavity_case(
        grid_size=grid_size,
        save_diagnostic=True,
    )

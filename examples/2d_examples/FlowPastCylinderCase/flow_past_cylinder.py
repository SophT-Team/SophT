import elastica as ea
import click
import matplotlib.pyplot as plt
import numpy as np
from sopht.utils.precision import get_real_t
import sopht_simulator as sps


def flow_past_cylinder_boundary_forcing_case(
    nondim_final_time,
    grid_size,
    reynolds,
    coupling_stiffness=-5e4,
    coupling_damping=-20,
    num_threads=4,
    precision="single",
    save_diagnostic=False,
):
    """
    This example considers a simple flow past cylinder using immersed
    boundary forcing.
    """
    grid_dim = 2
    grid_size_y, grid_size_x = grid_size
    real_t = get_real_t(precision)
    x_axis_idx = sps.VectorField.x_axis_idx()
    y_axis_idx = sps.VectorField.y_axis_idx()
    # Flow parameters
    velocity_scale = 1.0
    velocity_free_stream = np.zeros(grid_dim)
    velocity_free_stream[x_axis_idx] = velocity_scale
    cyl_radius = 0.03
    nu = cyl_radius * velocity_scale / reynolds
    x_range = 1.0

    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=True,
        real_t=real_t,
        num_threads=num_threads,
    )

    # Initialize fixed cylinder (elastica rigid body) with direction along Z
    x_cm = 2.5 * cyl_radius
    y_cm = 0.5 * grid_size_y / grid_size_x
    start = np.array([x_cm, y_cm, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    density = 1e3
    cylinder = ea.Cylinder(start, direction, normal, base_length, cyl_radius, density)
    # Since the cylinder is fixed, we dont add it to pyelastica simulator,
    # and directly use it for setting up the flow interactor.

    # ==================FLOW-BODY COMMUNICATOR SETUP START======
    num_lag_nodes = 60
    cylinder_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=cylinder,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        forcing_grid_cls=sps.CircularCylinderForcingGrid,
        num_forcing_points=num_lag_nodes,
    )
    # ==================FLOW-BODY COMMUNICATOR SETUP END======

    # iterate
    timescale = cyl_radius / velocity_scale
    final_time = nondim_final_time * timescale  # dimensional end time
    time = 0.0
    foto_timer = 0.0
    foto_timer_limit = final_time / 50

    data_timer = 0.0
    data_timer_limit = 0.25 * timescale
    drag_coeffs_time = []
    drag_coeffs = []

    # create fig for plotting flow fields
    fig, ax = sps.create_figure_and_axes()

    while time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(f"Vorticity, time: {time / timescale:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                flow_sim.vorticity_field,
                levels=np.linspace(-25, 25, 100),
                extend="both",
                cmap=sps.lab_cmap,
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.scatter(
                cylinder_flow_interactor.forcing_grid.position_field[x_axis_idx],
                cylinder_flow_interactor.forcing_grid.position_field[y_axis_idx],
                s=4,
                color="k",
            )
            sps.save_and_clear_fig(
                fig, ax, cbar, file_name="snap_" + str("%0.4d" % (time * 100)) + ".png"
            )
            print(
                f"time: {time:.2f} ({(time / final_time * 100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                "grid deviation L2 error: "
                f"{cylinder_flow_interactor.get_grid_deviation_error_l2_norm():.6f}"
            )

        # track diagnostic data
        if data_timer >= data_timer_limit or data_timer == 0:
            data_timer = 0.0
            drag_coeffs_time.append(time / timescale)
            # calculate drag
            F = np.sum(cylinder_flow_interactor.lag_grid_forcing_field[x_axis_idx, ...])
            drag_coeff = np.fabs(F) / velocity_scale / velocity_scale / cyl_radius
            drag_coeffs.append(drag_coeff)

        dt = flow_sim.compute_stable_timestep()

        # compute flow forcing and timestep forcing
        cylinder_flow_interactor.time_step(dt=dt)
        cylinder_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt=dt, free_stream_velocity=velocity_free_stream)

        # update time
        time += dt
        foto_timer += dt
        data_timer += dt

    # compile video
    sps.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=10
    )

    plt.figure()
    plt.plot(np.array(drag_coeffs_time), np.array(drag_coeffs))
    plt.ylim([0.7, 1.7])
    plt.xlabel("Non-dimensional time")
    plt.ylabel("Drag coefficient, Cd")
    plt.savefig("drag_vs_time.png")
    if save_diagnostic:
        np.savetxt(
            "drag_vs_time.csv",
            np.c_[np.array(drag_coeffs_time), np.array(drag_coeffs)],
            delimiter=",",
        )


if __name__ == "__main__":

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    @click.option(
        "--sim_grid_size_x", default=512, help="Number of grid points in x direction."
    )
    @click.option(
        "--nondim_final_time",
        default=200.0,
        help="Non-dimensional final simulation time.",
    )
    @click.option("--reynolds", default=200.0, help="Reynolds number.")
    def simulate_custom_flow_past_cylinder_case(
        num_threads, sim_grid_size_x, nondim_final_time, reynolds
    ):
        sim_grid_size_y = sim_grid_size_x // 2
        sim_grid_size = (sim_grid_size_y, sim_grid_size_x)
        click.echo(f"Number of threads for parallelism: {num_threads}")
        click.echo(f"Grid size: {sim_grid_size}")
        click.echo(f"Reynolds number: {reynolds}")

        flow_past_cylinder_boundary_forcing_case(
            nondim_final_time=nondim_final_time,
            grid_size=sim_grid_size,
            reynolds=reynolds,
            save_diagnostic=True,
            num_threads=num_threads,
        )

    simulate_custom_flow_past_cylinder_case()

import click
import elastica as ea
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu
import sys

sys.path.append("/")


def flow_past_constant_temperature_sphere_case(
    nondim_time: float,
    grid_size: tuple[int, int, int],
    reynolds: float = 100.0,
    prandtl: float = 1,
    sphere_temperature: float = 0,
    coupling_stiffness: float = -6e5 / 4,
    coupling_damping: float = -3.5e2 / 4,
    num_threads: int = 4,
    precision: str = "single",
    save_flow_data: bool = False,
) -> None:
    """
    This example considers the case of flow past a constant temperature sphere in 3D.
    """
    grid_dim = 3
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx: int = spu.VectorField.x_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    x_range = 1.0
    far_field_velocity = 1.0
    rho_f = 1.0
    sphere_diameter = 0.4 * min(grid_size_z, grid_size_y) / grid_size_x * x_range
    nu = far_field_velocity * sphere_diameter / reynolds
    thermal_diffusivity = nu / prandtl
    flow_sim = sps.UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
        with_forcing=True,
        with_free_stream_flow=True,
        time=0.0,
    )
    thermal_sim = sps.PassiveTransportScalarFieldFlowSimulator(
        diffusivity_constant=thermal_diffusivity,
        grid_dim=grid_dim,
        grid_size=grid_size,
        x_range=x_range,
        real_t=real_t,
        num_threads=num_threads,
        time=0.0,
        field_type="scalar",
        velocity_field=flow_sim.velocity_field,
        with_forcing=True,
    )
    #
    sphere_projected_area = 0.25 * np.pi * sphere_diameter**2
    drag_force_scale = 0.5 * rho_f * far_field_velocity**2 * sphere_projected_area
    sphere_surface_area = np.pi * sphere_diameter**2

    # Initialize velocity = c in X direction
    velocity_free_stream = np.array([far_field_velocity, 0.0, 0.0])

    # Initialize fixed sphere (elastica rigid body)
    x_cm = 0.25 * flow_sim.x_range
    y_cm = 0.5 * flow_sim.y_range
    z_cm = 0.5 * flow_sim.z_range
    sphere_com = np.array([x_cm, y_cm, z_cm])
    density = 1e3
    sphere = ea.Sphere(
        center=sphere_com, base_radius=(sphere_diameter / 2.0), density=density
    )
    # Since the sphere is fixed, we don't add it to pyelastica simulator,
    # and directly use it for setting up the flow interactor.
    # ==================FLOW-BODY COMMUNICATOR SETUP START======
    num_forcing_points_along_equator = int(
        1.875 * sphere_diameter / x_range * grid_size_x
    )
    sphere_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=sphere,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        forcing_grid_cls=sps.SphereForcingGrid,
        num_forcing_points_along_equator=num_forcing_points_along_equator,
    )
    sphere_thermal_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=sphere,
        eul_grid_forcing_field=thermal_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=thermal_sim.primary_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=thermal_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        field_type="scalar",
        forcing_grid_cls=sps.SphereConstantTemperatureForcingGrid,
        num_forcing_points_along_equator=num_forcing_points_along_equator,
        sphere_temperature=sphere_temperature,
    )

    # ==================FLOW-BODY COMMUNICATOR SETUP END======

    if save_flow_data:
        # setup flow IO
        io = spu.EulerianFieldIO(
            position_field=flow_sim.position_field,
            eulerian_fields_dict={
                "vorticity": flow_sim.vorticity_field,
                "velocity": flow_sim.velocity_field,
                "temperature": thermal_sim.primary_field,
            },
        )
        # Initialize sphere IO
        sphere_io = spu.IO(dim=grid_dim, real_dtype=real_t)
        # Add vector field on lagrangian grid
        sphere_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=sphere_flow_interactor.forcing_grid.position_field,
            lagrangian_grid_name="sphere",
            vector_3d=sphere_flow_interactor.lag_grid_forcing_field,
        )

    timescale = sphere_diameter / far_field_velocity
    t_end_hat = nondim_time  # non-dimensional end time
    t_end = t_end_hat * timescale  # dimensional end time
    foto_timer = 0.0
    foto_timer_limit = timescale
    # Find the sphere center on euler grid
    # First find the euler grid centers in z direction
    euler_grid_center_in_z_dir = 0.5 * (
        flow_sim.position_field[z_axis_idx, 1:, grid_size_y // 2, 0]
        + flow_sim.position_field[z_axis_idx, :-1, grid_size_y // 2, 0]
    )
    # Find the sphere center in z direction
    lag_grid_center_in_z_dir = np.mean(
        sphere_flow_interactor.forcing_grid.position_field[z_axis_idx]
    )
    # Find the Eulerian grid index that has the sphere center (z coordinate)
    sphere_center_on_euler_grid_idx = int(
        np.argmin(np.abs(euler_grid_center_in_z_dir - lag_grid_center_in_z_dir))
    )
    time = []
    drag_coeffs = []
    flow_vel_along_sphere_center = []
    nusslet_number_time = []
    nusslet_number = []

    # create fig for plotting flow fields
    fig_flow, ax_flow = spu.create_figure_and_axes()
    fig_thermal, ax_thermal = spu.create_figure_and_axes()

    # iterate
    while flow_sim.time < t_end:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            # calculate drag
            drag_force = np.fabs(
                np.sum(sphere_flow_interactor.lag_grid_forcing_field[x_axis_idx, ...])
            )
            drag_coeff = drag_force / drag_force_scale
            time.append(flow_sim.time)
            drag_coeffs.append(drag_coeff)
            with open(
                "../FlowPastConstantTemperatureSphereCase/drag_vs_time.csv", "ab"
            ) as f:
                np.savetxt(
                    f,
                    np.c_[np.array(time), np.array(drag_coeffs)],
                    delimiter=",",
                )
            flow_vel_along_sphere_center.append(
                np.mean(
                    np.mean(
                        flow_sim.velocity_field[
                            x_axis_idx,
                            sphere_center_on_euler_grid_idx : sphere_center_on_euler_grid_idx
                            + 2,
                            grid_size_y // 2 - 1 : grid_size_y // 2 + 1,
                            :,
                        ],
                        axis=1,  # Average velocities in y
                    ),
                    axis=0,  # Average velocities in z
                )
            )
            # Compute Nusslet number
            nusslet_number_time.append(thermal_sim.time / timescale)
            # Calculate Nusslet number
            Q = np.fabs(np.sum(sphere_thermal_interactor.lag_grid_forcing_field))
            nusslet = (
                Q
                * sphere_diameter
                / (
                    (sphere_temperature - 0.0)
                    * sphere_surface_area
                    * thermal_diffusivity
                )
            )
            nusslet_number.append(nusslet)
            with open(
                "../FlowPastConstantTemperatureSphereCase/nusslet_vs_time.csv", "ab"
            ) as f:
                np.savetxt(
                    f,
                    np.c_[np.array(nusslet_number_time), np.array(nusslet_number)],
                    delimiter=",",
                )
            if save_flow_data:
                io.save(
                    h5_file_name="sopht_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
                sphere_io.save(
                    h5_file_name="sphere_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
            ax_flow.set_title(f"Velocity X comp, time: {flow_sim.time / timescale:.2f}")
            contourf_obj = ax_flow.contourf(
                flow_sim.position_field[x_axis_idx, :, grid_size_y // 2, :],
                flow_sim.position_field[z_axis_idx, :, grid_size_y // 2, :],
                np.mean(
                    flow_sim.velocity_field[
                        x_axis_idx, :, grid_size_y // 2 - 1 : grid_size_y // 2 + 1, :
                    ],
                    axis=1,
                ),
                levels=50,
                extend="both",
                cmap=spu.get_lab_cmap(),
            )
            cbar = fig_flow.colorbar(mappable=contourf_obj, ax=ax_flow)
            ax_flow.scatter(
                sphere_flow_interactor.forcing_grid.position_field[x_axis_idx],
                sphere_flow_interactor.forcing_grid.position_field[z_axis_idx],
                s=5,
                color="k",
            )
            spu.save_and_clear_fig(
                fig_flow,
                ax_flow,
                cbar,
                file_name="flow_snap_" + str("%0.4d" % (flow_sim.time * 100)) + ".png",
            )

            ax_thermal.set_title(
                f"Temperature, time: {thermal_sim.time / timescale:.2f}"
            )
            contourf_obj = ax_thermal.contourf(
                thermal_sim.position_field[x_axis_idx, :, grid_size_y // 2, :],
                thermal_sim.position_field[z_axis_idx, :, grid_size_y // 2, :],
                np.mean(
                    thermal_sim.primary_field[
                        :, grid_size_y // 2 - 1 : grid_size_y // 2 + 1, :
                    ],
                    axis=1,
                ),
                levels=np.linspace(0, sphere_temperature * 1.5, 100),
                extend="both",
                cmap="Blues",
            )
            cbar = fig_thermal.colorbar(mappable=contourf_obj, ax=ax_thermal)
            ax_thermal.scatter(
                sphere_thermal_interactor.forcing_grid.position_field[x_axis_idx],
                sphere_thermal_interactor.forcing_grid.position_field[z_axis_idx],
                s=5,
                color="k",
            )
            spu.save_and_clear_fig(
                fig_thermal,
                ax_thermal,
                cbar,
                file_name="thermal_snap_"
                + str("%0.4d" % (flow_sim.time * 100))
                + ".png",
            )

            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/t_end*100):2.1f}%), "
                f"max_temp: {np.amax(thermal_sim.primary_field):.4f}, "
                f"min_temp: {np.amin(thermal_sim.primary_field):.4f}, "
                f"thermal grid deviation L2 error: {sphere_thermal_interactor.get_grid_deviation_error_l2_norm():.6f}, "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"drag coeff: {drag_coeff:.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f} "
                "grid deviation L2 error: "
                f"{sphere_flow_interactor.get_grid_deviation_error_l2_norm():.6f}"
            )

        dt_flow = flow_sim.compute_stable_timestep(dt_prefac=0.5)
        dt_thermal = thermal_sim.compute_stable_timestep(dt_prefac=0.5)
        dt = min(dt_flow, dt_thermal)

        sphere_thermal_interactor.time_step(dt)
        sphere_thermal_interactor()
        thermal_sim.time_step(dt)

        # # compute flow forcing and timestep forcing
        sphere_flow_interactor.time_step(dt=dt)
        sphere_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt=dt, free_stream_velocity=velocity_free_stream)

        # update timers
        foto_timer += dt

    fig, ax = spu.create_figure_and_axes(fig_aspect_ratio="default")
    ax.plot(np.array(time), np.array(drag_coeffs), label="numerical")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drag coefficient")
    fig.savefig("drag_coeff_vs_time.png")

    fig, ax = spu.create_figure_and_axes(fig_aspect_ratio="default")
    ax.plot(np.array(nusslet_number_time), np.array(nusslet_number), label="numerical")
    ax.set_xlabel("Non-dimensional time")
    ax.set_ylabel("Nusslet number, Nu")
    fig.savefig("nusslet_vs_time.png")

    np.savetxt(
        "drag_vs_time.csv",
        np.c_[np.array(time), np.array(drag_coeffs)],
        delimiter=",",
        header="time, drag_coeff",
    )

    np.savetxt(
        "x_vel_along_center_line_vs_time.csv",
        np.c_[np.array(time), np.array(flow_vel_along_sphere_center)],
        delimiter=",",
        header="time, flow_vel_along_sphere_center",
    )
    np.savetxt(
        "nusslet_vs_time.csv",
        np.c_[np.array(nusslet_number_time), np.array(nusslet_number)],
        delimiter=",",
        header="time, nusslet number",
    )

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="flow_snap", frame_rate=10
    )
    spu.make_video_from_image_series(
        video_name="thermal", image_series_name="thermal_snap", frame_rate=10
    )

    if save_flow_data:
        spu.make_dir_and_transfer_h5_data(dir_name="flow_data_h5")


if __name__ == "__main__":

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    @click.option("--nx", default=256, help="Number of grid points in x direction.")
    @click.option("--reynolds", default=100.0, help="Reynolds number of flow.")
    @click.option("--prandtl", default=1.0, help="Prandtl number.")
    @click.option("--temperature_sphere", default=10, help="Sphere temperature.")
    def simulate_flow_past_constant_temperature_sphere(
        num_threads: int,
        nx: int,
        reynolds: float,
        prandtl: float,
        temperature_sphere: float,
    ) -> None:
        ny = nx // 2
        nz = nx // 2
        # in order Z, Y, X
        grid_size = (nz, ny, nx)
        click.echo(f"Number of threads for parallelism: {num_threads}")
        click.echo(f"Grid size: {grid_size}")
        click.echo(f"Flow Reynolds number: {reynolds}")
        flow_past_constant_temperature_sphere_case(
            nondim_time=20.0,
            grid_size=grid_size,
            num_threads=num_threads,
            reynolds=reynolds,
            prandtl=prandtl,
            sphere_temperature=temperature_sphere,
            save_flow_data=False,
        )

    simulate_flow_past_constant_temperature_sphere()

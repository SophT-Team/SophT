import elastica as ea
import click
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu
import sys

sys.path.append("/")


def flow_past_insulated_cylinder_case(
    nondim_final_time: float,
    grid_size: tuple[int, int],
    reynolds: float,
    coupling_stiffness: float = -5e4,
    coupling_damping: float = -20,
    num_threads: int = 4,
    prandtl=1.0,
    temperature_cylinder=1.0,
    precision: str = "single",
    save_diagnostic=False,
) -> None:
    """
    This example considers a simple flow past constant temperature and insulated cylinder using immersed
    boundary forcing.
    """
    grid_dim = 2
    grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    # Flow parameters
    velocity_scale = 1.0
    velocity_free_stream = np.zeros(grid_dim)
    velocity_free_stream[x_axis_idx] = velocity_scale
    cyl_radius = 0.03
    nu = 2 * cyl_radius * velocity_scale / reynolds
    thermal_diffusivity = nu / prandtl
    x_range = 1.0 / 2
    y_range = x_range * grid_size_y / grid_size_x
    flow_sim = sps.UnboundedNavierStokesFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        with_forcing=True,
        with_free_stream_flow=True,
        real_t=real_t,
        num_threads=num_threads,
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

    # Initialize fixed cylinder (elastica rigid body) with direction along Z
    x_cm = 2.5 * cyl_radius  # 0.2 * x_range #2.5 * cyl_radius
    y_cm = 0.5 * y_range
    start = np.array([x_cm, y_cm, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    density = 1e3
    cylinder = ea.Cylinder(start, direction, normal, base_length, cyl_radius, density)

    start = start + np.array([x_cm, 0, 0.0])
    insulated_cylinder = ea.Cylinder(
        start, direction, normal, base_length, cyl_radius, density
    )
    # Since the cylinder is fixed, we dont add it to pyelastica simulator,
    # and directly use it for setting up the flow interactor.

    # ==================FLOW-BODY COMMUNICATOR SETUP START======
    num_lag_nodes = grid_size_x // 5

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

    cylinder_thermal_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=cylinder,
        eul_grid_forcing_field=thermal_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=thermal_sim.primary_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=thermal_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        field_type="scalar",
        forcing_grid_cls=sps.CircularCylinderConstantTemperatureForcingGrid,
        num_forcing_points=num_lag_nodes,
        cylinder_temperature=temperature_cylinder,
    )

    insulated_cylinder_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=insulated_cylinder,
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

    virtual_thermal_layer_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=insulated_cylinder,
        eul_grid_forcing_field=thermal_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=thermal_sim.primary_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness * 0,
        virtual_boundary_damping_coeff=coupling_damping * 0,
        dx=thermal_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        field_type="scalar",
        forcing_grid_cls=sps.CircularCylinderVirtualLayerTemperatureForcingGrid,
        num_forcing_points=num_lag_nodes,
        eul_dx=thermal_sim.dx,
    )

    insulated_cylinder_thermal_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=insulated_cylinder,
        eul_grid_forcing_field=thermal_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=thermal_sim.primary_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=thermal_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        field_type="scalar",
        forcing_grid_cls=sps.CircularCylinderIndirectNeummanConditionForcingGrid,
        num_forcing_points=num_lag_nodes,
        virtual_layer_interactor=virtual_thermal_layer_interactor,
    )

    # ==================FLOW-BODY COMMUNICATOR SETUP END======

    # iterate
    timescale = cyl_radius / velocity_scale
    final_time = nondim_final_time * timescale  # dimensional end time
    foto_timer = 0.0
    foto_timer_limit = final_time / 50

    data_timer = 0.0
    data_timer_limit = 0.25 * timescale
    drag_coeffs_time = []
    drag_coeffs = []
    q_constant_temperature_time = []
    q_constant_temperature = []
    q_insulated = []
    q_insulated_time = []
    # create fig for plotting flow fields
    fig_thermal, ax_thermal = spu.create_figure_and_axes()
    fig_flow, ax_flow = spu.create_figure_and_axes()

    while thermal_sim.time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax_thermal.set_title(
                f"Temperature, time: {thermal_sim.time / timescale:.2f}"
            )
            contourf_obj = ax_thermal.contourf(
                thermal_sim.position_field[x_axis_idx],
                thermal_sim.position_field[y_axis_idx],
                thermal_sim.primary_field,
                levels=np.linspace(0, temperature_cylinder * 1.5, 100),
                extend="both",
                cmap="Blues",
            )
            cbar = fig_thermal.colorbar(mappable=contourf_obj, ax=ax_thermal)
            ax_thermal.scatter(
                cylinder_thermal_interactor.forcing_grid.position_field[x_axis_idx],
                cylinder_thermal_interactor.forcing_grid.position_field[y_axis_idx],
                s=4,
                color="k",
            )
            ax_thermal.scatter(
                insulated_cylinder_thermal_interactor.forcing_grid.position_field[
                    x_axis_idx
                ],
                insulated_cylinder_thermal_interactor.forcing_grid.position_field[
                    y_axis_idx
                ],
                s=4,
                color="k",
            )
            spu.save_and_clear_fig(
                fig_thermal,
                ax_thermal,
                cbar,
                file_name="thermal_snap_"
                + str("%0.4d" % (thermal_sim.time * 100))
                + ".png",
            )
            ax_flow.set_title(f"Vorticity, time: {flow_sim.time / timescale:.2f}")
            contourf_obj = ax_flow.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                flow_sim.vorticity_field,
                levels=np.linspace(-25, 25, 100),
                extend="both",
                cmap=spu.get_lab_cmap(),
            )
            cbar = fig_thermal.colorbar(mappable=contourf_obj, ax=ax_flow)
            ax_flow.scatter(
                cylinder_flow_interactor.forcing_grid.position_field[x_axis_idx],
                cylinder_flow_interactor.forcing_grid.position_field[y_axis_idx],
                s=4,
                color="k",
            )
            ax_flow.scatter(
                insulated_cylinder_flow_interactor.forcing_grid.position_field[
                    x_axis_idx
                ],
                insulated_cylinder_flow_interactor.forcing_grid.position_field[
                    y_axis_idx
                ],
                s=4,
                color="k",
            )
            spu.save_and_clear_fig(
                fig_flow,
                ax_flow,
                cbar,
                file_name="flow_snap_"
                + str("%0.4d" % (thermal_sim.time * 100))
                + ".png",
            )
            print(
                f"time: {thermal_sim.time:.2f} ({(thermal_sim.time / final_time * 100):2.1f}%), "
                f"max_temp: {np.amax(thermal_sim.primary_field):.4f}, "
                f"min_temp: {np.amin(thermal_sim.primary_field):.4f}, "
                f"thermal grid deviation L2 error: {cylinder_thermal_interactor.get_grid_deviation_error_l2_norm():.6f}, "
                f"temperature mismatch : {np.sum(np.fabs(cylinder_thermal_interactor.lag_grid_velocity_mismatch_field)):.6f}, "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"flow grid deviation L2 error: {cylinder_flow_interactor.get_grid_deviation_error_l2_norm():.6f}, "
                f"flow_dt: {flow_sim.compute_stable_timestep():.6f}"
                f" thermal_dt: {thermal_sim.compute_stable_timestep():.6f}"
            )

        if data_timer >= data_timer_limit or data_timer == 0:
            data_timer = 0.0

            # Compute drag coefficient
            drag_coeffs_time.append(flow_sim.time / timescale)
            # calculate drag
            F = np.sum(cylinder_flow_interactor.lag_grid_forcing_field[x_axis_idx, ...])
            drag_coeff = np.fabs(F) / velocity_scale / velocity_scale / cyl_radius
            drag_coeffs.append(drag_coeff)

            with open("../PassiveDiffusionCase/drag_vs_time.csv", "ab") as f:
                np.savetxt(
                    f,
                    np.c_[np.array(drag_coeffs_time), np.array(drag_coeffs)],
                    delimiter=",",
                )

            # Compute heat flux of the constant temperature cylinder.
            q_constant_temperature_time.append(thermal_sim.time / timescale)
            Q = np.fabs(
                np.sum(np.fabs(cylinder_thermal_interactor.lag_grid_forcing_field))
            )
            q_constant_temperature.append(Q)

            with open(
                "../FlowPastInsulatedCylinderCase/constant_temperature_cylinder_heat_flux_vs_time.csv",
                "ab",
            ) as f:
                np.savetxt(
                    f,
                    np.c_[
                        np.array(q_constant_temperature_time),
                        np.array(q_constant_temperature),
                    ],
                    delimiter=",",
                )

            # Compute heat flux of the insulated cylinder.
            q_insulated_time.append(thermal_sim.time / timescale)
            Q = np.fabs(
                np.sum(insulated_cylinder_thermal_interactor.lag_grid_forcing_field)
            )

            q_insulated.append(Q)

            with open(
                "../FlowPastInsulatedCylinderCase/insulated_cylinder_heat_flux_vs_time.csv",
                "ab",
            ) as f:
                np.savetxt(
                    f,
                    np.c_[np.array(q_insulated_time), np.array(q_insulated)],
                    delimiter=",",
                )

        dt_prefac = 0.2
        dt_flow = flow_sim.compute_stable_timestep(dt_prefac=dt_prefac)
        dt_thermal = thermal_sim.compute_stable_timestep(dt_prefac=dt_prefac)

        dt = min((dt_thermal, dt_flow))

        # compute thermal forcing and timestep thermal forcing
        cylinder_thermal_interactor.time_step(dt=dt)
        cylinder_thermal_interactor()

        virtual_thermal_layer_interactor.time_step(dt=dt)
        virtual_thermal_layer_interactor()

        insulated_cylinder_thermal_interactor.compute_flow_forces_and_torques()
        insulated_cylinder_thermal_interactor.time_step(dt=dt)
        insulated_cylinder_thermal_interactor()

        # timestep the thermal field.
        thermal_sim.time_step(dt=dt)

        # compute flow forcing and timestep forcing
        cylinder_flow_interactor.time_step(dt=dt)
        cylinder_flow_interactor()

        insulated_cylinder_flow_interactor.time_step(dt=dt)
        insulated_cylinder_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt=dt, free_stream_velocity=velocity_free_stream)

        # update timers
        foto_timer += dt
        data_timer += dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="flow_snap", frame_rate=10
    )
    spu.make_video_from_image_series(
        video_name="thermal", image_series_name="thermal_snap", frame_rate=10
    )

    # Save drag data
    fig, ax = spu.create_figure_and_axes(fig_aspect_ratio="default")
    ax.plot(np.array(drag_coeffs_time), np.array(drag_coeffs), label="numerical")
    ax.set_ylim([0.7, 1.7])
    ax.set_xlabel("Non-dimensional time")
    ax.set_ylabel("Drag coefficient")
    fig.savefig("drag_coeff_vs_time.png")

    fig, ax = spu.create_figure_and_axes(fig_aspect_ratio="default")
    ax.plot(
        np.array(q_constant_temperature),
        np.array(q_constant_temperature),
        label="numerical",
    )
    ax.set_xlabel("Non-dimensional time")
    ax.set_ylabel("q constant temperature cylinder, Nu")
    fig.savefig("constant_temp_cylinder_q_vs_time.png")

    fig, ax = spu.create_figure_and_axes(fig_aspect_ratio="default")
    ax.semilogy(np.array(q_insulated_time), np.array(q_insulated), label="numerical")
    ax.set_xlabel("time")
    ax.set_ylabel("q insulated cylinder")
    fig.savefig("insulated_cylinder_q_vs_time.png")

    if save_diagnostic:
        np.savetxt(
            "drag_vs_time.csv",
            np.c_[np.array(drag_coeffs_time), np.array(drag_coeffs)],
            delimiter=",",
            header="time, cd",
        )
        np.savetxt(
            "constant_temperature_cylinder_heat_flux_vs_time.csv",
            np.c_[np.array(q_constant_temperature), np.array(q_constant_temperature)],
            delimiter=",",
            header="time, heat flux",
        )
        np.savetxt(
            "insulated_cylinder_heat_flux_vs_time.csv",
            np.c_[np.array(q_insulated_time), np.array(q_insulated)],
            delimiter=",",
            header="time, heat flux",
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
    @click.option("--reynolds", default=100.0, help="Reynolds number.")
    @click.option("--prandtl", default=1.0, help="Prandtl number.")
    @click.option("--temperature_cylinder", default=10, help="Cylinder temperature.")
    def simulate_flow_past_insulated_cylinder_case(
        num_threads: int,
        sim_grid_size_x: int,
        nondim_final_time: float,
        reynolds: float,
        prandtl: float,
        temperature_cylinder: float,
    ) -> None:
        sim_grid_size_y = sim_grid_size_x // 2
        sim_grid_size = (sim_grid_size_y, sim_grid_size_x)
        click.echo(f"Number of threads for parallelism: {num_threads}")
        click.echo(f"Grid size: {sim_grid_size}")
        click.echo(f"Reynolds number: {reynolds}")
        click.echo(f"Prandtl number: {prandtl}")
        click.echo(f"Cylinder temperature: {temperature_cylinder}")

        flow_past_insulated_cylinder_case(
            nondim_final_time=nondim_final_time,
            grid_size=sim_grid_size,
            reynolds=reynolds,
            num_threads=num_threads,
            prandtl=prandtl,
            temperature_cylinder=temperature_cylinder,
            save_diagnostic=True,
        )

    simulate_flow_past_insulated_cylinder_case()

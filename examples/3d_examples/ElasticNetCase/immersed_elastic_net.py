import click
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu
from elastic_net import ElasticNetSimulator


def immersed_elastic_net_case(
    elastic_net_sim: ElasticNetSimulator,
    domain_range: tuple[float, float, float],
    grid_size_x: int,
    reynolds: float,
    vel_free_stream: float = 1.0,
    coupling_stiffness: float = -2e4,
    coupling_damping: float = -1e1,
    num_threads: int = 4,
    precision: str = "single",
    save_flow_data: bool = False,
) -> None:
    # ==================FLOW SETUP START=========================
    grid_dim = 3
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    z_range, y_range, x_range = domain_range
    grid_size_y = round(y_range / x_range * grid_size_x)
    grid_size_z = round(z_range / x_range * grid_size_x)
    # order Z, Y, X
    grid_size = (grid_size_z, grid_size_y, grid_size_x)
    print(f"Flow grid size: {grid_size}")
    velocity_free_stream = [0.0, 0.0, vel_free_stream]
    length_scale = max(
        elastic_net_sim.elastic_net_length_x,
        elastic_net_sim.elastic_net_length_y,
    )
    kinematic_viscosity = length_scale * vel_free_stream / reynolds
    flow_sim = sps.UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=domain_x_range,
        kinematic_viscosity=kinematic_viscosity,
        with_forcing=True,
        with_free_stream_flow=True,
        real_t=real_t,
        num_threads=num_threads,
        filter_vorticity=True,
        filter_setting_dict={"order": 1, "type": "multiplicative"},
    )
    # ==================FLOW SETUP END=========================
    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    rod_flow_interactor_list = []
    for rod in elastic_net_sim.rod_list:
        rod_flow_interactor = sps.CosseratRodFlowInteraction(
            cosserat_rod=rod,
            eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
            eul_grid_velocity_field=flow_sim.velocity_field,
            virtual_boundary_stiffness_coeff=coupling_stiffness,
            virtual_boundary_damping_coeff=coupling_damping,
            dx=flow_sim.dx,
            grid_dim=grid_dim,
            real_t=real_t,
            num_threads=num_threads,
            forcing_grid_cls=sps.CosseratRodElementCentricForcingGrid,
        )
        rod_flow_interactor_list.append(rod_flow_interactor)
        elastic_net_sim.net_simulator.add_forcing_to(rod).using(
            sps.FlowForces,
            rod_flow_interactor,
        )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======
    if save_flow_data:
        # setup flow IO
        io = spu.EulerianFieldIO(
            position_field=flow_sim.position_field,
            eulerian_fields_dict={
                "vorticity": flow_sim.vorticity_field,
                "velocity": flow_sim.velocity_field,
            },
        )
        # Initialize carpet IO
        elastic_net_io = spu.IO(dim=grid_dim, real_dtype=real_t)
        rod_num_lag_nodes_list = [
            interactor.forcing_grid.num_lag_nodes
            for interactor in rod_flow_interactor_list
        ]
        elastic_net_num_lag_nodes = sum(rod_num_lag_nodes_list)
        elastic_net_lag_grid_position_field = np.zeros(
            (grid_dim, elastic_net_num_lag_nodes)
        )
        elastic_net_lag_grid_forcing_field = np.zeros_like(
            elastic_net_lag_grid_position_field
        )

        def update_elastic_net_lag_grid_fields():
            """Updates the combined lag grid with individual rod grids"""
            start_idx = 0
            for interactor in rod_flow_interactor_list:
                elastic_net_lag_grid_position_field[
                    ..., start_idx : start_idx + interactor.forcing_grid.num_lag_nodes
                ] = interactor.forcing_grid.position_field
                elastic_net_lag_grid_forcing_field[
                    ..., start_idx : start_idx + interactor.forcing_grid.num_lag_nodes
                ] = interactor.lag_grid_forcing_field
                start_idx += interactor.forcing_grid.num_lag_nodes

        update_elastic_net_lag_grid_fields()
        # Add vector field on lagrangian grid
        elastic_net_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=elastic_net_lag_grid_position_field,
            lagrangian_grid_name="elastic_net",
            vector_3d=elastic_net_lag_grid_forcing_field,
        )
    elastic_net_sim.finalize()
    # =================TIMESTEPPING====================
    foto_timer = 0.0
    frames_per_second = 32
    foto_timer_limit = 1.0 / frames_per_second
    time_history = []
    max_displacement_history = []
    drag_history = []
    time_scale = length_scale / vel_free_stream
    net_projected_area = (
        elastic_net_sim.elastic_net_length_x * elastic_net_sim.elastic_net_length_y
    )
    drag_scale = rho_f * vel_free_stream**2 * net_projected_area

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()
    # iterate
    while flow_sim.time < elastic_net_sim.final_time:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            if save_flow_data:
                update_elastic_net_lag_grid_fields()
                io.save(
                    h5_file_name="flow_" + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )
                elastic_net_io.save(
                    h5_file_name="lag_grid_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
            ax.set_title(f"Velocity magnitude, time: {flow_sim.time:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx, :, grid_size_y // 2, :],
                flow_sim.position_field[z_axis_idx, :, grid_size_y // 2, :],
                # TODO function for velocity magnitude
                np.linalg.norm(
                    np.mean(
                        flow_sim.velocity_field[
                            :, :, grid_size_y // 2 - 1 : grid_size_y // 2 + 1, :
                        ],
                        axis=2,
                    ),
                    axis=0,
                ),
                levels=50,
                extend="both",
                cmap="Purples",
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            for rod in elastic_net_sim.rod_list:
                ax.scatter(
                    rod.position_collection[x_axis_idx],
                    rod.position_collection[z_axis_idx],
                    s=5,
                    color="k",
                )
            spu.save_and_clear_fig(
                fig,
                ax,
                cbar,
                file_name="snap_" + str("%0.5d" % (flow_sim.time * 100)) + ".png",
            )
            grid_dev_error = 0.0
            for flow_body_interactor in rod_flow_interactor_list:
                grid_dev_error += (
                    flow_body_interactor.get_grid_deviation_error_l2_norm()
                )
            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time / elastic_net_sim.final_time * 100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f}, "
                f"grid deviation L2 error: {grid_dev_error:.6f}, "
            )
            # compute max displacement along flow and drag
            time_history.append(flow_sim.time / time_scale)
            max_displacement = np.amax(
                np.array(
                    [
                        rod.position_collection[z_axis_idx].max()
                        for rod in elastic_net_sim.rod_list
                    ]
                )
            )
            max_displacement_history.append(max_displacement / length_scale)
            drag = np.sum(
                np.array(
                    [
                        rod_flow_interactor.body_flow_forces[z_axis_idx, 0]
                        for rod_flow_interactor in rod_flow_interactor_list
                    ]
                )
            )
            drag_history.append(drag / drag_scale)

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.25)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, elastic_net_sim.dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            # timestep the cilia simulator
            rod_time = elastic_net_sim.time_step(time=rod_time, time_step=local_rod_dt)
            # timestep the rod_flow_interactors
            for rod_flow_interactor in rod_flow_interactor_list:
                rod_flow_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and rod
        for rod_flow_interactor in rod_flow_interactor_list:
            rod_flow_interactor()

        flow_sim.time_step(
            dt=flow_dt,
            free_stream_velocity=velocity_free_stream,
        )

        # update timer
        foto_timer += flow_dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=frames_per_second
    )

    if save_flow_data:
        spu.make_dir_and_transfer_h5_data(dir_name="flow_data_h5")

    # plot and save relevant diagnostics
    initial_max_displacement = max_displacement_history[0]
    fig, ax = spu.create_figure_and_axes(fig_aspect_ratio="default")
    ax.plot(time_history, max_displacement_history)
    ax.set_xlabel("Time")
    ax.set_ylabel("Max displacement along flow")
    fig.savefig("max_displacement_vs_time.png")
    np.savetxt(
        "net_diagnostics.csv",
        np.c_[
            np.array(time_history),
            np.array(max_displacement_history) - initial_max_displacement,
            np.array(drag_history),
        ],
        delimiter=",",
        header="time, max_displacement, drag",
    )


if __name__ == "__main__":

    # setup the structure of the carpet
    num_rods_along_x = 8  # set >= 2
    num_rods_along_y = 8  # set >= 2
    gap_between_rods = 0.2
    gap_radius_ratio = 10
    base_radius = gap_between_rods / gap_radius_ratio
    spacing_between_rods = gap_between_rods + 2 * base_radius
    elastic_net_length_x = (
        num_rods_along_x - 1
    ) * spacing_between_rods + 2 * base_radius
    elastic_net_length_y = (
        num_rods_along_y - 1
    ) * spacing_between_rods + 2 * base_radius
    # get the flow domain range based on the carpet
    domain_x_range = 1.3 * elastic_net_length_x
    domain_y_range = 1.3 * elastic_net_length_y
    domain_z_range = 1.3 * elastic_net_length_x
    offset_between_net_origin_and_flow_grid_center_x = (
        0.49 * elastic_net_length_x - base_radius
    )
    offset_between_net_origin_and_flow_grid_center_y = (
        0.49 * elastic_net_length_y - base_radius
    )
    elastic_net_origin = np.array(
        [
            0.5 * domain_x_range - offset_between_net_origin_and_flow_grid_center_x,
            0.5 * domain_y_range - offset_between_net_origin_and_flow_grid_center_y,
            0.25 * domain_z_range,
        ]
    )
    # nondim characteristic numbers
    rod_to_fluid_density_ratio = 1e2
    rho_f = 1.0
    rod_density = rho_f * rod_to_fluid_density_ratio

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    @click.option(
        "--grid_size_x", default=128, help="Number of grid points in x direction."
    )
    @click.option("--reynolds", default=100.0, help="Reynolds number of flow.")
    @click.option(
        "--nondim_youngs_modulus",
        default=1e3,
        help="Non-dimensional Youngs modulus of the net.",
    )
    @click.option("--final_time", default=2.0, help="Final simulation time.")
    def simulate_parallelised_immersed_net_case(
        num_threads: int,
        grid_size_x: int,
        reynolds: float,
        nondim_youngs_modulus: float,
        final_time: float,
    ) -> None:
        vel_free_stream_z = 1.0
        youngs_modulus = nondim_youngs_modulus * rho_f * vel_free_stream_z**2
        # discretisation stuff
        n_elem_per_gap = int(grid_size_x / 16)
        elastic_net_simulator = ElasticNetSimulator(
            rod_density=rod_density,
            youngs_modulus=youngs_modulus,
            num_rods_along_x=num_rods_along_x,
            num_rods_along_y=num_rods_along_y,
            final_time=final_time,
            gap_between_rods=gap_between_rods,
            gap_radius_ratio=gap_radius_ratio,
            num_rod_elements_per_gap=n_elem_per_gap,
            elastic_net_origin=elastic_net_origin,
            plot_result=False,
        )
        click.echo(f"Number of threads for parallelism: {num_threads}")
        click.echo(f"Flow Reynolds number: {reynolds}")
        click.echo(
            f"Non-dimensional Youngs modulus of the nest: {nondim_youngs_modulus}"
        )
        immersed_elastic_net_case(
            elastic_net_sim=elastic_net_simulator,
            reynolds=reynolds,
            vel_free_stream=vel_free_stream_z,
            grid_size_x=grid_size_x,
            domain_range=(domain_z_range, domain_y_range, domain_x_range),
            num_threads=num_threads,
            save_flow_data=False,
        )

    simulate_parallelised_immersed_net_case()

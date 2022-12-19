import elastica as ea
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu


def analytical_pipe_flow_velocity(
    radial_coordinate: float | np.ndarray, mean_velocity: float, pipe_radius: float
) -> float | np.ndarray:
    return 2.0 * mean_velocity * (1.0 - (radial_coordinate / pipe_radius) ** 2)


def flow_through_circular_pipe_case(
    grid_size: tuple[int, int, int],
    coupling_stiffness: float = -4e5,
    coupling_damping: float = -2.5e2,
    num_threads: int = 4,
    precision: str = "single",
    save_data: bool = False,
) -> None:
    """
    This example considers the case of steady flow through a pipe in 3D.
    """
    grid_dim = 3
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    x_range = 1.0
    nu = 1e-2
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=True,
        real_t=real_t,
        num_threads=num_threads,
    )
    # Initialize velocity = c in X direction
    mean_velocity = 1.0
    velocity_free_stream = [mean_velocity, 0.0, 0.0]
    # Initialize fixed cylinder (elastica rigid body) with direction along X
    boundary_offset = 2 * flow_sim.dx  # to avoid interpolation at boundaries
    base_length = flow_sim.x_range - 2 * boundary_offset
    cyl_radius = 0.375 * min(flow_sim.y_range, flow_sim.z_range)
    x_cm = boundary_offset
    y_cm = 0.5 * flow_sim.y_range
    z_cm = 0.5 * flow_sim.z_range
    start = np.array([x_cm, y_cm, z_cm])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    density = 1e3
    cylinder = ea.Cylinder(start, direction, normal, base_length, cyl_radius, density)
    # Since the cylinder is fixed, we dont add it to pyelastica simulator,
    # and directly use it for setting up the flow interactor.
    # ==================FLOW-BODY COMMUNICATOR SETUP START======
    num_forcing_points_along_length = 64
    cylinder_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=cylinder,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        forcing_grid_cls=sps.OpenEndCircularCylinderForcingGrid,
        num_forcing_points_along_length=num_forcing_points_along_length,
    )
    # ==================FLOW-BODY COMMUNICATOR SETUP END======

    if save_data:
        # setup IO
        # TODO internalise this in flow simulator as dump_fields
        io_origin = np.array(
            [
                flow_sim.position_field[z_axis_idx].min(),
                flow_sim.position_field[y_axis_idx].min(),
                flow_sim.position_field[x_axis_idx].min(),
            ]
        )
        io_dx = flow_sim.dx * np.ones(grid_dim)
        io_grid_size = np.array(grid_size)
        io = spu.IO(dim=grid_dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field, velocity=flow_sim.velocity_field
        )

    t_end = 1.0
    foto_timer = 0.0
    foto_timer_limit = t_end / 40

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes(fig_aspect_ratio="default")
    radial_coordinate = (
        flow_sim.position_field[y_axis_idx, grid_size_z // 2, ..., grid_size_x // 2]
        - y_cm
    )
    anal_velocity_profile = analytical_pipe_flow_velocity(
        radial_coordinate=radial_coordinate,
        mean_velocity=mean_velocity,
        pipe_radius=cyl_radius,
    )

    # iterate
    while flow_sim.time < t_end:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/t_end*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f}, "
                "grid deviation L2 error: "
                f"{cylinder_flow_interactor.get_grid_deviation_error_l2_norm():.6f}"
            )
            if save_data:
                io.save(
                    h5_file_name="sopht_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
            # midplane along X
            sim_velocity_profile = 0.5 * np.sum(
                flow_sim.velocity_field[
                    x_axis_idx,
                    grid_size_z // 2 - 1 : grid_size_z // 2 + 1,
                    ...,
                    grid_size_x // 2,
                ],
                axis=0,
            )
            ax.plot(radial_coordinate, sim_velocity_profile, label="numerical")
            ax.plot(radial_coordinate, anal_velocity_profile, label="analytical")
            ax.legend()
            ax.set_xlim(-cyl_radius, cyl_radius)
            ax.set_ylim(0.0, 2.5 * mean_velocity)
            ax.set_xlabel("Y")
            ax.set_ylabel("axial velocity")
            spu.save_and_clear_fig(
                fig,
                ax,
                file_name="snap_" + str("%0.4d" % (flow_sim.time * 100)) + ".png",
            )

        dt = flow_sim.compute_stable_timestep(dt_prefac=0.5)

        # compute flow forcing and timestep forcing
        cylinder_flow_interactor.time_step(dt=dt)
        cylinder_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt=dt, free_stream_velocity=velocity_free_stream)

        # update timers
        foto_timer += dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=10
    )


if __name__ == "__main__":
    # in order Z, Y, X
    grid_size = (64, 64, 128)
    flow_through_circular_pipe_case(grid_size=grid_size, save_data=False)

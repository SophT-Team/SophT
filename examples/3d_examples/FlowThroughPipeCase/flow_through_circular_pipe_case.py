import elastica as ea
import numpy as np
import os
from sopht.utils.IO import IO
from sopht.utils.precision import get_real_t
import sopht_simulator as sps


def analytical_pipe_flow_velocity(radial_coordinate, mean_velocity, pipe_radius):
    return 2.0 * mean_velocity * (1.0 - (radial_coordinate / pipe_radius) ** 2)


def flow_through_circular_pipe_case(
    grid_size,
    coupling_stiffness=-1e2,
    coupling_damping=-6e-2,
    num_threads=4,
    precision="single",
    save_data=False,
):
    """
    This example considers the case of steady flow through a pipe in 3D.
    """
    dim = 3
    real_t = get_real_t(precision)
    x_range = 1.0
    nu = 1e-2
    CFL = 0.1
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        CFL=CFL,
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
    X_cm = boundary_offset
    Y_cm = 0.5 * flow_sim.y_range
    Z_cm = 0.5 * flow_sim.z_range
    start = np.array([X_cm, Y_cm, Z_cm])
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
        grid_dim=dim,
        real_t=real_t,
        forcing_grid_cls=sps.OpenEndCircularCylinderForcingGrid,
        num_forcing_points_along_length=num_forcing_points_along_length,
    )
    # ==================FLOW-BODY COMMUNICATOR SETUP END======

    if save_data:
        # setup IO
        # TODO internalise this in flow simulator as dump_fields
        io_origin = np.array(
            [flow_sim.z_grid.min(), flow_sim.y_grid.min(), flow_sim.x_grid.min()]
        )
        io_dx = flow_sim.dx * np.ones(dim)
        io_grid_size = np.array(grid_size)
        io = IO(dim=dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field, velocity=flow_sim.velocity_field
        )

    t = 0.0
    t_end = 1.0
    foto_timer = 0.0
    foto_timer_limit = t_end / 40

    # create fig for plotting flow fields
    fig, ax = sps.create_figure_and_axes(fig_aspect_ratio="default")
    x_axis = 0
    grid_size_z, grid_size_y, grid_size_x = flow_sim.grid_size
    radial_coordinate = flow_sim.y_grid[grid_size_z // 2, ..., grid_size_x // 2] - Y_cm
    anal_velocity_profile = analytical_pipe_flow_velocity(
        radial_coordinate=radial_coordinate,
        mean_velocity=mean_velocity,
        pipe_radius=cyl_radius,
    )

    # iterate
    while t < t_end:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            print(
                f"time: {t:.2f} ({(t/t_end*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}"
            )
            if save_data:
                io.save(
                    h5_file_name="sopht_" + str("%0.4d" % (t * 100)) + ".h5", time=t
                )
            # midplane along X
            sim_velocity_profile = 0.5 * np.sum(
                flow_sim.velocity_field[
                    x_axis,
                    grid_size_z // 2 - 1 : grid_size_z // 2 + 1,
                    ...,
                    grid_size_x // 2,
                ],
                axis=0,
            )
            ax.plot(radial_coordinate, sim_velocity_profile, label="numerical")
            ax.plot(radial_coordinate, anal_velocity_profile, label="analytical")
            ax.legend()
            ax.set_xlim([-cyl_radius, cyl_radius])
            ax.set_ylim([0.0, 2.5 * mean_velocity])
            ax.set_xlabel("Y")
            ax.set_ylabel("axial velocity")
            sps.save_and_clear_fig(
                fig, ax, file_name="snap_" + str("%0.4d" % (t * 100)) + ".png"
            )

        dt = flow_sim.compute_stable_timestep(dt_prefac=0.5)

        # compute flow forcing and timestep forcing
        cylinder_flow_interactor.time_step(dt=dt)
        cylinder_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt=dt, free_stream_velocity=velocity_free_stream)

        # update timers
        t = t + dt
        foto_timer += dt

    os.system("rm -f flow.mp4")
    os.system(
        "ffmpeg -r 10 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
        "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
        " flow.mp4"
    )
    os.system("rm -f snap*.png")


if __name__ == "__main__":
    # in order Z, Y, X
    grid_size = (64, 64, 128)
    flow_through_circular_pipe_case(grid_size=grid_size, save_data=False)

from hill_sphere_vortex_helpers import HillSphereVortex
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu


def hill_sphere_vortex_case(
    grid_size: tuple[int, int, int],
    num_threads: int = 4,
    precision: str = "single",
    save_data: bool = False,
) -> None:
    """
    This test case considers the Hill's spherical vortex, and tests
    the velocity recovery and vortex stretching steps, for the solver,
    by comparing against analytical expressions.
    """
    grid_dim = 3
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    # Consider a 1 by 1 by 1 3D domain
    x_range = 1.0
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=0.0,
        flow_type="navier_stokes",
        real_t=real_t,
        num_threads=num_threads,
        navier_stokes_inertial_term_form="advection_stretching_split",
    )
    # init vortex at domain center
    vortex_origin = (
        flow_sim.x_range / 2.0,
        flow_sim.y_range / 2.0,
        flow_sim.z_range / 2.0,
    )
    free_stream_velocity = 1.0
    vortex_radius = 0.25 * flow_sim.x_range
    hill_sphere_vortex = HillSphereVortex(
        free_stream_velocity, vortex_radius, vortex_origin
    )
    flow_sim.vorticity_field[...] = hill_sphere_vortex.get_vorticity(
        x_grid=flow_sim.position_field[x_axis_idx],
        y_grid=flow_sim.position_field[y_axis_idx],
        z_grid=flow_sim.position_field[z_axis_idx],
    )
    analytical_velocity_field = hill_sphere_vortex.get_velocity(
        x_grid=flow_sim.position_field[x_axis_idx],
        y_grid=flow_sim.position_field[y_axis_idx],
        z_grid=flow_sim.position_field[z_axis_idx],
    )
    flow_sim.compute_flow_velocity(free_stream_velocity=np.zeros(grid_dim))
    numerical_kinetic_energy = (
        0.5 * np.sum(np.square(flow_sim.velocity_field)) * flow_sim.dx**3
    )
    analytical_kinetic_energy = hill_sphere_vortex.get_kinetic_energy()
    kinetic_energy_error = (
        np.fabs(analytical_kinetic_energy - numerical_kinetic_energy)
        / analytical_kinetic_energy
    )
    print(f"kinetic energy error (%): {kinetic_energy_error*100}")
    vel_error_field = (flow_sim.velocity_field - analytical_velocity_field) / np.amax(
        np.fabs(analytical_velocity_field)
    )
    l2_norm_error = np.linalg.norm(vel_error_field) * flow_sim.dx**1.5
    linf_norm_error = np.amax(np.fabs(vel_error_field))
    print(f"Velocity error (%): L2: {l2_norm_error*100}, Linf: {linf_norm_error*100}")

    # check centerline velocity (axial velocity along Z axis)
    centerline_z = flow_sim.position_field[
        z_axis_idx, ..., grid_size_y // 2, grid_size_x // 2
    ]
    sim_centerline_vel_z = flow_sim.velocity_field[
        z_axis_idx, ..., grid_size_y // 2, grid_size_x // 2
    ]
    anal_centerline_vel_z = analytical_velocity_field[
        z_axis_idx, ..., grid_size_y // 2, grid_size_x // 2
    ]
    fig, ax = spu.create_figure_and_axes(fig_aspect_ratio="default")
    ax.plot(centerline_z, sim_centerline_vel_z, label="numerical")
    ax.plot(centerline_z, anal_centerline_vel_z, label="analytical")
    ax.legend()
    ax.set_xlabel("Z")
    ax.set_ylabel("Axial velocity")
    fig.savefig("centerline_axial_velocity.png")

    # check radial velocity (offset midplane)
    midplane_r = (
        flow_sim.position_field[y_axis_idx, grid_size_z // 3, ..., grid_size_x // 2]
        - vortex_origin[y_axis_idx]
    )
    sim_midplane_vel_r = flow_sim.velocity_field[
        y_axis_idx, grid_size_z // 3, ..., grid_size_x // 2
    ]
    anal_midplane_vel_r = analytical_velocity_field[
        y_axis_idx, grid_size_z // 3, ..., grid_size_x // 2
    ]
    fig2, ax2 = spu.create_figure_and_axes(fig_aspect_ratio="default")
    ax2.plot(midplane_r, sim_midplane_vel_r, label="numerical")
    ax2.plot(midplane_r, anal_midplane_vel_r, label="analytical")
    ax2.legend()
    ax2.set_xlabel("R")
    ax2.set_ylabel("Radial velocity")
    fig2.savefig("midplane_radial_velocity.png")

    # check the vorticity stretching term
    _, _, _, _, sphere_r_grid = hill_sphere_vortex.compute_local_coordinates(
        flow_sim.position_field[x_axis_idx],
        flow_sim.position_field[y_axis_idx],
        flow_sim.position_field[z_axis_idx],
    )

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
            vorticity=flow_sim.vorticity_field,
            stream_func=flow_sim.stream_func_field,
            velocity=flow_sim.velocity_field,
        )
        io.save(h5_file_name="sopht.h5")


if __name__ == "__main__":
    sim_grid_size = (128, 128, 128)
    hill_sphere_vortex_case(grid_size=sim_grid_size)

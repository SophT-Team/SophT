from hill_sphere_vortex_helpers import HillSphereVortex
import numpy as np
from sopht.utils.IO import IO
from sopht.utils.precision import get_real_t
import sopht_simulator as sps


def hill_sphere_vortex_case(
    grid_size,
    num_threads=4,
    precision="single",
    save_data=False,
):
    """
    This test case considers the Hill's spherical vortex, and tests
    the velocity recovery and vortex stretching steps, for the solver,
    by comparing against analytical expressions.
    """
    dim = 3
    real_t = get_real_t(precision)
    # Consider a 1 by 1 3D domain
    grid_size_x = grid_size
    grid_size_y = grid_size_x
    grid_size_z = grid_size_x
    x_range = 1.0
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=(grid_size_z, grid_size_y, grid_size_x),
        x_range=x_range,
        kinematic_viscosity=0.0,
        flow_type="navier_stokes",
        real_t=real_t,
        num_threads=num_threads,
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
        x_grid=flow_sim.x_grid, y_grid=flow_sim.y_grid, z_grid=flow_sim.z_grid
    )
    analytical_velocity_field = hill_sphere_vortex.get_velocity(
        x_grid=flow_sim.x_grid, y_grid=flow_sim.y_grid, z_grid=flow_sim.z_grid
    )
    flow_sim.compute_velocity_from_vorticity()
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
    z_axis = 2
    centerline_z = flow_sim.z_grid[..., grid_size_y // 2, grid_size_x // 2]
    sim_centerline_vel_z = flow_sim.velocity_field[
        z_axis, ..., grid_size_y // 2, grid_size_x // 2
    ]
    anal_centerline_vel_z = analytical_velocity_field[
        z_axis, ..., grid_size_y // 2, grid_size_x // 2
    ]
    fig, ax = sps.create_figure_and_axes(fig_aspect_ratio="default")
    ax.plot(centerline_z, sim_centerline_vel_z, label="numerical")
    ax.plot(centerline_z, anal_centerline_vel_z, label="analytical")
    ax.legend()
    ax.set_xlabel("Z")
    ax.set_ylabel("Axial velocity")
    fig.savefig("centerline_axial_velocity.png")

    # check radial velocity (offset midplane)
    y_axis = 1
    midplane_r = (
        flow_sim.y_grid[grid_size_z // 3, ..., grid_size_x // 2] - vortex_origin[y_axis]
    )
    sim_midplane_vel_r = flow_sim.velocity_field[
        y_axis, grid_size_z // 3, ..., grid_size_x // 2
    ]
    anal_midplane_vel_r = analytical_velocity_field[
        y_axis, grid_size_z // 3, ..., grid_size_x // 2
    ]
    fig2, ax2 = sps.create_figure_and_axes(fig_aspect_ratio="default")
    ax2.plot(midplane_r, sim_midplane_vel_r, label="numerical")
    ax2.plot(midplane_r, anal_midplane_vel_r, label="analytical")
    ax2.legend()
    ax2.set_xlabel("R")
    ax2.set_ylabel("Radial velocity")
    fig2.savefig("midplane_radial_velocity.png")

    # check the vorticity stretching term
    numerical_vortex_stretching = np.zeros_like(flow_sim.vorticity_field)
    flow_sim.vorticity_stretching_timestep(
        vorticity_field=flow_sim.vorticity_field.copy(),
        velocity_field=flow_sim.velocity_field,
        vorticity_stretching_flux_field=numerical_vortex_stretching,
        dt_by_2_dx=flow_sim.real_t(1.0 / (2 * flow_sim.dx)),
    )
    analytical_vortex_stretching = hill_sphere_vortex.get_vortex_stretching(
        x_grid=flow_sim.x_grid, y_grid=flow_sim.y_grid, z_grid=flow_sim.z_grid
    )
    _, _, _, _, sphere_r_grid = hill_sphere_vortex.compute_local_coordinates(
        flow_sim.x_grid, flow_sim.y_grid, flow_sim.z_grid
    )
    # the gradient is discontinous at the boundaries, hence interior
    interior_vortex = sphere_r_grid < 0.9 * hill_sphere_vortex.vortex_radius
    vortex_stretching_error = (
        interior_vortex
        * (numerical_vortex_stretching - analytical_vortex_stretching)
        / np.amax(np.fabs(analytical_vortex_stretching))
    )
    l2_norm_error = np.linalg.norm(vortex_stretching_error) * flow_sim.dx**1.5
    linf_norm_error = np.amax(np.fabs(vortex_stretching_error))
    print(
        f"Vortex stretching error (%): L2: {l2_norm_error*100}, Linf: {linf_norm_error*100}"
    )

    x_axis = 0
    sim_midplane_vortex_stretching = numerical_vortex_stretching[
        x_axis, grid_size_z // 3, ..., grid_size_x // 2
    ]
    anal_midplane_vortex_stretching = analytical_vortex_stretching[
        x_axis, grid_size_z // 3, ..., grid_size_x // 2
    ]
    fig3, ax3 = sps.create_figure_and_axes(fig_aspect_ratio="default")
    ax3.plot(midplane_r, sim_midplane_vortex_stretching, label="numerical")
    ax3.plot(midplane_r, anal_midplane_vortex_stretching, label="analytical")
    ax3.legend()
    ax3.set_xlabel("R")
    ax3.set_ylabel("Vorticity stretching")
    fig3.savefig("midplane_vorticity_stretching.png")

    if save_data:
        # setup IO
        # TODO internalise this in flow simulator as dump_fields
        io_origin = np.array(
            [flow_sim.z_grid.min(), flow_sim.y_grid.min(), flow_sim.x_grid.min()]
        )
        io_dx = flow_sim.dx * np.ones(dim)
        io_grid_size = np.array([grid_size_z, grid_size_y, grid_size_x])
        io = IO(dim=dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field,
            stream_func=flow_sim.stream_func_field,
            velocity=flow_sim.velocity_field,
        )
        io.save(h5_file_name="sopht.h5")


if __name__ == "__main__":
    grid_size = 128
    hill_sphere_vortex_case(grid_size=grid_size)

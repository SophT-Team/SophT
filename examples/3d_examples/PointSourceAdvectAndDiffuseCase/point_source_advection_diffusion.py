from point_source_helpers import compute_diffused_point_source_field
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu


def point_source_advection_diffusion_case(
    grid_size: tuple[int, int, int],
    num_threads: int = 4,
    precision: str = "single",
    save_data: bool = False,
) -> None:
    """
    This example considers a simple case of point source vortex, advecting with a
    constant velocity in 3D, while it diffuses in time.
    """
    grid_dim = 3
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    # Consider a 1 by 1 3D domain
    x_range = 1.0
    nu = 1e-3
    # start with non-zero to avoid singularity in point source
    t_start = 5.0
    t_end = 5.4
    # init vortex at (0.3 0.3, 0.3)
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="passive_vector",
        real_t=real_t,
        num_threads=num_threads,
        time=t_start,
    )
    x_cm_start = 0.3
    y_cm_start = x_cm_start
    z_cm_start = x_cm_start
    # to start with point source magnitude = 1
    point_mag = 4.0 * np.pi * nu * t_start**1.5
    vorticity_field = flow_sim.primary_vector_field.view()
    for i in range(grid_dim):
        vorticity_field[i] = compute_diffused_point_source_field(
            x_grid=flow_sim.position_field[x_axis_idx],
            y_grid=flow_sim.position_field[y_axis_idx],
            z_grid=flow_sim.position_field[z_axis_idx],
            x_grid_cm=x_cm_start,
            y_grid_cm=y_cm_start,
            z_grid_cm=z_cm_start,
            nu=nu,
            point_mag=point_mag,
            t=t_start,
            real_dtype=real_t,
        )

    # Initialize velocity = c in X, Y and Z direction
    velocity_free_stream = 1.0
    flow_sim.velocity_field[...] = velocity_free_stream

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
        io_grid_size = np.array([grid_size_z, grid_size_y, grid_size_x])
        io = spu.IO(dim=grid_dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(vorticity=vorticity_field)

    foto_timer = 0.0
    foto_timer_limit = (t_end - t_start) / 20

    # iterate
    while flow_sim.time < t_end:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            print(
                f"time: {flow_sim.time:.2f} ({((flow_sim.time-t_start)/(t_end-t_start)*100):2.1f}%), "
                f"max_vort: {np.amax(vorticity_field):.4f}"
            )
            if save_data:
                io.save(
                    h5_file_name="sopht_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )

        dt = flow_sim.compute_stable_timestep()
        flow_sim.time_step(dt=dt)

        foto_timer += dt

    # final vortex computation
    t_end = flow_sim.time
    x_cm_final = x_cm_start + velocity_free_stream * (t_end - t_start)
    y_cm_final = y_cm_start + velocity_free_stream * (t_end - t_start)
    z_cm_final = z_cm_start + velocity_free_stream * (t_end - t_start)
    final_analytical_vorticity_field = np.zeros_like(vorticity_field)
    for i in range(grid_dim):
        final_analytical_vorticity_field[i] = compute_diffused_point_source_field(
            x_grid=flow_sim.position_field[x_axis_idx],
            y_grid=flow_sim.position_field[y_axis_idx],
            z_grid=flow_sim.position_field[z_axis_idx],
            x_grid_cm=x_cm_final,
            y_grid_cm=y_cm_final,
            z_grid_cm=z_cm_final,
            nu=nu,
            point_mag=point_mag,
            t=t_start,
            real_dtype=real_t,
        )

    # check error
    error_field = np.fabs(vorticity_field - final_analytical_vorticity_field)
    l2_error = np.linalg.norm(error_field) * (flow_sim.dx**1.5)
    linf_error = np.amax(error_field)
    print(f"Final vortex center location: ({x_cm_final}, {y_cm_final}, {z_cm_final})")
    print(f"vorticity L2 error: {l2_error}")
    print(f"vorticity Linf error: {linf_error}")


if __name__ == "__main__":
    sim_grid_size = (128, 128, 128)
    point_source_advection_diffusion_case(grid_size=sim_grid_size, save_data=True)

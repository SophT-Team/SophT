from point_source_helpers import compute_diffused_point_source_field
import numpy as np
from sopht.utils.IO import IO
from sopht.utils.precision import get_real_t
import sopht_simulator as sps


def point_source_advection_diffusion_case(
    grid_size, num_threads=4, precision="single", save_data=False
):
    """
    This example considers a simple case of point source vortex, advecting with a
    constant velocity in 3D, while it diffuses in time.
    """
    dim = 3
    real_t = get_real_t(precision)
    # Consider a 1 by 1 3D domain
    grid_size_x = grid_size
    grid_size_y = grid_size_x
    grid_size_z = grid_size_x
    x_range = 1.0
    nu = real_t(1e-3)
    CFL = real_t(0.05)
    # init vortex at (0.3 0.3, 0.3)
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=(grid_size_z, grid_size_y, grid_size_x),
        x_range=x_range,
        kinematic_viscosity=nu,
        CFL=CFL,
        flow_type="passive_vector",
        real_t=real_t,
        num_threads=num_threads,
    )
    x_cm_start = real_t(0.3)
    y_cm_start = x_cm_start
    z_cm_start = x_cm_start
    # start with non-zero to avoid singularity in point source
    t_start = real_t(5)
    t_end = real_t(5.4)
    # to start with point source magnitude = 1
    point_mag = real_t(4 * np.pi * nu * t_start) ** (3 / 2)
    vorticity_field = flow_sim.primary_vector_field.view()
    for i in range(dim):
        vorticity_field[i] = compute_diffused_point_source_field(
            x_grid=flow_sim.x_grid,
            y_grid=flow_sim.y_grid,
            z_grid=flow_sim.z_grid,
            x_grid_cm=x_cm_start,
            y_grid_cm=y_cm_start,
            z_grid_cm=z_cm_start,
            nu=nu,
            point_mag=point_mag,
            t=t_start,
            real_dtype=real_t,
        )

    # Initialize velocity = c in X, Y and Z direction
    velocity_free_stream = real_t(1.0)
    flow_sim.velocity_field[...] = velocity_free_stream

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
        io.add_as_eulerian_fields_for_io(vorticity=vorticity_field)

    t = t_start
    foto_timer = 0.0
    foto_timer_limit = (t_end - t_start) / 20

    # iterate
    while t < t_end:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            print(
                f"time: {t:.2f} ({((t-t_start)/(t_end-t_start)*100):2.1f}%), "
                f"max_vort: {np.amax(vorticity_field):.4f}"
            )
            if save_data:
                io.save(
                    h5_file_name="sopht_" + str("%0.4d" % (t * 100)) + ".h5", time=t
                )

        dt = flow_sim.compute_stable_timestep()
        flow_sim.time_step(dt=dt)

        t = t + dt
        foto_timer += dt

    # final vortex computation
    t_end = t
    x_cm_final = x_cm_start + velocity_free_stream * (t_end - t_start)
    y_cm_final = y_cm_start + velocity_free_stream * (t_end - t_start)
    z_cm_final = z_cm_start + velocity_free_stream * (t_end - t_start)
    final_analytical_vorticity_field = np.zeros_like(vorticity_field)
    for i in range(dim):
        final_analytical_vorticity_field[i] = compute_diffused_point_source_field(
            x_grid=flow_sim.x_grid,
            y_grid=flow_sim.y_grid,
            z_grid=flow_sim.z_grid,
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
    grid_size = 128
    point_source_advection_diffusion_case(grid_size=grid_size, save_data=True)

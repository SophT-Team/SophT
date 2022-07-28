from flow_algo_assembly.flow_solver_steps import (
    gen_advection_diffusion_timestep_kernel_3d,
)
from flow_algo_assembly.timestep_limits import compute_advection_diffusion_timestep

from point_source_helpers import compute_diffused_point_source_field

import numpy as np

from sopht.utils.IO import IO
from sopht.utils.precision import get_real_t


def point_source_advection_diffusion_case(
    grid_size_x, num_threads=4, precision="single", save_data=False
):
    """
    This example considers a simple case of point source vortex, advecting with a
    constant velocity in 3D, while it diffuses in time.
    """
    dim = 3
    real_t = get_real_t(precision)
    # Consider a 1 by 1 3D domain
    grid_size_y = grid_size_x
    grid_size_z = grid_size_x
    dx = real_t(1.0 / grid_size_x)
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, grid_size_x).astype(real_t)
    z_grid, y_grid, x_grid = np.meshgrid(x, x, x, indexing="ij")
    nu = real_t(1e-3)
    CFL = real_t(0.05)
    # init vortex at (0.3 0.3, 0.3)
    x_cm_start = real_t(0.3)
    y_cm_start = x_cm_start
    z_cm_start = x_cm_start
    # start with non-zero to avoid singularity in point source
    t_start = real_t(5)
    t_end = real_t(5.4)
    # to start with point source magnitude = 1
    point_mag = real_t(4 * np.pi * nu * t_start) ** (3 / 2)
    vorticity_field = np.zeros(
        (dim, grid_size_z, grid_size_y, grid_size_x), dtype=real_t
    )
    for i in range(dim):
        vorticity_field[i] = compute_diffused_point_source_field(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
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
    velocity_field = velocity_free_stream * np.ones(
        (dim, grid_size_z, grid_size_y, grid_size_x), dtype=real_t
    )
    # Initialize buffer for advection and diffusion kernels (need only (n, n, n))
    buffer_field = np.zeros_like(vorticity_field[0])

    # compile kernel
    advection_and_diffusion_timestep = gen_advection_diffusion_timestep_kernel_3d(
        real_t,
        (grid_size_z, grid_size_y, grid_size_x),
        dx,
        nu,
        num_threads,
        field_type="vector",
    )

    if save_data:
        # setup IO
        io_origin = np.array([z_grid.min(), y_grid.min(), x_grid.min()])
        io_dx = dx * np.ones(dim)
        io_grid_size = np.array([grid_size_z, grid_size_y, grid_size_x])
        io = IO(dim=dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(vorticity=vorticity_field)

    # iterate
    t = t_start
    if save_data:
        foto_timer = 0.0
        foto_timer_limit = (t_end - t_start) / 20
    while t < t_end:
        # Save data
        if save_data and (foto_timer >= foto_timer_limit or foto_timer == 0):
            foto_timer = 0.0
            io.save(h5_file_name="sopht_" + str("%0.4d" % (t * 100)) + ".h5", time=t)

        # compute timestep and update time
        dt = compute_advection_diffusion_timestep(
            velocity_field=velocity_field, CFL=CFL, nu=nu, dx=dx
        )
        t = t + dt
        if save_data:
            foto_timer += dt

        # advect and diffuse vorticity
        advection_and_diffusion_timestep(
            field=vorticity_field,
            velocity_field=velocity_field,
            flux_buffer=buffer_field,
            dt=dt,
        )

    # final vortex computation
    t_end = t
    x_cm_final = x_cm_start + velocity_free_stream * (t_end - t_start)
    y_cm_final = y_cm_start + velocity_free_stream * (t_end - t_start)
    z_cm_final = z_cm_start + velocity_free_stream * (t_end - t_start)
    final_analytical_vorticity_field = np.zeros_like(vorticity_field)
    for i in range(dim):
        final_analytical_vorticity_field[i] = compute_diffused_point_source_field(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
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
    l2_error = np.linalg.norm(error_field) * (dx**1.5)
    linf_error = np.amax(error_field)
    print(f"Final vortex center location: ({x_cm_final}, {y_cm_final}, {z_cm_final})")
    print(f"vorticity L2 error: {l2_error}")
    print(f"vorticity Linf error: {linf_error}")


if __name__ == "__main__":
    grid_size_x = 128
    point_source_advection_diffusion_case(grid_size_x=grid_size_x, save_data=True)

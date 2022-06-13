from flow_algo_assembly.flow_solver_steps import (
    gen_compute_velocity_from_vorticity_kernel_2d,
    gen_advection_diffusion_timestep_kernel_2d,
)
from flow_algo_assembly.timestep_limits import compute_advection_diffusion_timestep

import numpy as np

import os

from lamb_oseen_helpers import compute_lamb_oseen_velocity, compute_lamb_oseen_vorticity

from plot_utils.lab_cmap import lab_cmap

from sopht.numeric.eulerian_grid_ops import (
    gen_add_fixed_val_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t


def lamb_oseen_vortex_flow_case(
    grid_size, num_threads=4, precision="single", save_snap=False
):
    """
    This example considers a simple case of Lamb-Oseen vortex, advecting with a
    constant velocity in 2D, while it diffuses in time, and involves solving
    the Navier-Stokes equation.
    """
    real_t = get_real_t(precision)
    # Consider a 1 by 1 2D domain
    dx = real_t(1.0 / grid_size)
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, grid_size).astype(real_t)
    x_grid, y_grid = np.meshgrid(x, x)
    nu = real_t(1e-3)
    CFL = real_t(0.1)
    # init vortex at (0.3 0.3)
    x_cm_start = real_t(0.3)
    y_cm_start = x_cm_start
    # start with non-zero to avoid singularity in Lamb-Oseen
    t_start = real_t(1)
    t_end = real_t(1.4)
    # to start with max circulation = 1
    gamma = real_t(4 * np.pi * nu * t_start)

    vorticity_field = compute_lamb_oseen_vorticity(
        x=x_grid,
        y=y_grid,
        x_cm=x_cm_start,
        y_cm=y_cm_start,
        nu=nu,
        gamma=gamma,
        t=t_start,
        real_t=real_t,
    )

    # Initialize velocity free stream magnitude in X and Y direction
    velocity_free_stream = np.ones(2, dtype=real_t)
    velocity_field = np.zeros((2, grid_size, grid_size), dtype=real_t)
    # Initialize buffer for advection, diffusion and velocity recovery kernels
    buffer_scalar_field = np.zeros_like(vorticity_field)

    # compile kernels
    add_fixed_val = gen_add_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(grid_size, grid_size),
        num_threads=num_threads,
        field_type="vector",
    )

    compute_velocity_from_vorticity_kernel_2d = (
        gen_compute_velocity_from_vorticity_kernel_2d(
            real_t, grid_size, dx, num_threads
        )
    )

    advection_and_diffusion_timestep = gen_advection_diffusion_timestep_kernel_2d(
        real_t, grid_size, dx, nu, num_threads
    )

    # iterate
    t = t_start
    if save_snap:
        foto_timer = 0.0
        foto_timer_limit = (t_end - t_start) / 50
        import matplotlib.pyplot as plt

        plt.style.use("seaborn")
    while t < t_end:
        # compute velocity from vorticity
        compute_velocity_from_vorticity_kernel_2d(
            velocity_field=velocity_field,
            vorticity_field=vorticity_field,
            stream_func_field=buffer_scalar_field,
        )

        # add freestream
        add_fixed_val(
            sum_field=velocity_field,
            vector_field=velocity_field,
            fixed_vals=velocity_free_stream,
        )

        # compute timestep and update time
        dt = compute_advection_diffusion_timestep(
            velocity_field=velocity_field, CFL=CFL, nu=nu, dx=dx
        )
        t = t + dt

        # Plot solution
        if save_snap:
            if foto_timer >= foto_timer_limit or foto_timer == 0:
                foto_timer = 0.0
                fig = plt.figure(frameon=True, dpi=150)
                ax = fig.add_subplot(111)
                plt.contourf(
                    x_grid,
                    y_grid,
                    vorticity_field,
                    levels=np.linspace(0, 1, 50),
                    extend="both",
                    cmap=lab_cmap,
                )
                plt.colorbar()
                ax.set_aspect(aspect=1)
                plt.savefig(
                    # os.path.join(SAVE_PATH, "snap_" + str("%0.4d" % (t * 100)) + ".png"),
                    "snap_" + str("%0.4d" % (t * 100)) + ".png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.clf()
                plt.close("all")
            foto_timer += dt

        # advect and diffuse vorticity
        advection_and_diffusion_timestep(
            field=vorticity_field,
            velocity_field=velocity_field,
            flux_buffer=buffer_scalar_field,
            dt=dt,
        )

    # final vortex computation
    t_end = t
    x_cm_final = x_cm_start + velocity_free_stream[0] * (t_end - t_start)
    y_cm_final = y_cm_start + velocity_free_stream[1] * (t_end - t_start)
    final_analytical_vorticity_field = compute_lamb_oseen_vorticity(
        x=x_grid,
        y=y_grid,
        x_cm=x_cm_final,
        y_cm=y_cm_final,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )

    # compile video
    if save_snap:
        os.system("rm -f flow.mp4")
        os.system(
            "ffmpeg -r 16 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
            "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
            " flow.mp4"
        )
        os.system("rm -f snap*.png")

    # check error
    error_field = np.fabs(vorticity_field - final_analytical_vorticity_field)
    l2_error = np.linalg.norm(error_field) * dx
    linf_error = np.amax(error_field)
    print(f"Final vortex center location: ({x_cm_final}, {y_cm_final})")
    print(f"vorticity L2 error: {l2_error}")
    print(f"vorticity Linf error: {linf_error}")
    final_analytical_velocity_field = compute_lamb_oseen_velocity(
        x=x_grid,
        y=y_grid,
        x_cm=x_cm_final,
        y_cm=y_cm_final,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )
    compute_velocity_from_vorticity_kernel_2d(
        velocity_field=velocity_field,
        vorticity_field=vorticity_field,
        stream_func_field=buffer_scalar_field,
    )
    error_field = np.fabs(velocity_field - final_analytical_velocity_field)
    l2_error = np.linalg.norm(error_field) * dx
    linf_error = np.amax(error_field)
    print(f"velocity L2 error: {l2_error}")
    print(f"velocity Linf error: {linf_error}")


if __name__ == "__main__":
    grid_size = 256
    lamb_oseen_vortex_flow_case(grid_size=grid_size, save_snap=True)

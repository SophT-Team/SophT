# from sopht_simulator.flow.FlowSimulator2D import UnboundedFlowSimulator2D

from lamb_oseen_helpers import compute_lamb_oseen_velocity, compute_lamb_oseen_vorticity

import matplotlib.pyplot as plt

import numpy as np

import os

# from sopht_simulator.plot_utils.lab_cmap import lab_cmap
from sopht_simulator import *

from sopht.numeric.eulerian_grid_ops import (
    gen_add_fixed_val_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t


def lamb_oseen_vortex_flow_case(grid_size_x, num_threads=4, precision="single"):
    """
    This example considers a simple case of Lamb-Oseen vortex, advecting with a
    constant velocity in 2D, while it diffuses in time, and involves solving
    the Navier-Stokes equation.
    """
    real_t = get_real_t(precision)
    # Consider a 1 by 1 2D domain
    x_range = real_t(1.0)
    grid_size_y = grid_size_x
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

    flow_sim = UnboundedFlowSimulator2D(
        grid_size=(grid_size_y, grid_size_x),
        x_range=x_range,
        kinematic_viscosity=nu,
        CFL=CFL,
        flow_type="navier_stokes",
        real_t=real_t,
        num_threads=num_threads,
    )

    flow_sim.vorticity_field[...] = compute_lamb_oseen_vorticity(
        x=flow_sim.x_grid,
        y=flow_sim.y_grid,
        x_cm=x_cm_start,
        y_cm=y_cm_start,
        nu=nu,
        gamma=gamma,
        t=t_start,
        real_t=real_t,
    )

    # Initialize velocity free stream magnitude in X and Y direction
    velocity_free_stream = np.ones(2, dtype=real_t)
    flow_sim.velocity_field[...] = compute_lamb_oseen_velocity(
        x=flow_sim.x_grid,
        y=flow_sim.y_grid,
        x_cm=x_cm_start,
        y_cm=y_cm_start,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )
    # compile additional kernels
    # TODO push in flow sim!
    add_fixed_val = gen_add_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(grid_size_y, grid_size_x),
        num_threads=num_threads,
        field_type="vector",
    )

    # iterate
    t = t_start
    foto_timer = 0.0
    foto_timer_limit = (t_end - t_start) / 25
    plt.style.use("seaborn")

    while t < t_end:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            fig = plt.figure(frameon=True, dpi=150)
            ax = fig.add_subplot(111)
            plt.contourf(
                flow_sim.x_grid,
                flow_sim.y_grid,
                flow_sim.vorticity_field,
                levels=np.linspace(0, 1, 50),
                extend="both",
                cmap=lab_cmap,
            )
            plt.colorbar()
            ax.set_aspect(aspect=1)
            plt.savefig(
                "snap_" + str("%0.4d" % (t * 100)) + ".png",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.clf()
            plt.close("all")
            print(
                f"time: {t:.2f} ({((t-t_start)/(t_end-t_start)*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}"
            )

        dt = flow_sim.compute_stable_timestep()
        flow_sim.time_step(dt=dt)

        # add freestream
        # TODO merge in flow sim
        add_fixed_val(
            sum_field=flow_sim.velocity_field,
            vector_field=flow_sim.velocity_field,
            fixed_vals=velocity_free_stream,
        )

        # update time
        t = t + dt
        foto_timer += dt

    # final vortex computation
    t_end = t
    x_cm_final = x_cm_start + velocity_free_stream[0] * (t_end - t_start)
    y_cm_final = y_cm_start + velocity_free_stream[1] * (t_end - t_start)
    final_analytical_vorticity_field = compute_lamb_oseen_vorticity(
        x=flow_sim.x_grid,
        y=flow_sim.y_grid,
        x_cm=x_cm_final,
        y_cm=y_cm_final,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )

    # compile video
    os.system("rm -f flow.mp4")
    os.system(
        "ffmpeg -r 16 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
        "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
        " flow.mp4"
    )
    os.system("rm -f snap*.png")

    # check error
    error_field = np.fabs(flow_sim.vorticity_field - final_analytical_vorticity_field)
    l2_error = np.linalg.norm(error_field) * flow_sim.dx
    linf_error = np.amax(error_field)
    print(f"Final vortex center location: ({x_cm_final}, {y_cm_final})")
    print(f"vorticity L2 error: {l2_error}")
    print(f"vorticity Linf error: {linf_error}")
    final_analytical_velocity_field = compute_lamb_oseen_velocity(
        x=flow_sim.x_grid,
        y=flow_sim.y_grid,
        x_cm=x_cm_final,
        y_cm=y_cm_final,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )
    flow_sim.compute_velocity_from_vorticity()
    error_field = np.fabs(flow_sim.velocity_field - final_analytical_velocity_field)
    l2_error = np.linalg.norm(error_field) * flow_sim.dx
    linf_error = np.amax(error_field)
    print(f"velocity L2 error: {l2_error}")
    print(f"velocity Linf error: {linf_error}")


if __name__ == "__main__":
    grid_size_x = 256
    lamb_oseen_vortex_flow_case(grid_size_x=grid_size_x)

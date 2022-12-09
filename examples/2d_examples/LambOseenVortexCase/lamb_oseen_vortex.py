from lamb_oseen_helpers import compute_lamb_oseen_velocity, compute_lamb_oseen_vorticity
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu
from typing import Tuple


def lamb_oseen_vortex_flow_case(
    grid_size: Tuple[int, int], num_threads: int = 4, precision: str = "single"
) -> None:
    """
    This example considers a simple case of Lamb-Oseen vortex, advecting with a
    constant velocity in 2D, while it diffuses in time, and involves solving
    the Navier-Stokes equation.
    """
    grid_dim = 2
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    # Consider a 1 by 1 2D domain
    x_range = 1.0
    nu = 1e-3
    # init vortex at (0.3 0.3)
    x_cm_start = 0.3
    y_cm_start = x_cm_start
    # start with non-zero to avoid singularity in Lamb-Oseen
    t_start = 1.0
    t_end = 1.4
    # to start with max circulation = 1
    gamma = 4 * np.pi * nu * t_start
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes",
        with_free_stream_flow=True,
        real_t=real_t,
        num_threads=num_threads,
        time=t_start,
    )

    flow_sim.vorticity_field[...] = compute_lamb_oseen_vorticity(
        x=flow_sim.position_field[x_axis_idx],
        y=flow_sim.position_field[y_axis_idx],
        x_cm=x_cm_start,
        y_cm=y_cm_start,
        nu=nu,
        gamma=gamma,
        t=t_start,
        real_t=real_t,
    )

    # Initialize velocity free stream magnitude in X and Y direction
    velocity_free_stream = np.ones(grid_dim, dtype=real_t)
    flow_sim.velocity_field[...] = compute_lamb_oseen_velocity(
        x=flow_sim.position_field[x_axis_idx],
        y=flow_sim.position_field[y_axis_idx],
        x_cm=x_cm_start,
        y_cm=y_cm_start,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )

    # iterate
    foto_timer = 0.0
    foto_timer_limit = (t_end - t_start) / 25

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()

    while flow_sim.time < t_end:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(f"Vorticity, time: {flow_sim.time:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                flow_sim.vorticity_field,
                levels=np.linspace(0, 1, 50),
                extend="both",
                cmap=spu.get_lab_cmap(),
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            spu.save_and_clear_fig(
                fig,
                ax,
                cbar,
                file_name="snap_" + str("%0.4d" % (flow_sim.time * 100)) + ".png",
            )
            print(
                f"time: {flow_sim.time:.2f} ({((flow_sim.time-t_start)/(t_end-t_start)*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}"
            )

        dt = flow_sim.compute_stable_timestep()
        flow_sim.time_step(dt=dt, free_stream_velocity=velocity_free_stream)

        # update timer
        foto_timer += dt

    # final vortex computation
    t_end = flow_sim.time
    x_cm_final = x_cm_start + velocity_free_stream[x_axis_idx] * (t_end - t_start)
    y_cm_final = y_cm_start + velocity_free_stream[y_axis_idx] * (t_end - t_start)
    final_analytical_vorticity_field = compute_lamb_oseen_vorticity(
        x=flow_sim.position_field[x_axis_idx],
        y=flow_sim.position_field[y_axis_idx],
        x_cm=x_cm_final,
        y_cm=y_cm_final,
        nu=nu,
        gamma=gamma,
        t=t_end,
        real_t=real_t,
    )

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=10
    )

    # check error
    error_field = np.fabs(flow_sim.vorticity_field - final_analytical_vorticity_field)
    l2_error = np.linalg.norm(error_field) * flow_sim.dx
    linf_error = np.amax(error_field)
    print(f"Final vortex center location: ({x_cm_final}, {y_cm_final})")
    print(f"vorticity L2 error: {l2_error}")
    print(f"vorticity Linf error: {linf_error}")
    final_analytical_velocity_field = compute_lamb_oseen_velocity(
        x=flow_sim.position_field[x_axis_idx],
        y=flow_sim.position_field[y_axis_idx],
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
    lamb_oseen_vortex_flow_case(grid_size=(256, 256))

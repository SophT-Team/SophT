import matplotlib.pyplot as plt
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu
from lid_driven_cavity_grid import LidDrivenCavityForcingGrid


def lid_driven_cavity_case(
    grid_size: tuple[int, int],
    reynolds: float = 100.0,
    num_threads: int = 4,
    coupling_stiffness: float = -5e4,
    coupling_damping: float = -20,
    precision: str = "single",
    save_diagnostic: bool = False,
) -> None:
    """
    This example considers a lid driven cavity flow using immersed
    boundary forcing.
    """
    grid_dim = 2
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    # Flow parameters
    lid_velocity = 1.0
    ldc_side_length = 0.6
    nu = ldc_side_length * lid_velocity / reynolds
    x_range = 1.0
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        num_threads=num_threads,
    )

    # Initialize lid driven cavity forcing grid
    num_lag_nodes_per_side = grid_size[x_axis_idx] * 3 // 8
    ldc_com = (0.5, 0.5)
    ldc_flow_interactor = sps.ImmersedBodyFlowInteraction(
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        body_flow_forces=np.zeros((grid_dim, 1)),
        body_flow_torques=np.zeros((grid_dim, 1)),
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        forcing_grid_cls=LidDrivenCavityForcingGrid,
        num_lag_nodes_per_side=num_lag_nodes_per_side,
        side_length=ldc_side_length,
        lid_velocity=lid_velocity,
        cavity_com=ldc_com,
    )
    ldc_mask = (
        np.fabs(flow_sim.position_field[x_axis_idx] - ldc_com[x_axis_idx])
        < 0.5 * ldc_side_length
    ) * (
        np.fabs(flow_sim.position_field[y_axis_idx] - ldc_com[y_axis_idx])
        < 0.5 * ldc_side_length
    )

    # iterate
    timescale = ldc_side_length / lid_velocity
    t_end_hat = 3.0  # non-dimensional end time
    t_end = t_end_hat * timescale  # dimensional end time
    foto_timer = 0.0
    foto_timer_limit = t_end / 50

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()

    while flow_sim.time < t_end:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(f"U velocity, t_hat: {flow_sim.time / timescale:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                ldc_mask * flow_sim.velocity_field[x_axis_idx],
                levels=np.linspace(-0.5, 0.5, 100),
                extend="both",
                cmap=spu.get_lab_cmap(),
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.scatter(
                ldc_flow_interactor.forcing_grid.position_field[x_axis_idx],
                ldc_flow_interactor.forcing_grid.position_field[y_axis_idx],
                s=10,
                color="k",
            )
            spu.save_and_clear_fig(
                fig,
                ax,
                cbar,
                file_name="snap_" + str("%0.4d" % (flow_sim.time * 100)) + ".png",
            )
            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/t_end*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}"
            )

        dt = flow_sim.compute_stable_timestep()

        # compute flow forcing and timestep forcing
        ldc_flow_interactor.time_step(dt=dt)
        ldc_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt)

        # update timer
        foto_timer += dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=10
    )

    if save_diagnostic:
        plt.figure()
        _, grid_size_x = grid_size
        plt.plot(
            (
                flow_sim.position_field[y_axis_idx, ..., grid_size_x // 2]
                - (0.5 - 0.5 * ldc_side_length)
            )
            / ldc_side_length,
            (ldc_mask * flow_sim.velocity_field)[x_axis_idx, ..., grid_size_x // 2],
        )
        plt.xlim([0.0, 1.0])
        plt.xlabel("Y")
        plt.ylabel("U(Y) at X = 0.5")
        plt.savefig("U_variation_with_Y.png")


if __name__ == "__main__":
    lid_driven_cavity_case(
        grid_size=(256, 256),
        save_diagnostic=True,
    )

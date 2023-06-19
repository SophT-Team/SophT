import elastica as ea
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu


def immersed_continuum_snake_case(
    final_time_by_period: float,
    grid_size: tuple[int, int],
    reynolds: float = 10,
    coupling_type: str = "one_way",
    coupling_stiffness: float = -1.6e4,
    coupling_damping: float = -16,
    num_threads: int = 4,
    precision: str = "single",
) -> None:
    grid_dim = 2
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    # =================PYELASTICA STUFF BEGIN=====================

    class ImmersedContinuumSnakeSimulator(
        ea.BaseSystemCollection, ea.Forcing, ea.Damping
    ):
        pass

    snake_sim = ImmersedContinuumSnakeSimulator()
    # setting up test params
    n_elem = 40
    base_length = 1.0
    start = np.array([base_length, base_length, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_radius = 0.025
    density = 250
    E = 1e7
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    snake_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )

    snake_sim.append(snake_rod)
    period = 1.0
    wave_length = 1.0 * base_length
    spline_coeff = 5.0 * np.array([3.4, 3.3, 4.2, 2.6, 3.6, 3.5])
    # Head and tail control points are zero.
    control_points = np.hstack((0, spline_coeff, 0))
    snake_sim.add_forcing_to(snake_rod).using(
        ea.MuscleTorques,
        base_length=base_length,
        b_coeff=control_points,
        period=period,
        wave_number=2.0 * np.pi / wave_length,
        phase_shift=0.0,
        rest_lengths=snake_rod.rest_lengths,
        ramp_up_time=period,
        direction=normal,
        with_spline=True,
    )

    # add damping
    damping_constant = 1.0
    rod_dt = 0.5e-4 * period
    snake_sim.dampen(snake_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=rod_dt,
    )
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    # 2x1 domain with x_range = 4 * base_length
    x_range = 4 * base_length
    # Flow parameters
    vel_scale = base_length / period
    nu = base_length * vel_scale / reynolds
    flow_sim = sps.UnboundedNavierStokesFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        with_forcing=True,
        real_t=real_t,
        num_threads=num_threads,
    )
    # ==================FLOW SETUP END=========================

    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=snake_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        forcing_grid_cls=sps.CosseratRodElementCentricForcingGrid,
        real_t=real_t,
        num_threads=num_threads,
    )
    if coupling_type == "two_way":
        snake_sim.add_forcing_to(snake_rod).using(
            sps.FlowForces,
            cosserat_rod_flow_interactor,
        )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======

    # =================TIMESTEPPING====================

    snake_sim.finalize()
    timestepper = ea.PositionVerlet()
    do_step, stages_and_updates = ea.extend_stepper_interface(timestepper, snake_sim)
    final_time = period * final_time_by_period
    foto_timer = 0.0
    foto_timer_limit = period / 10

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()

    while flow_sim.time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(f"Vorticity, time: {flow_sim.time:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                flow_sim.vorticity_field,
                levels=np.linspace(-5, 5, 100),
                extend="both",
                cmap=spu.get_lab_cmap(),
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.plot(
                snake_rod.position_collection[x_axis_idx],
                snake_rod.position_collection[y_axis_idx],
                linewidth=3,
                color="k",
            )
            spu.save_and_clear_fig(
                fig,
                ax,
                cbar,
                file_name="snap_" + str("%0.4d" % (flow_sim.time * 100)) + ".png",
            )
            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"snake com: {np.mean(snake_rod.position_collection[x_axis_idx]):.4f}"
                "grid deviation L2 error: "
                f"{cosserat_rod_flow_interactor.get_grid_deviation_error_l2_norm():.6f}"
            )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.25)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            rod_time = do_step(
                timestepper, stages_and_updates, snake_sim, rod_time, local_rod_dt
            )
            # timestep the cosserat_rod_flow_interactor
            cosserat_rod_flow_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and rod
        cosserat_rod_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt=flow_dt)

        # update timers
        foto_timer += flow_dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=10
    )


if __name__ == "__main__":
    sim_grid_size_x = 256
    sim_grid_size_y = sim_grid_size_x // 2
    immersed_continuum_snake_case(
        final_time_by_period=20.0,
        grid_size=(sim_grid_size_y, sim_grid_size_x),
        coupling_type="two_way",
    )

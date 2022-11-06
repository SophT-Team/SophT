import click
import elastica as ea
import matplotlib.pyplot as plt
import numpy as np
from sopht.utils.precision import get_real_t
import sopht_simulator as sps
from sopht.utils.IO import IO


def flow_past_rod_case(
    nondim_final_time,
    grid_size,
    reynolds,
    nondim_bending_stiffness,
    nondim_mass_ratio,
    froude,
    num_threads,
    rod_start_incline_angle=0.0,
    coupling_stiffness=-8e4,
    coupling_damping=-30,
    precision="single",
    save_flow_data=False,
):
    # =================COMMON SIMULATOR STUFF=======================
    velocity_free_stream = 1.0
    rho_f = 1.0
    base_length = 1.0
    x_range = 6.0 * base_length
    x_axis_idx = sps.VectorField.x_axis_idx()
    y_axis_idx = sps.VectorField.y_axis_idx()
    grid_dim = 2
    grid_size_y, grid_size_x = grid_size
    y_range = grid_size_y / grid_size_x * x_range
    # =================PYELASTICA STUFF BEGIN=====================

    class FlowPastRodSimulator(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping
    ):
        pass

    flow_past_sim = FlowPastRodSimulator()
    # setting up test params
    n_elem = grid_size_x // 8
    start = np.array([base_length, 0.501 * y_range, 0.0])
    direction = np.array(
        [np.cos(rod_start_incline_angle), np.sin(rod_start_incline_angle), 0.0]
    )
    normal = np.array([0.0, 0.0, 1.0])
    base_radius = 0.01
    base_area = np.pi * base_radius**2
    z_axis_width = 1.0
    # nondim_mass_ratio = rod_line_density / (rho_f * base_length * z_axis_width)
    rod_line_density = nondim_mass_ratio * rho_f * base_length * z_axis_width
    density = rod_line_density / base_area
    moment_of_inertia = np.pi / 4 * base_radius**4
    # nondim_bending_stiffness = youngs_modulus * moment_of_inertia
    # / (rho_f vel_free_stream^2 base_length^3)
    youngs_modulus = (
        nondim_bending_stiffness
        * (rho_f * velocity_free_stream**2 * base_length**3 * z_axis_width)
        / moment_of_inertia
    )
    poisson_ratio = 0.5
    # Fr = gravitational_acc * base_length / velocity_free_stream ^ 2
    gravitational_acc = froude * velocity_free_stream**2 / base_length

    flow_past_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus,
        shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
    )
    tip_start_position = flow_past_rod.position_collection[(x_axis_idx, y_axis_idx), -1]
    flow_past_sim.append(flow_past_rod)
    flow_past_sim.constrain(flow_past_rod).using(
        ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    # Add gravitational forces
    flow_past_sim.add_forcing_to(flow_past_rod).using(
        ea.GravityForces, acc_gravity=np.array([gravitational_acc, 0.0, 0.0])
    )
    # add damping
    dl = base_length / n_elem
    rod_dt = 0.01 * dl
    damping_constant = 0.5e-3
    flow_past_sim.dampen(flow_past_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=rod_dt,
    )
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    flow_solver_precision = precision
    real_t = get_real_t(flow_solver_precision)
    # Flow parameters
    # Re = velocity_free_stream * base_length / nu
    nu = base_length * velocity_free_stream / reynolds
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=(grid_size_y, grid_size_x),
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=True,
        real_t=real_t,
        num_threads=num_threads,
    )
    # ==================FLOW SETUP END=========================

    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=flow_past_rod,
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
    flow_past_sim.add_forcing_to(flow_past_rod).using(
        sps.FlowForces,
        cosserat_rod_flow_interactor,
    )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======
    if save_flow_data:
        # setup IO
        # TODO internalise this in flow simulator as dump_fields
        io_origin = np.array(
            [
                flow_sim.position_field[y_axis_idx].min(),
                flow_sim.position_field[x_axis_idx].min(),
            ]
        )
        io_dx = flow_sim.dx * np.ones(grid_dim)
        io_grid_size = np.array(grid_size)
        io = IO(dim=grid_dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field, velocity=flow_sim.velocity_field
        )
        # Initialize rod IO
        rod_io = IO(dim=grid_dim, real_dtype=real_t)
        # Add vector field on lagrangian grid
        rod_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=cosserat_rod_flow_interactor.forcing_grid.position_field,
            vector_3d=cosserat_rod_flow_interactor.lag_grid_forcing_field,
        )

    # =================TIMESTEPPING====================
    flow_past_sim.finalize()
    timestepper = ea.PositionVerlet()
    do_step, stages_and_updates = ea.extend_stepper_interface(
        timestepper, flow_past_sim
    )
    time = 0.0
    foto_timer = 0.0
    timescale = base_length / velocity_free_stream
    final_time = nondim_final_time * timescale
    foto_timer_limit = final_time / 60

    # setup freestream ramping
    ramp_timescale = timescale
    velocity_free_stream_perturb = 0.5 * velocity_free_stream

    data_timer = 0.0
    data_timer_limit = 0.1 * timescale
    tip_time = []
    tip_position = []

    # create fig for plotting flow fields
    fig, ax = sps.create_figure_and_axes()

    while time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(f"Vorticity, time: {time / timescale:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                flow_sim.vorticity_field,
                levels=np.linspace(-5, 5, 100),
                extend="both",
                cmap=sps.lab_cmap,
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.plot(
                flow_past_rod.position_collection[x_axis_idx],
                flow_past_rod.position_collection[y_axis_idx],
                linewidth=3,
                color="k",
            )
            sps.save_and_clear_fig(
                fig, ax, cbar, file_name="snap_" + str("%0.4d" % (time * 100)) + ".png"
            )
            print(
                f"time: {time:.2f} ({(time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                "grid deviation L2 error: "
                f"{cosserat_rod_flow_interactor.get_grid_deviation_error_l2_norm():.6f}"
            )
            if save_flow_data:
                io.save(
                    h5_file_name="sopht_" + str("%0.4d" % (time * 100)) + ".h5",
                    time=time,
                )
                rod_io.save(
                    h5_file_name="rod_" + str("%0.4d" % (time * 100)) + ".h5", time=time
                )

        # save diagnostic data
        if data_timer >= data_timer_limit or data_timer == 0:
            data_timer = 0.0
            tip_time.append(time / timescale)
            tip_position.append(
                (
                    flow_past_rod.position_collection[(x_axis_idx, y_axis_idx), -1]
                    - tip_start_position
                )
                / base_length
            )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.5)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = time
        for i in range(rod_time_steps):
            rod_time = do_step(
                timestepper, stages_and_updates, flow_past_sim, rod_time, local_rod_dt
            )
            # timestep the cosserat_rod_flow_interactor
            cosserat_rod_flow_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and rod
        cosserat_rod_flow_interactor()

        ramp_factor = np.exp(-time / ramp_timescale)
        # timestep the flow
        flow_sim.time_step(
            dt=flow_dt,
            free_stream_velocity=[
                velocity_free_stream * (1.0 - ramp_factor),
                velocity_free_stream_perturb * ramp_factor,
            ],
        )

        # update simulation time
        time += flow_dt
        foto_timer += flow_dt
        data_timer += flow_dt

    # compile video
    sps.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=10
    )

    plt.figure()
    plt.plot(np.array(tip_time), np.array(tip_position)[..., x_axis_idx], label="X")
    plt.plot(np.array(tip_time), np.array(tip_position)[..., y_axis_idx], label="Y")
    plt.legend()
    plt.xlabel("Non-dimensional time")
    plt.ylabel("Tip deflection")
    plt.savefig("tip_position_vs_time.png")

    np.savetxt(
        fname="rod_diagnostics_vs_time.csv",
        X=np.c_[
            np.array(tip_time),
            np.array(tip_position)[..., x_axis_idx],
            np.array(tip_position)[..., y_axis_idx],
        ],
        header="time, tip_x, tip_y",
        delimiter=",",
    )

    if save_flow_data:
        sps.make_dir_and_transfer_h5_data(dir_name="flow_data_h5")


if __name__ == "__main__":
    # classical benchmark params
    bmk_reynolds = 200.0
    bmk_nondim_bending_stiffness = 1.5e-3
    bmk_nondim_mass_ratio = 1.5
    bmk_froude = 0.5

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    @click.option(
        "--sim_grid_size_x", default=256, help="Number of grid points in x direction."
    )
    @click.option(
        "--nondim_final_time",
        default=75.0,
        help="Non-dimensional final simulation time.",
    )
    @click.option("--reynolds", default=bmk_reynolds, help="Reynolds number.")
    @click.option(
        "--nondim_bending_stiffness",
        default=bmk_nondim_bending_stiffness,
        help="Non-dimensional bending stiffness",
    )
    @click.option(
        "--nondim_mass_ratio",
        default=bmk_nondim_mass_ratio,
        help="Non-dimensional mass ratio.",
    )
    @click.option(
        "--froude",
        default=bmk_froude,
        help="Froude number.",
    )
    def simulate_custom_flow_past_rod_case(
        num_threads,
        sim_grid_size_x,
        nondim_final_time,
        reynolds,
        nondim_bending_stiffness,
        nondim_mass_ratio,
        froude,
    ):
        sim_grid_size_y = sim_grid_size_x // 2
        sim_grid_size = (sim_grid_size_y, sim_grid_size_x)
        click.echo(f"Number of threads for parallelism: {num_threads}")
        click.echo(f"Grid size: {sim_grid_size}")
        click.echo(f"Reynolds number: {reynolds}")
        click.echo(f"Non-dimensional bending stiffness: {nondim_bending_stiffness}")
        click.echo(f"Non-dimensional mass ratio: {nondim_mass_ratio}")
        click.echo(f"Froude number: {froude}")

        flow_past_rod_case(
            nondim_final_time=nondim_final_time,
            grid_size=sim_grid_size,
            reynolds=reynolds,
            nondim_bending_stiffness=nondim_bending_stiffness,
            nondim_mass_ratio=nondim_mass_ratio,
            froude=froude,
            num_threads=num_threads,
        )

    simulate_custom_flow_past_rod_case()

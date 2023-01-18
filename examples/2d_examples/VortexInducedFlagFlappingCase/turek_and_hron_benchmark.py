import click
import elastica as ea
import matplotlib.pyplot as plt
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu


def flow_past_rod_case(
    non_dim_final_time: float,
    grid_size: tuple[int, int],
    reynolds: float,
    cyl_diameter_to_rod_length: float,
    beam_aspect_ratio: float,
    density_ratio: float,
    cauchy: float,
    coupling_stiffness: float = -5e4,
    coupling_damping: float = -20,
    num_threads: int = 4,
    flow_precision: str = "single",
    save_flow_data: bool = False,
) -> None:
    grid_dim = 2
    grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(flow_precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    # =================COMMON SIMULATOR STUFF=======================
    nondim_mass_ratio = density_ratio * beam_aspect_ratio
    # last term on rhs corresponds to nondim moment of inertia of plate
    nondim_bending_stiffness = cauchy * (beam_aspect_ratio**3) / 12.0
    print(f"Rod non-dimensional mass ratio: {nondim_mass_ratio:.6f}")
    print(f"Rod non-dimensional bending stiffness: {nondim_bending_stiffness:.6f}")
    velocity_free_stream = 1.0
    rho_f = 1.0
    base_length = 1.0
    x_range = 5.0 * base_length
    y_range = x_range * grid_size_y / grid_size_x
    # =================PYELASTICA STUFF BEGIN=====================

    class FlowPastRodSimulator(ea.BaseSystemCollection, ea.Constraints, ea.Forcing):
        pass

    flow_past_sim = FlowPastRodSimulator()
    # setting up test params
    n_elem = int(0.8 * base_length / x_range * grid_size_x)
    start = np.array(
        [
            1.0 * base_length,
            0.5 * y_range,
            0.0,
        ]
    )
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_radius = beam_aspect_ratio * base_length / 2.0
    base_area = np.pi * base_radius**2
    # nondim_mass_ratio = rod_line_density / (rho_f * base_length)
    rod_line_density = nondim_mass_ratio * rho_f * base_length
    density = rod_line_density / base_area
    moment_of_inertia = np.pi / 4 * base_radius**4
    # for converting plate to rod stiffness
    # see Gilmanov et al. 2015
    # poisson_ratio_plate = 0.4
    # poisson_ratio_correction_factor = 1.0 - poisson_ratio_plate**2
    # TODO after convergence see if we really need it?
    # nondim_bending_stiffness /= poisson_ratio_correction_factor
    # nondim_bending_stiffness = youngs_modulus * moment_of_inertia
    # / (rho_f vel_free_stream^2 base_length^3)
    youngs_modulus = (
        nondim_bending_stiffness
        * (rho_f * velocity_free_stream**2 * base_length**3)
        / moment_of_inertia
    )
    poisson_ratio = 0.5
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
    dl = base_length / n_elem
    rod_dt = 1e-3 * dl
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    # Flow parameters
    # Re = velocity_free_stream * cyl_diameter / nu
    cyl_diameter = base_length * cyl_diameter_to_rod_length
    nu = cyl_diameter * velocity_free_stream / reynolds
    flow_sim = sps.UnboundedNavierStokesFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        with_forcing=True,
        with_free_stream_flow=True,
        real_t=real_t,
        num_threads=num_threads,
    )
    # taken from Bhardwaj et. al 2012
    drag_force_scale = 0.5 * cyl_diameter * rho_f * velocity_free_stream**2
    # ==================FLOW SETUP END=========================
    # Initialise the top and bottom walls as fixed rods
    wall_boundary_offset = 4 * flow_sim.dx  # to avoid interpolation at boundaries
    wall_length = x_range - 2 * wall_boundary_offset
    wall_n_elem = n_elem * int(x_range / base_length)
    top_wall_start = np.array(
        [
            wall_boundary_offset,
            flow_past_rod.position_collection[y_axis_idx, 0] + 2.0 * cyl_diameter,
            0.0,
        ]
    )
    top_wall_rod = ea.CosseratRod.straight_rod(
        wall_n_elem,
        top_wall_start,
        direction,
        normal,
        wall_length,
        base_radius,
        density,
        0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus,
        shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
    )
    bottom_wall_start = np.array(
        [
            wall_boundary_offset,
            flow_past_rod.position_collection[y_axis_idx, 0] - 2.0 * cyl_diameter,
            0.0,
        ]
    )
    bottom_wall_rod = ea.CosseratRod.straight_rod(
        wall_n_elem,
        bottom_wall_start,
        direction,
        normal,
        wall_length,
        base_radius,
        density,
        0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus,
        shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
    )
    # Since the walls are fixed, we don't add it to pyelastica simulator,
    # and directly use it for setting up the flow interactor.
    # Initialize fixed cylinder (elastica rigid body) with direction along Z
    # for Turek/Hron case diameter / rod_length = 0.1 / 0.35
    x_cm = flow_past_rod.position_collection[x_axis_idx, 0] - 0.5 * cyl_diameter
    y_cm = flow_past_rod.position_collection[y_axis_idx, 0]
    start = np.array([x_cm, y_cm, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    density = 1e3
    cylinder = ea.Cylinder(
        start,
        direction,
        normal,
        base_length,
        base_radius=0.5 * cyl_diameter,
        density=density,
    )
    # Since the cylinder is fixed, we don't add it to pyelastica simulator,
    # and directly use it for setting up the flow interactor.
    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    flow_body_interactors: list[
        sps.RigidBodyFlowInteraction | sps.CosseratRodFlowInteraction
    ] = []
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=flow_past_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        forcing_grid_cls=sps.CosseratRodEdgeForcingGrid,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_body_interactors.append(cosserat_rod_flow_interactor)
    flow_past_sim.add_forcing_to(flow_past_rod).using(
        sps.FlowForces,
        cosserat_rod_flow_interactor,
    )
    cyl_num_forcing_points = n_elem
    cylinder_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=cylinder,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        forcing_grid_cls=sps.CircularCylinderForcingGrid,
        real_t=real_t,
        num_forcing_points=cyl_num_forcing_points,
    )
    flow_body_interactors.append(cylinder_flow_interactor)
    top_wall_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=top_wall_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        forcing_grid_cls=sps.CosseratRodNodalForcingGrid,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_body_interactors.append(top_wall_flow_interactor)
    bottom_wall_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=bottom_wall_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        forcing_grid_cls=sps.CosseratRodNodalForcingGrid,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_body_interactors.append(bottom_wall_flow_interactor)
    # ==================FLOW-ROD COMMUNICATOR SETUP END======
    # =================FLOW H5 DUMPING====================
    if save_flow_data:
        # setup flow IO
        io = spu.EulerianFieldIO(
            position_field=flow_sim.position_field,
            eulerian_fields_dict={
                "vorticity": flow_sim.vorticity_field,
                "velocity": flow_sim.velocity_field,
            },
        )
        # Initialize rod IO
        rod_io = spu.CosseratRodIO(
            cosserat_rod=flow_past_rod, dim=grid_dim, real_dtype=real_t
        )
        # Next we initialise fixed locations IOs and dump then only once
        fixed_location_interactors = {
            "top_wall": top_wall_flow_interactor,
            "bottom_wall": bottom_wall_flow_interactor,
            "cylinder": cylinder_flow_interactor,
        }
        for tag, interactor in fixed_location_interactors.items():
            body_io = spu.IO(dim=grid_dim, real_dtype=real_t)
            # Add vector field on lagrangian grid
            body_io.add_as_lagrangian_fields_for_io(
                lagrangian_grid=interactor.forcing_grid.position_field,
                vector_3d=interactor.lag_grid_forcing_field,
            )
            body_io.save(h5_file_name=tag + ".h5", time=0.0)
            del body_io
    # =================TIMESTEPPING====================
    flow_past_sim.finalize()
    timestepper = ea.PositionVerlet()
    do_step, stages_and_updates = ea.extend_stepper_interface(
        timestepper, flow_past_sim
    )
    foto_timer = 0.0
    timescale = base_length / velocity_free_stream
    final_time = non_dim_final_time * timescale
    foto_timer_limit = final_time / (2 * non_dim_final_time)

    # setup freestream ramping
    ramp_timescale = 0.5 * timescale
    velocity_free_stream_perturb = 0.5 * velocity_free_stream

    data_timer = 0.0
    data_timer_limit = 0.05 * timescale
    tip_time = []
    tip_position = []
    drag_coeff = []

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()

    while flow_sim.time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(f"Vorticity, time: {flow_sim.time / timescale:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx],
                flow_sim.position_field[y_axis_idx],
                flow_sim.vorticity_field,
                levels=np.linspace(-15, 15, 100),
                extend="both",
                cmap=spu.get_lab_cmap(),
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.plot(
                flow_past_rod.position_collection[x_axis_idx],
                flow_past_rod.position_collection[y_axis_idx],
                linewidth=3,
                color="k",
            )
            for flow_body_interactor in flow_body_interactors:
                ax.scatter(
                    flow_body_interactor.forcing_grid.position_field[x_axis_idx],
                    flow_body_interactor.forcing_grid.position_field[y_axis_idx],
                    s=5,
                    color="k",
                )
            spu.save_and_clear_fig(
                fig,
                ax,
                cbar,
                file_name="snap_" + str("%0.4d" % (flow_sim.time * 100)) + ".png",
            )
            grid_dev_error = 0.0
            for flow_body_interactor in flow_body_interactors:
                grid_dev_error += (
                    flow_body_interactor.get_grid_deviation_error_l2_norm()
                )
            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"grid deviation L2 error: {grid_dev_error:.6f}"
            )
            if save_flow_data:
                io.save(
                    h5_file_name="sopht_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
                rod_io.save(
                    h5_file_name="rod_" + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                    time=flow_sim.time,
                )

        # save diagnostic data
        if data_timer > data_timer_limit or data_timer == 0.0:
            data_timer = 0.0
            tip_time.append(flow_sim.time / timescale)
            tip_position.append(
                (
                    flow_past_rod.position_collection[(x_axis_idx, y_axis_idx), -1]
                    - tip_start_position
                )
                / cyl_diameter
            )
            drag_force = np.sum(
                cosserat_rod_flow_interactor.lag_grid_forcing_field[x_axis_idx]
            ) + np.sum(cylinder_flow_interactor.lag_grid_forcing_field[x_axis_idx])
            drag_coeff.append(abs(drag_force) / drag_force_scale)

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.5)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            rod_time = do_step(
                timestepper, stages_and_updates, flow_past_sim, rod_time, local_rod_dt
            )
            # timestep the body_flow_interactors
            for flow_body_interactor in flow_body_interactors:
                flow_body_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and rod
        for flow_body_interactor in flow_body_interactors:
            flow_body_interactor()

        # timestep the flow
        ramp_factor = 1.0 - np.exp(-flow_sim.time / ramp_timescale)
        flow_sim.time_step(
            dt=flow_dt,
            free_stream_velocity=[
                velocity_free_stream * ramp_factor,
                velocity_free_stream_perturb * (1.0 - ramp_factor),
            ],
        )

        # update simulation time
        foto_timer += flow_dt
        data_timer += flow_dt

    # compile video
    spu.make_video_from_image_series(
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
        fname="plate_diagnostics_vs_time.csv",
        X=np.c_[
            np.array(tip_time),
            np.array(tip_position)[..., x_axis_idx],
            np.array(tip_position)[..., y_axis_idx],
            np.array(drag_coeff),
        ],
        header="time, tip_x, tip_y, drag_coeff",
        delimiter=",",
    )

    if save_flow_data:
        spu.make_dir_and_transfer_h5_data(dir_name="flow_data_h5")


if __name__ == "__main__":
    # experimental params
    exp_reynolds = 100.0
    exp_cyl_diameter = 0.1
    exp_rod_length = 0.35
    exp_rod_diameter = 2e-2
    exp_beam_aspect_ratio = exp_rod_diameter / exp_rod_length
    exp_cyl_diameter_to_rod_length = exp_cyl_diameter / exp_rod_length
    exp_density_ratio = 10.0
    exp_cauchy = 1.4e3

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    @click.option(
        "--sim_grid_size_x", default=256, help="Number of grid points in x direction."
    )
    @click.option(
        "--nondim_final_time",
        default=29.0,
        help="Non-dimensional final simulation time.",
    )
    def simulate_custom_flow_past_rod_case(
        num_threads: int, sim_grid_size_x: int, nondim_final_time: float
    ) -> None:
        sim_grid_size_y = sim_grid_size_x * 3 // 8
        sim_grid_size = (sim_grid_size_y, sim_grid_size_x)
        click.echo(f"Number of threads for parallelism: {num_threads}")
        click.echo(f"Grid size: {sim_grid_size}")
        flow_past_rod_case(
            reynolds=exp_reynolds,
            cyl_diameter_to_rod_length=exp_cyl_diameter_to_rod_length,
            beam_aspect_ratio=exp_beam_aspect_ratio,
            density_ratio=exp_density_ratio,
            cauchy=exp_cauchy,
            non_dim_final_time=nondim_final_time,
            grid_size=sim_grid_size,
            num_threads=num_threads,
        )

    simulate_custom_flow_past_rod_case()

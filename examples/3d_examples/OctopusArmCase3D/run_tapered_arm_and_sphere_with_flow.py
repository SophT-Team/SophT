import numpy as np
from set_environment_tapered_arm_and_sphere_with_flow import Environment
from arm_functions_3d import SigmoidActivationLongitudinalMuscles  # , LocalActivation
import sopht.simulator as sps
import sopht.utils as spu
import elastica as ea
import click
from matplotlib import pyplot as plt


def tapered_arm_and_cylinder_flow_coupling(
    non_dimensional_final_time,
    n_elems,
    slenderness_ratio,
    cauchy_number,
    mass_ratio,
    reynolds_number,
    stretch_bending_ratio,
    taper_ratio,
    grid_size,
    coupling_stiffness=-2e4,
    coupling_damping=-1e1,
    num_threads=4,
    precision="single",
    save_data=True,
):
    # =================COMMON STUFF BEGIN=====================
    grid_dim = 3
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    period = 1
    rho_f = 1.0
    base_length = 1
    vel_scale = base_length / period
    final_time = period * non_dimensional_final_time
    x_range = 1.8 * base_length
    y_range = grid_size_y / grid_size_x * x_range
    z_range = grid_size_z / grid_size_x * x_range
    # 2x1 domain with x_range = 4 * base_length

    # =================PYELASTICA STUFF BEGIN=====================
    rod_dt = 3.0e-4
    env = Environment(final_time, time_step=rod_dt, rendering_fps=30)
    rho_s = mass_ratio * rho_f
    base_diameter = base_length / slenderness_ratio
    base_radius = base_diameter / 2
    base_area = np.pi * base_radius**2
    moment_of_inertia = np.pi / 4 * base_radius**4
    # Cau = (rho_f U^2 L^3 D) / EI
    youngs_modulus = (rho_f * vel_scale**2 * base_length**3 * base_diameter) / (
        cauchy_number * moment_of_inertia
    )
    Es_Eb = stretch_bending_ratio * moment_of_inertia / (base_area * base_length**2)
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))

    start = np.zeros(grid_dim) + np.array(
        [0.3 * x_range, 0.5 * y_range, 0.12 * z_range]
    )
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])

    radius = np.linspace(base_radius, base_radius / taper_ratio, n_elems + 1)
    radius_mean = (radius[:-1] + radius[1:]) / 2

    shearable_rod = ea.CosseratRod.straight_rod(
        n_elements=n_elems,
        start=start,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=radius_mean.copy(),
        density=rho_s,
        nu=0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    shearable_rod.shear_matrix[2, 2, :] *= Es_Eb
    env.reset(youngs_modulus, shearable_rod)
    # Add gravity
    env.simulator.constrain(env.shearable_rod).using(
        ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    # Setup activation functions to control muscles
    activations = []
    activation_functions = []
    for m in range(len(env.muscle_groups)):
        activations.append(np.zeros(env.muscle_groups[m].activation.shape))
        activation_functions.append(
            SigmoidActivationLongitudinalMuscles(
                beta=1,
                tau=0.05 * 50,
                start_time=1.0,
                end_time=10,  # 0.1 + 2 + 0.1 * 10,
                start_non_dim_length=0,
                end_non_dim_length=0.5,
                activation_level_max=0.15,
                activation_level_end=0.0,
                activation_lower_threshold=1e-3,
                n_elems=n_elems,
            )
        )
    #     activation_functions.append(
    #         LocalActivation(
    #             ramp_interval=1.0,
    #             ramp_up_time=0.0,
    #             ramp_down_time=15,
    #             start_idx=0,
    #             end_idx=n_elems,
    #             activation_level=0.1,
    #         )
    #     )

    # Initialize fixed sphere (elastica rigid body)
    sphere_com = np.array([0.6667 * x_range, 0.5 * y_range, 0.3111 * z_range])
    sphere = ea.Sphere(
        center=sphere_com,
        base_radius=radius_mean[0] * 1.5,
        density=rho_s / 1000 / 5 / 2,
    )
    env.simulator.append(sphere)
    # =================PYELASTICA STUFF END=====================
    # ==================FLOW SETUP START=========================
    # Flow parameters
    kinematic_viscosity = base_diameter * vel_scale / reynolds_number
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=kinematic_viscosity,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=False,
        real_t=real_t,
        num_threads=num_threads,
        navier_stokes_inertial_term_form="rotational",
        filter_vorticity=True,
        filter_setting_dict={"order": 1, "type": "multiplicative"},
    )
    # ==================FLOW SETUP END=========================
    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    flow_body_interactors = []
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=env.shearable_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        num_threads=num_threads,
        forcing_grid_cls=sps.CosseratRodSurfaceForcingGrid,
        surface_grid_density_for_largest_element=15,
    )
    flow_body_interactors.append(cosserat_rod_flow_interactor)

    env.simulator.add_forcing_to(env.shearable_rod).using(
        sps.FlowForces,
        cosserat_rod_flow_interactor,
    )

    num_forcing_points_along_equator = 17
    sphere_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=sphere,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness / 5,
        virtual_boundary_damping_coeff=coupling_damping / 5,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        forcing_grid_cls=sps.SphereForcingGrid,
        num_forcing_points_along_equator=num_forcing_points_along_equator,
    )
    flow_body_interactors.append(sphere_flow_interactor)
    env.simulator.add_forcing_to(sphere).using(
        sps.FlowForces,
        sphere_flow_interactor,
    )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======

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
        io_grid_size = np.array(grid_size)
        # Initialize flow eulerian grid IO
        io = spu.IO(dim=grid_dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field, velocity=flow_sim.velocity_field
        )
        # Initialize rod io
        rod_io = spu.CosseratRodIO(
            cosserat_rod=shearable_rod, dim=grid_dim, real_dtype=real_t
        )
        # Initialize sphere io
        sphere_io = spu.IO(dim=grid_dim, real_dtype=real_t)
        # Add vector field on lagrangian grid
        sphere_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=sphere_flow_interactor.forcing_grid.position_field,
            lagrangian_grid_name="sphere",
            vector_3d=sphere_flow_interactor.lag_grid_forcing_field,
        )
    # =================TIMESTEPPING====================
    # Finalize the pyelastica environment
    _, _ = env.finalize()

    foto_timer = 0.0
    foto_timer_limit = period / 10
    time_history = []

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()

    while flow_sim.time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            if save_data:
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
                sphere_io.save(
                    h5_file_name="sphere_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
            ax.set_title(f"Vorticity magnitude, time: {flow_sim.time / final_time:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx, :, grid_size_y // 2, :],
                flow_sim.position_field[z_axis_idx, :, grid_size_y // 2, :],
                # TODO have a function for computing velocity magnitude
                np.linalg.norm(
                    np.mean(
                        flow_sim.velocity_field[
                            :, :, grid_size_y // 2 - 1 : grid_size_y // 2 + 1, :
                        ],
                        axis=2,
                    ),
                    axis=0,
                ),
                levels=np.linspace(0, vel_scale, 50),
                extend="both",
                cmap="Purples",
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.scatter(
                cosserat_rod_flow_interactor.forcing_grid.position_field[x_axis_idx],
                cosserat_rod_flow_interactor.forcing_grid.position_field[z_axis_idx],
                s=5,
                color="k",
            )
            ax.scatter(
                sphere_flow_interactor.forcing_grid.position_field[x_axis_idx],
                sphere_flow_interactor.forcing_grid.position_field[z_axis_idx],
                s=5,
                color="k",
            )
            spu.save_and_clear_fig(
                fig,
                ax,
                cbar,
                file_name="snap_" + str("%0.5d" % (flow_sim.time * 100)) + ".png",
            )

            plt.rcParams.update({"font.size": 22})
            fig_2 = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
            axs = []
            axs.append(plt.subplot2grid((1, 1), (0, 0)))
            axs[0].plot(
                env.shearable_rod.velocity_collection[x_axis_idx],
            )
            axs[0].plot(
                env.shearable_rod.velocity_collection[y_axis_idx],
            )
            axs[0].plot(
                env.shearable_rod.velocity_collection[z_axis_idx],
            )
            axs[0].set_xlabel("idx", fontsize=20)
            axs[0].set_ylabel("vel", fontsize=20)
            axs[0].set_ylim(-1.5, 1.5)
            plt.tight_layout()
            fig_2.align_ylabels()
            fig_2.savefig("vel_" + str("%0.5d" % (flow_sim.time * 100)) + ".png")
            plt.close(plt.gcf())

            time_history.append(flow_sim.time)
            grid_dev_error = 0.0
            for flow_body_interactor in flow_body_interactors:
                grid_dev_error += (
                    flow_body_interactor.get_grid_deviation_error_l2_norm()
                )
            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f}, "
                f"grid deviation L2 error: {grid_dev_error:.6f}"
            )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.25)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            # Activate longitudinal muscle
            activation_functions[2].apply_activation(
                env.shearable_rod, activations[2], rod_time
            )
            # Do one elastica step
            env.time_step = local_rod_dt
            rod_time, systems, done = env.step(rod_time, activations)

            # timestep the body_flow_interactors
            for flow_body_interactor in flow_body_interactors:
                flow_body_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and bodies
        for flow_body_interactor in flow_body_interactors:
            flow_body_interactor()

        # timestep the flow
        flow_sim.time_step(dt=flow_dt)

        # update timer
        foto_timer += flow_dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=10
    )
    spu.make_video_from_image_series(
        video_name="rod_vel", image_series_name="vel", frame_rate=10
    )


if __name__ == "__main__":

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    def simulate_parallelised_octopus_arm(num_threads):

        click.echo(f"Number of threads for parallelism: {num_threads}")

        final_time = 5  # 7.5
        period = 1.0
        exp_rho_s = 1e3  # kg/m3
        exp_rho_f = 1.21  # kg/m3
        exp_youngs_modulus = 2.25e5  # Pa
        exp_base_length = 25e-3  # m
        exp_base_diameter = exp_base_length / 10  # 0.4e-3  # m
        exp_kinematic_viscosity = 1.51e-5 / 5  # m2/s
        exp_U_free_stream = 1.1  # m/s
        exp_mass_ratio = exp_rho_s / exp_rho_f
        exp_slenderness_ratio = exp_base_length / exp_base_diameter
        exp_base_radius = exp_base_diameter / 2
        exp_base_area = np.pi * exp_base_radius**2
        exp_moment_of_inertia = np.pi / 4 * exp_base_radius**4
        exp_bending_rigidity = exp_youngs_modulus * exp_moment_of_inertia
        exp_cauchy_number = (
            exp_rho_f
            * exp_U_free_stream**2
            * exp_base_length**3
            * exp_base_diameter
            / exp_bending_rigidity
        )
        exp_Re = exp_U_free_stream * exp_base_diameter / exp_kinematic_viscosity
        # stretch to bending ratio EAL2 / EI
        exp_Ks_Kb = (exp_youngs_modulus * exp_base_area * exp_base_length**2) / (
            exp_youngs_modulus * exp_moment_of_inertia
        )
        exp_non_dimensional_final_time = final_time / period
        exp_n_elem = 50
        exp_taper_ratio = 7
        print(f"Re: {exp_Re}, Ca: {exp_cauchy_number}, Ks_Kb: {exp_Ks_Kb}")
        grid_size = (128, 32, 128)
        tapered_arm_and_cylinder_flow_coupling(
            non_dimensional_final_time=exp_non_dimensional_final_time,
            n_elems=exp_n_elem,
            slenderness_ratio=exp_slenderness_ratio,
            cauchy_number=exp_cauchy_number,
            mass_ratio=exp_mass_ratio,
            reynolds_number=exp_Re,
            stretch_bending_ratio=exp_Ks_Kb,
            taper_ratio=exp_taper_ratio,
            grid_size=grid_size,
            num_threads=num_threads,
        )

    simulate_parallelised_octopus_arm()

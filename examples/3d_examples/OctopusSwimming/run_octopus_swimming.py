import numpy as np
from sopht.utils.precision import get_real_t
from set_environment_octopus import OctopusEnvironment
import sopht.simulator as sps
import elastica as ea
import sopht.utils as spu
import click
from octopus_initializer_functions import (
    assemble_octopus,
    initialize_activation_functions,
)


def octopus_swimming(
    non_dimensional_final_time,
    n_elems,
    slenderness_ratio,
    non_dim_bending_stiffness,
    mass_ratio,
    reynolds_number,
    taper_ratio,
    activation_period,
    activation_level_max,
    grid_size,
    surface_grid_density_for_largest_element,
    coupling_stiffness=-2e2 * 1e2,
    coupling_damping=-1e-1 * 1e2,
    num_threads=4,
    precision="single",
    save_data=True,
):
    # =================COMMON STUFF BEGIN=====================
    grid_dim = 3
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    period = activation_period
    rho_f = 1
    base_length = 1.0
    vel_scale = base_length / period
    final_time = period * non_dimensional_final_time
    y_range = 2.5 * base_length
    x_range = grid_size_x / grid_size_y * y_range
    z_range = grid_size_z / grid_size_y * y_range
    # 2x1 domain with x_range = 4 * base_length
    # =================PYELASTICA STUFF BEGIN=====================
    rod_dt = 3.0e-4 / 10 / 2

    env = OctopusEnvironment(final_time, time_step=rod_dt, rendering_fps=30)
    rho_s = mass_ratio * rho_f
    base_diameter = base_length / slenderness_ratio
    base_radius = base_diameter / 2
    # base_area = np.pi * base_radius**2
    moment_of_inertia = np.pi / 4 * base_radius**4
    # Kb = EI / (rho_f U^2 L^3 D)
    youngs_modulus = (
        non_dim_bending_stiffness
        * rho_f
        * vel_scale**2
        * base_length**3
        * base_diameter
    ) / (moment_of_inertia)
    # bending_rigidity = youngs_modulus * moment_of_inertia
    # natural_frequency = 3.5160 / (base_length ** 2) * np.sqrt(bending_rigidity / (rho_s * base_area))

    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))

    start = np.zeros(grid_dim) + np.array([0.7 * x_range, 0.5 * y_range, 0.5 * z_range])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    binormal = np.cross(direction, normal)

    # Initialize octopus arms, head and body
    arm_rod_list, rod_list, rigid_body_list = assemble_octopus(
        n_elems,
        start,
        direction,
        binormal,
        normal,
        rho_s,
        base_length,
        base_radius,
        taper_ratio,
        youngs_modulus,
        shear_modulus,
    )
    body_rod = rod_list[-1]
    sphere = rigid_body_list[0]

    env.reset(youngs_modulus, rho_s, rod_list, arm_rod_list, rigid_body_list)

    # Connect rods
    for i, rod in enumerate(env.arm_rod_list):
        env.simulator.connect(
            rod, body_rod, first_connect_idx=0, second_connect_idx=0
        ).using(ea.FreeJoint, k=youngs_modulus / 100, nu=0)

    # head tip
    env.simulator.connect(
        sphere, body_rod, first_connect_idx=0, second_connect_idx=-1
    ).using(ea.FreeJoint, k=youngs_modulus / 100, nu=0)

    # Setup activation functions to control muscles
    activations, activation_functions = initialize_activation_functions(
        env, activation_period, activation_level_max
    )

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
        filter_setting_dict={"order": 5, "type": "convolution"},
    )
    # ==================FLOW SETUP END=========================
    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    flow_body_interactors = []
    for rod in env.rod_list:
        scale_grid_density = np.ceil(np.max(rod.radius) / base_radius)
        cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
            cosserat_rod=rod,
            eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
            eul_grid_velocity_field=flow_sim.velocity_field,
            virtual_boundary_stiffness_coeff=coupling_stiffness,
            virtual_boundary_damping_coeff=coupling_damping,
            dx=flow_sim.dx,
            grid_dim=grid_dim,
            real_t=real_t,
            num_threads=num_threads,
            forcing_grid_cls=sps.CosseratRodSurfaceForcingGrid,
            surface_grid_density_for_largest_element=surface_grid_density_for_largest_element
            * scale_grid_density,
        )
        flow_body_interactors.append(cosserat_rod_flow_interactor)

        env.simulator.add_forcing_to(rod).using(
            sps.FlowForces,
            cosserat_rod_flow_interactor,
        )

    sphere_diameter = 2 * sphere.radius
    num_forcing_points_along_equator = int(
        2 * 1.875 * sphere_diameter / y_range * grid_size_y
    )
    sphere_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=sphere,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
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
        rod_io_list = []
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
        for rod_id, rod in enumerate(env.rod_list):
            rod_io_list.append(
                spu.CosseratRodIO(cosserat_rod=rod, dim=grid_dim, real_dtype=real_t)
            )

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
    foto_timer_limit = 1 / 30
    time_history = []
    head_com_velocity_history = []
    head_com_position_history = []

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
                for rod_id, rod in enumerate(env.rod_list):
                    rod_io_list[rod_id].save(
                        h5_file_name="rod_"
                        + str(rod_id)
                        + "_"
                        + str("%0.4d" % (flow_sim.time * 100))
                        + ".h5",
                        time=flow_sim.time,
                    )
                sphere_io.save(
                    h5_file_name="sphere_"
                    + str("%0.4d" % (flow_sim.time * 100))
                    + ".h5",
                    time=flow_sim.time,
                )
                env.save_data()

            ax.set_title(f"Velocity magnitude, time: {flow_sim.time / final_time:.2f}")
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
            for flow_body_interactor in flow_body_interactors:
                ax.scatter(
                    flow_body_interactor.forcing_grid.position_field[x_axis_idx],
                    flow_body_interactor.forcing_grid.position_field[z_axis_idx],
                    s=5,
                    color="k",
                )

            spu.save_and_clear_fig(
                fig,
                ax,
                cbar,
                file_name="snap_" + str("%0.5d" % (flow_sim.time * 100)) + ".png",
            )

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
            head_com_velocity_history.append(sphere.velocity_collection[..., 0].copy())
            head_com_position_history.append(sphere.position_collection[..., 0].copy())
            with open("octopus_head_velocity_vs_time.csv", "ab") as f:
                np.savetxt(
                    f,
                    np.c_[
                        np.hstack(
                            (
                                flow_sim.time,
                                head_com_velocity_history[-1],
                                np.linalg.norm(head_com_velocity_history[-1]),
                            )
                        )
                    ].T,
                    delimiter=",",
                )

            with open("octopus_head_position_vs_time.csv", "ab") as f:
                np.savetxt(
                    f,
                    np.c_[np.hstack((flow_sim.time, head_com_position_history[-1]))].T,
                    delimiter=",",
                )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.125)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            # Activate longitudinal muscle
            for rod_id, rod in enumerate(env.arm_rod_list):  # exclude head
                activation_functions[rod_id][4].apply_activation(
                    rod, activations[rod_id][4], rod_time
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

        # update simulation time
        foto_timer += flow_dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=30
    )
    np.savetxt(
        "octopus_head_velocity_vs_time.csv",
        np.c_[
            np.array(time_history),
            np.array(head_com_velocity_history),
            np.linalg.norm(np.array(head_com_velocity_history), axis=1),
        ],
        delimiter=",",
        header="time, vel x, vel y, vel z, vel norm",
    )
    np.savetxt(
        "octopus_head_position_vs_time.csv",
        np.c_[
            np.array(time_history),
            np.array(head_com_position_history),
        ],
        delimiter=",",
        header="time, pos x, pos y, pos z",
    )

    if save_data:
        spu.make_dir_and_transfer_h5_data(dir_name="flow_data_h5")

    env.save_data()


if __name__ == "__main__":

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    @click.option("--nz", default=196, help="Number of grid points in z direction.")
    @click.option("--taper_ratio", default=12, help="Arm taper ratio.")
    @click.option("--activation_mag", default=0.2, help="Muscle activation magnitude.")
    @click.option("--period", default=2.0, help="Activation period.")
    @click.option("--re_scale", default=2.0, help="Reynold number scale.")
    @click.option(
        "--adult", default=True, help="True for Adult octopus, False for Juvenile"
    )
    def simulate_swimming_octopus(
        num_threads, nz, taper_ratio, activation_mag, period, re_scale, adult
    ):

        nx = 2 * nz
        ny = nz
        grid_size = (nz, ny, nx)
        surface_grid_density_for_largest_element = nz // 10
        exp_n_elem = nz // 3

        click.echo(f"Number of threads for parallelism: {num_threads}")

        if adult:
            geometry_scale = 1.0
        else:
            geometry_scale = 0.1

        # final_time = 40
        exp_non_dimensional_final_time = 10
        # period = 1.0
        exp_rho_s = 1044  # kg/m3
        exp_rho_f = 1022  # kg/m3
        exp_youngs_modulus = 1e4  # Pa
        exp_base_length = 0.2 * geometry_scale  # m
        exp_base_diameter = 24e-3 * geometry_scale  # m
        exp_kinematic_viscosity = 1e-6  # m2/s
        exp_activation_period = period  # 2.533 * 3
        exp_activation_level_max = activation_mag  # 0.2
        exp_U_free_stream = exp_base_length / exp_activation_period  # m/s
        exp_mass_ratio = exp_rho_s / exp_rho_f
        exp_slenderness_ratio = exp_base_length / exp_base_diameter
        exp_base_radius = exp_base_diameter / 2
        # exp_base_area = np.pi * exp_base_radius**2
        exp_moment_of_inertia = np.pi / 4 * exp_base_radius**4
        exp_bending_rigidity = exp_youngs_modulus * exp_moment_of_inertia

        # natural_frequency = 3.5160 / (exp_base_length**2) * np.sqrt(exp_bending_rigidity/(exp_rho_s*exp_base_area))
        # non_dimensional_period = natural_frequency * period

        exp_non_dim_bending_stiffness = exp_bending_rigidity / (
            exp_rho_f
            * exp_U_free_stream**2
            * exp_base_length**3
            * exp_base_diameter
        )
        exp_Re = exp_U_free_stream * exp_base_diameter / exp_kinematic_viscosity
        exp_Re *= re_scale

        exp_taper_ratio = taper_ratio  # 7
        print(f"Re: {exp_Re}, Kb: {exp_non_dim_bending_stiffness}")
        octopus_swimming(
            non_dimensional_final_time=exp_non_dimensional_final_time,
            n_elems=exp_n_elem,
            slenderness_ratio=exp_slenderness_ratio,
            non_dim_bending_stiffness=exp_non_dim_bending_stiffness,
            mass_ratio=exp_mass_ratio,
            reynolds_number=exp_Re,
            taper_ratio=exp_taper_ratio,
            activation_period=exp_activation_period,
            activation_level_max=exp_activation_level_max,
            grid_size=grid_size,
            surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
            num_threads=num_threads,
            save_data=False,
        )

    simulate_swimming_octopus()

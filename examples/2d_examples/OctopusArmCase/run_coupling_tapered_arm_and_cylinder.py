import numpy as np
import os
from sopht.utils.precision import get_real_t
from set_environment_tapered_arm_cylinder import Environment
from arm_functions import SigmoidActivationLongitudinalMuscles, LocalActivation
import sopht_simulator as sps


def tapered_arm_and_cylinder_flow_coupling(
    final_time_by_period,
    grid_size,
    Re=200,
    rod_coupling_type="one_way",
    rigid_body_coupling_type="one_way",
    num_threads=8,
    precision="single",
):
    # =================PYELASTICA STUFF BEGIN=====================
    period = 1
    final_time = period * final_time_by_period
    rod_dt = 3.0e-4
    env = Environment(final_time, time_step=rod_dt, rendering_fps=30)
    env.reset()

    base_length = env.shearable_rod.rest_lengths.sum()
    dl = env.shearable_rod.rest_lengths[0]

    # Setup activation functions to control muscles
    n_elements = env.shearable_rod.n_elems

    activations = []
    activation_functions = []
    for m in range(len(env.muscle_groups)):
        activations.append(np.zeros(env.muscle_groups[m].activation.shape))
        activation_functions.append(
            SigmoidActivationLongitudinalMuscles(
                beta=1,
                tau=0.04,
                start_time=0,
                end_time=10 * 0.04,
                start_idx=0,
                end_idx=n_elements,
                activation_level_max=1.0,
                activation_level_end=0.0,
                activation_lower_threshold=1e-3,
            )
        )
        # activation_functions.append(
        #     LocalActivation(
        #         ramp_interval=1.0,
        #         ramp_up_time=0.0,
        #         ramp_down_time=15,
        #         start_idx=0,
        #         end_idx=n_elements,
        #         activation_level=1.0,
        #     )
        # )

    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    flow_solver_precision = precision
    real_t = get_real_t(flow_solver_precision)
    CFL = 0.1
    grid_size_x = grid_size
    grid_size_y = grid_size_x  # // 2
    # 2x1 domain with x_range = 4 * base_length
    x_range = 2 * base_length
    # Flow parameters
    vel_scale = base_length / period
    nu = base_length * vel_scale / Re
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=(grid_size_y, grid_size_x),
        x_range=x_range,
        kinematic_viscosity=nu,
        CFL=CFL,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        num_threads=num_threads,
    )
    # ==================FLOW SETUP END=========================

    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    virtual_boundary_stiffness_coeff = real_t(-2e4 * dl)
    virtual_boundary_damping_coeff = real_t(-2e1 * dl)
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=env.shearable_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
        dx=flow_sim.dx,
        grid_dim=2,
        real_t=real_t,
        num_threads=num_threads,
        forcing_grid_cls=sps.CosseratRodEdgeForcingGrid,
    )
    if rod_coupling_type == "two_way":
        env.simulator.add_forcing_to(env.shearable_rod).using(
            sps.FlowForces,
            cosserat_rod_flow_interactor,
        )
    cyl_num_forcing_points = 30
    cylinder_flow_interactor = sps.RigidBodyFlowInteraction(
        num_forcing_points=cyl_num_forcing_points,
        rigid_body=env.cylinder,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
        dx=flow_sim.dx,
        grid_dim=2,
        real_t=real_t,
        forcing_grid_cls=sps.CircularCylinderForcingGrid,
    )
    if rigid_body_coupling_type == "two_way":
        env.simulator.add_forcing_to(env.cylinder).using(
            sps.FlowForces,
            cylinder_flow_interactor,
        )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======

    # =================TIMESTEPPING====================

    # Finalize the pyelastica environment
    _, _ = env.finalize()

    time = 0.0
    foto_timer = 0.0
    foto_timer_limit = period / 10
    rod_time_steps = 1

    # create fig for plotting flow fields
    fig, ax = sps.create_figure_and_axes()

    while time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(f"Vorticity, time: {time:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.x_grid,
                flow_sim.y_grid,
                flow_sim.vorticity_field,
                levels=np.linspace(-5, 5, 100),
                extend="both",
                cmap=sps.lab_cmap,
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            element_position = 0.5 * (
                env.shearable_rod.position_collection[:, 1:]
                + env.shearable_rod.position_collection[:, :-1]
            )
            # for plotting rod with correct radii
            # for reference see
            # https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot
            # -axes-scatter-markersize-by-x-scale/48174228#48174228
            scaling_factor = (
                ax.get_window_extent().width
                / max(flow_sim.x_range, flow_sim.y_range)
                * 72.0
                / fig.dpi
            )
            ax.scatter(
                element_position[0],
                element_position[1],
                s=4 * (scaling_factor * env.shearable_rod.radius) ** 2,
                c="k",
            )
            # plot rod and cylinder forcing points
            ax.scatter(
                cosserat_rod_flow_interactor.forcing_grid.position_field[0],
                cosserat_rod_flow_interactor.forcing_grid.position_field[1],
                s=4,
                color="g",
            )
            ax.scatter(
                cylinder_flow_interactor.forcing_grid.position_field[0],
                cylinder_flow_interactor.forcing_grid.position_field[1],
                s=10,
                color="k",
            )
            sps.save_and_clear_fig(
                fig, ax, cbar, file_name="snap_" + str("%0.4d" % (time * 100)) + ".png"
            )
            print(
                f"time: {time:.2f} ({(time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"rod sub-timesteps: {rod_time_steps}"
            )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.25)

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        # print(flow_dt, rod_dt)
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = time
        for i in range(rod_time_steps):
            # Activate longitudinal muscle
            activation_functions[2].apply_activation(
                env.shearable_rod, activations[2], rod_time
            )
            # Do one elastica step
            env.time_step = local_rod_dt
            rod_time, systems, done = env.step(rod_time, activations)

            # timestep the cosserat_rod_flow_interactor
            cosserat_rod_flow_interactor.time_step(dt=local_rod_dt)
            cylinder_flow_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and rod
        cosserat_rod_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt=flow_dt)

        # update simulation time
        time += flow_dt
        foto_timer += flow_dt

    os.system("rm -f flow.mp4")
    os.system(
        "ffmpeg -r 10 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
        "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
        " flow.mp4"
    )
    os.system("rm -f snap*.png")


if __name__ == "__main__":
    tapered_arm_and_cylinder_flow_coupling(
        final_time_by_period=5.0,
        grid_size=512,
        rod_coupling_type="two_way",
        rigid_body_coupling_type="two_way",
    )

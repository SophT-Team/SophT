import elastica as ea
import numpy as np
import matplotlib.pyplot as plt
from sopht.utils.precision import get_real_t
import sopht_simulator as sps


def immersed_flexible_pendulum_with_rigid_cylinder_case(
    final_time,
    grid_size,
    rod_start_incline_angle,
    coupling_stiffness=-2e4,
    coupling_damping=-1e1,
    rod_coupling_type="two_way",
    rigid_body_coupling_type="one_way",
    num_threads=4,
    precision="single",
):
    # =================PYELASTICA STUFF BEGIN=====================
    class ImmersedFlexiblePendulumSimulator(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping
    ):
        pass

    pendulum_sim = ImmersedFlexiblePendulumSimulator()
    # setting up test params
    n_elem = 40
    start = np.array([0.5, 0.7, 0.0])
    direction = np.array(
        [np.sin(rod_start_incline_angle), -np.cos(rod_start_incline_angle), 0.0]
    )
    normal = np.array([0.0, 0.0, 1.0])
    rod_length = 0.25
    base_radius = 0.0025
    density = 1e3
    youngs_modulus = 1e6
    poisson_ratio = 0.5

    pendulum_rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        rod_length,
        base_radius,
        density,
        0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus,
        shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
    )
    pendulum_sim.append(pendulum_rod)
    pendulum_sim.constrain(pendulum_rod).using(
        ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    # Add gravitational forces
    gravitational_acc = -9.80665
    pendulum_sim.add_forcing_to(pendulum_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )
    # add damping
    dl = rod_length / n_elem
    rod_dt = 0.005 * dl
    damping_constant = 1e-2
    pendulum_sim.dampen(pendulum_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=rod_dt,
    )

    # Initialize rigid cylinder (elastica rigid body) with direction along Z
    X_cm = 0.6
    Y_cm = 0.5
    cyl_start = np.array([X_cm, Y_cm, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    cyl_length = 1.0
    cyl_radius = 0.05
    cylinder_density = 1e-3 * density
    cylinder = ea.Cylinder(
        cyl_start, direction, normal, cyl_length, cyl_radius, cylinder_density
    )
    pendulum_sim.append(cylinder)
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    flow_solver_precision = precision
    real_t = get_real_t(flow_solver_precision)
    grid_size_x = grid_size
    grid_size_y = grid_size_x
    CFL = 0.1
    x_range = 1.0
    # Flow parameters
    vel_scale = np.sqrt(np.fabs(gravitational_acc) * rod_length)
    Re = 500
    nu = rod_length * vel_scale / Re
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

    # ==================FLOW-BODY COMMUNICATORS SETUP START======
    flow_body_interactors = []
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=pendulum_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=2,
        forcing_grid_cls=sps.CosseratRodElementCentricForcingGrid,
        real_t=real_t,
        num_threads=num_threads,
    )
    if rod_coupling_type == "two_way":
        pendulum_sim.add_forcing_to(pendulum_rod).using(
            sps.FlowForces,
            cosserat_rod_flow_interactor,
        )
    flow_body_interactors.append(cosserat_rod_flow_interactor)
    cyl_circumference = 2 * np.pi * cyl_radius
    cyl_num_forcing_points = int(cyl_circumference / rod_length * n_elem)
    cylinder_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=cylinder,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=2,
        real_t=real_t,
        forcing_grid_cls=sps.CircularCylinderForcingGrid,
        num_forcing_points=cyl_num_forcing_points,
    )
    if rigid_body_coupling_type == "two_way":
        pendulum_sim.add_forcing_to(cylinder).using(
            sps.FlowForces,
            cylinder_flow_interactor,
        )
    flow_body_interactors.append(cylinder_flow_interactor)
    # ==================FLOW-BODY COMMUNICATORS SETUP END======

    # =================TIMESTEPPING====================

    pendulum_sim.finalize()
    timestepper = ea.PositionVerlet()
    do_step, stages_and_updates = ea.extend_stepper_interface(timestepper, pendulum_sim)
    time = 0.0
    foto_timer = 0.0
    foto_timer_limit = final_time / 50
    T = []
    rod_force = []
    cylinder_force = []

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
            for flow_body_interactor in flow_body_interactors:
                ax.scatter(
                    flow_body_interactor.forcing_grid.position_field[0],
                    flow_body_interactor.forcing_grid.position_field[1],
                    s=8,
                    color="k",
                )
            sps.save_and_clear_fig(
                fig, ax, cbar, file_name="snap_" + str("%0.4d" % (time * 100)) + ".png"
            )
            grid_dev_error = 0.0
            for flow_body_interactor in flow_body_interactors:
                grid_dev_error += (
                    flow_body_interactor.get_grid_deviation_error_l2_norm()
                )
            print(
                f"time: {time:.2f} ({(time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"grid deviation L2 error: {grid_dev_error:.6f}"
            )

            # dump forces
            T.append(time)
            cylinder_force.append(
                np.linalg.norm(cylinder_flow_interactor.body_flow_forces)
            )
            rod_force.append(
                np.linalg.norm(cosserat_rod_flow_interactor.body_flow_forces)
            )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.25)
        # flow_dt = rod_dt

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = time
        for i in range(rod_time_steps):
            rod_time = do_step(
                timestepper, stages_and_updates, pendulum_sim, rod_time, local_rod_dt
            )
            # timestep the body_flow_interactors
            for flow_body_interactor in flow_body_interactors:
                flow_body_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and bodies
        for flow_body_interactor in flow_body_interactors:
            flow_body_interactor()

        # timestep the flow
        flow_sim.time_step(dt=flow_dt)

        # update simulation time
        time += flow_dt
        foto_timer += flow_dt

    # compile video
    sps.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=10
    )

    plt.figure()
    plt.plot(np.array(T), np.array(cylinder_force), label="force on cylinder")
    plt.plot(np.array(T), np.array(rod_force), label="force on rod")
    plt.legend()
    plt.xlabel("Non-dimensional time")
    plt.ylabel("Force")
    plt.savefig("body_forces_vs_time.png")


if __name__ == "__main__":
    immersed_flexible_pendulum_with_rigid_cylinder_case(
        final_time=3.0,
        grid_size=256,
        rod_start_incline_angle=(np.pi / 2),
        rod_coupling_type="two_way",
        rigid_body_coupling_type="two_way",
    )

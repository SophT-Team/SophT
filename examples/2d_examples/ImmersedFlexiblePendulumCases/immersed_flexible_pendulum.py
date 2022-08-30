import elastica as ea
import numpy as np
import os
from sopht.utils.precision import get_real_t
import sopht_simulator as sps


def immersed_flexible_pendulum_one_way_coupling(
    final_time,
    grid_size,
    rod_start_incline_angle,
    coupling_type="one_way",
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
    n_elem = 25
    start = np.array([0.5, 0.7, 0.0])
    direction = np.array(
        [np.sin(rod_start_incline_angle), -np.cos(rod_start_incline_angle), 0.0]
    )
    normal = np.array([0.0, 0.0, 1.0])
    base_length = 0.25
    base_radius = 0.0025
    density = 1e3
    youngs_modulus = 1e6
    poisson_ratio = 0.5

    pendulum_rod = ea.CosseratRod.straight_rod(
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
    dl = base_length / n_elem
    rod_dt = 0.005 * dl
    damping_constant = 1e-2
    pendulum_sim.dampen(pendulum_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=rod_dt,
    )
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    flow_solver_precision = precision
    real_t = get_real_t(flow_solver_precision)
    grid_size_x = grid_size
    grid_size_y = grid_size_x
    CFL = 0.1
    x_range = 1.0
    # Flow parameters
    vel_scale = np.sqrt(np.fabs(gravitational_acc) * base_length)
    Re = 500
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
    virtual_boundary_stiffness_coeff = real_t(-5e4 * dl)
    virtual_boundary_damping_coeff = real_t(-2e1 * dl)
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=pendulum_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
        dx=flow_sim.dx,
        grid_dim=2,
        forcing_grid_cls=sps.CosseratRodElementCentricForcingGrid,
        real_t=real_t,
        num_threads=num_threads,
    )
    if coupling_type == "two_way":
        pendulum_sim.add_forcing_to(pendulum_rod).using(
            sps.FlowForces,
            cosserat_rod_flow_interactor,
        )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======

    # =================TIMESTEPPING====================

    pendulum_sim.finalize()
    timestepper = ea.PositionVerlet()
    do_step, stages_and_updates = ea.extend_stepper_interface(timestepper, pendulum_sim)
    time = 0.0
    foto_timer = 0.0
    foto_timer_limit = final_time / 50

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
            ax.plot(
                pendulum_rod.position_collection[0],
                pendulum_rod.position_collection[1],
                linewidth=3,
                color="k",
            )
            sps.save_and_clear_fig(
                fig, ax, cbar, file_name="snap_" + str("%0.4d" % (time * 100)) + ".png"
            )
            print(
                f"time: {time:.2f} ({(time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}"
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
            # timestep the cosserat_rod_flow_interactor
            cosserat_rod_flow_interactor.time_step(dt=local_rod_dt)

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
    immersed_flexible_pendulum_one_way_coupling(
        final_time=3.0,
        grid_size=256,
        rod_start_incline_angle=(np.pi / 2),
        coupling_type="two_way",
    )

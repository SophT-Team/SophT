from elastica.boundary_conditions import OneEndFixedBC
from elastica.dissipation import AnalyticalLinearDamper
from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import GravityForces
from elastica.timestepper import PositionVerlet, extend_stepper_interface
from elastica.wrappers import BaseSystemCollection, Constraints, Forcing, Damping

import matplotlib.pyplot as plt

import numpy as np

import os

from sopht.numeric.eulerian_grid_ops import (
    gen_add_fixed_val_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t

from sopht_simulator.immersed_body import CosseratRodFlowInteraction, FlowForces
from sopht_simulator.immersed_body.cosserat_rod import (
    CosseratRodElementCentricForcingGrid,
)
from sopht_simulator.flow.FlowSimulator2D import UnboundedFlowSimulator2D
from sopht_simulator.plot_utils.lab_cmap import lab_cmap


def flow_past_rod_case(
    non_dim_final_time,
    grid_size_x,
    grid_size_y,
    reynolds=200.0,
    nondim_bending_stiffness=1.5e-3,
    mass_ratio=1.5,
    froude=0.5,
    rod_start_incline_angle=0.0,
    num_threads=4,
    precision="single",
):
    # =================COMMON SIMULATOR STUFF=======================
    plt.style.use("seaborn")
    U_free_stream = 1.0
    rho_f = 1.0
    base_length = 1.0
    x_range = 6.0 * base_length
    # =================PYELASTICA STUFF BEGIN=====================
    class FlowPastRodSimulator(BaseSystemCollection, Constraints, Forcing, Damping):
        pass

    flow_past_sim = FlowPastRodSimulator()
    # setting up test params
    n_elem = 40
    start = np.array([base_length, 0.5 * x_range * grid_size_y / grid_size_x, 0.0])
    direction = np.array(
        [np.cos(rod_start_incline_angle), np.sin(rod_start_incline_angle), 0.0]
    )
    normal = np.array([0.0, 0.0, 1.0])
    base_radius = 0.01
    base_area = np.pi * base_radius**2
    z_axis_width = 1.0
    # mass_ratio = rod_line_density / (rho_f * L * z_axis_width)
    rod_line_density = mass_ratio * rho_f * base_length * z_axis_width
    density = rod_line_density / base_area
    moment_of_inertia = np.pi / 4 * base_radius**4
    # Kb = E I / (rho_f U^2 L^3 * z_axis_width)
    youngs_modulus = (
        nondim_bending_stiffness
        * (rho_f * U_free_stream**2 * base_length**3 * z_axis_width)
        / moment_of_inertia
    )
    poisson_ratio = 0.5
    # Fr = g L^2 / U
    gravitational_acc = froude * U_free_stream**2 / base_length
    print(f"density:{density}")
    print(f"youngs modulus:{youngs_modulus}")
    print(f"gravitational acceleration:{gravitational_acc}")

    flow_past_rod = CosseratRod.straight_rod(
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
    tip_y_start = flow_past_rod.position_collection[1, -1]
    flow_past_sim.append(flow_past_rod)
    flow_past_sim.constrain(flow_past_rod).using(
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    # Add gravitational forces
    flow_past_sim.add_forcing_to(flow_past_rod).using(
        GravityForces, acc_gravity=np.array([gravitational_acc, 0.0, 0.0])
    )
    # add damping
    dl = base_length / n_elem
    rod_dt = 0.01 * dl
    damping_constant = 0.5e-3
    flow_past_sim.dampen(flow_past_rod).using(
        AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=rod_dt,
    )
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    flow_solver_precision = precision
    real_t = get_real_t(flow_solver_precision)
    CFL = 0.1
    # Flow parameters
    # Re = U * L / nu
    nu = base_length * U_free_stream / reynolds
    flow_sim = UnboundedFlowSimulator2D(
        grid_size=(grid_size_y, grid_size_x),
        x_range=x_range,
        kinematic_viscosity=nu,
        CFL=CFL,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        num_threads=num_threads,
    )
    # Compile additional kernels
    # TODO put in flow sim
    add_fixed_val = gen_add_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(grid_size_y, grid_size_x),
        num_threads=num_threads,
        field_type="vector",
    )
    # ==================FLOW SETUP END=========================

    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    virtual_boundary_stiffness_coeff = real_t(-5e4 * dl)
    virtual_boundary_damping_coeff = real_t(-2e1 * dl)
    cosserat_rod_flow_interactor = CosseratRodFlowInteraction(
        cosserat_rod=flow_past_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
        dx=flow_sim.dx,
        grid_dim=2,
        forcing_grid_cls=CosseratRodElementCentricForcingGrid,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_past_sim.add_forcing_to(flow_past_rod).using(
        FlowForces,
        cosserat_rod_flow_interactor,
    )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======

    # =================TIMESTEPPING====================

    flow_past_sim.finalize()
    timestepper = PositionVerlet()
    do_step, stages_and_updates = extend_stepper_interface(timestepper, flow_past_sim)
    time = 0.0
    foto_timer = 0.0
    timescale = base_length / U_free_stream
    final_time = non_dim_final_time * timescale
    foto_timer_limit = final_time / 50

    # setup freestream ramping
    ramp_timescale = 0.0125 * final_time
    V_free_stream_perturb = 0.5 * U_free_stream

    data_timer = 0.0
    data_timer_limit = 0.1 * timescale
    tip_time = []
    tip_y = []

    while time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            fig = plt.figure(frameon=True, dpi=150)
            ax = fig.add_subplot(111)
            plt.contourf(
                flow_sim.x_grid,
                flow_sim.y_grid,
                flow_sim.vorticity_field,
                levels=np.linspace(-5, 5, 100),
                extend="both",
                cmap=lab_cmap,
            )
            plt.colorbar()
            plt.plot(
                flow_past_rod.position_collection[0],
                flow_past_rod.position_collection[1],
                linewidth=3,
                color="k",
            )
            ax.set_aspect(aspect=1)
            ax.set_title(f"Vorticity, time: {time:.2f}")
            plt.savefig(
                "snap_" + str("%0.4d" % (time * 100)) + ".png",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.clf()
            plt.close("all")
            print(
                f"time: {time:.2f} ({(time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}"
            )

        # save diagnostic data
        if data_timer >= data_timer_limit or data_timer == 0:
            data_timer = 0.0
            tip_time.append(time / timescale)
            tip_y.append(
                (flow_past_rod.position_collection[1, -1] - tip_y_start) / base_length
            )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.5)
        # flow_dt = rod_dt

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

        # timestep the flow
        flow_sim.time_step(dt=flow_dt)

        # add freestream
        ramp_factor = np.exp(-time / ramp_timescale)
        # TODO merge later into flow sim
        add_fixed_val(
            sum_field=flow_sim.velocity_field,
            vector_field=flow_sim.velocity_field,
            fixed_vals=[
                U_free_stream * (1.0 - ramp_factor),
                V_free_stream_perturb * ramp_factor,
            ],
        )

        # update simulation time
        time += flow_dt
        foto_timer += flow_dt
        data_timer += flow_dt

    os.system("rm -f flow.mp4")
    os.system(
        "ffmpeg -r 10 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
        "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
        " flow.mp4"
    )
    os.system("rm -f snap*.png")

    plt.figure()
    plt.plot(np.array(tip_time), np.array(tip_y))
    plt.ylim([-0.35, 0.35])
    plt.xlabel("Non-dimensional time")
    plt.ylabel("Tip deflection")
    plt.savefig("tip_y_vs_time.png")


if __name__ == "__main__":
    flow_past_rod_case(
        non_dim_final_time=60.0,
        grid_size_x=512,
        grid_size_y=256,
    )

from cosserat_rod_support.CosseratRodFlowInteraction import CosseratRodFlowInteraction
from cosserat_rod_support.flow_forces import FlowForces

from elastica.boundary_conditions import OneEndFixedBC
from elastica.dissipation import ExponentialDamper, LaplaceDissipationFilter
from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import GravityForces
from elastica.timestepper import PositionVerlet, extend_stepper_interface
from elastica.wrappers import BaseSystemCollection, Constraints, Forcing, Damping

from flow_algo_assembly.flow_solver_steps import (
    gen_full_flow_timestep_with_forcing_and_boundary_penalisation_kernel_2d,
)
from flow_algo_assembly.timestep_limits import compute_advection_diffusion_timestep

import matplotlib.pyplot as plt

import numpy as np

import os

from plot_utils.lab_cmap import lab_cmap

from sopht.utils.precision import get_real_t


def immersed_flexible_pendulum_one_way_coupling(
    final_time,
    grid_size,
    rod_start_incline_angle,
    coupling_type="one_way",
    num_threads=4,
    precision="single",
):
    # =================COMMON SIMULATOR STUFF=======================
    grid_size_x = grid_size
    grid_size_y = grid_size_x
    flow_solver_precision = precision
    real_t = get_real_t(flow_solver_precision)
    CFL = 0.1
    plt.style.use("seaborn")

    # =================PYELASTICA STUFF BEGIN=====================
    class ImmersedFlexiblePendulumSimulator(
        BaseSystemCollection, Constraints, Forcing, Damping
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

    pendulum_rod = CosseratRod.straight_rod(
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
        OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    # Add gravitational forces
    gravitational_acc = -9.80665
    pendulum_sim.add_forcing_to(pendulum_rod).using(
        GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )
    # add damping
    dl = base_length / n_elem
    rod_dt = 0.005 * dl
    damping_constant = 1e-2
    pendulum_sim.dampen(pendulum_rod).using(
        ExponentialDamper,
        damping_constant=damping_constant,
        time_step=rod_dt,
    )
    # pendulum_sim.dampen(pendulum_rod).using(
    #     LaplaceDissipationFilter,
    #     filter_order=5,
    # )
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    # Initialize 2D domain
    grid_size_y_by_x = grid_size_y / grid_size_x
    dx = real_t(1.0 / grid_size_x)
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, grid_size_x).astype(real_t)
    y = np.linspace(
        eul_grid_shift, grid_size_y_by_x - eul_grid_shift, grid_size_y
    ).astype(real_t)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flow parameters
    vel_scale = np.sqrt(np.fabs(gravitational_acc) * base_length)
    Re = 500
    nu = base_length * vel_scale / Re

    # Initialize flow field
    vorticity_field = np.zeros_like(x_grid)
    velocity_field = np.zeros((2, grid_size_y, grid_size_x), dtype=real_t)
    # we use the same buffer for advection, diffusion and velocity recovery
    buffer_scalar_field = np.zeros_like(vorticity_field)
    # this one holds the forcing from bodies
    eul_grid_forcing_field = np.zeros_like(velocity_field)

    # Compile kernels
    full_flow_timestep = (
        gen_full_flow_timestep_with_forcing_and_boundary_penalisation_kernel_2d(
            real_t=real_t,
            dx=dx,
            nu=nu,
            grid_size=(grid_size_y, grid_size_x),
            num_threads=num_threads,
            penalty_zone_width=2,
            x_grid=x_grid,
            y_grid=y_grid,
        )
    )
    # ==================FLOW SETUP END=========================

    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    virtual_boundary_stiffness_coeff = real_t(-5e4 * dl)
    virtual_boundary_damping_coeff = real_t(-2e1 * dl)
    # cosserat_rod_flow_interactor = CosseratRodFlowInteraction(
    cosserat_rod_flow_interactor = CosseratRodFlowInteraction(
        cosserat_rod=pendulum_rod,
        eul_grid_forcing_field=eul_grid_forcing_field,
        eul_grid_velocity_field=velocity_field,
        virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
        dx=dx,
        grid_dim=2,
        real_t=real_t,
        enable_eul_grid_forcing_reset=True,
        num_threads=num_threads,
        forcing_grid_type="nodal",
    )
    if coupling_type == "two_way":
        pendulum_sim.add_forcing_to(pendulum_rod).using(
            FlowForces,
            cosserat_rod_flow_interactor,
        )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======

    # =================TIMESTEPPING====================

    pendulum_sim.finalize()
    timestepper = PositionVerlet()
    do_step, stages_and_updates = extend_stepper_interface(timestepper, pendulum_sim)
    time = 0.0
    foto_timer = 0.0
    foto_timer_limit = final_time / 50

    while time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            fig = plt.figure(frameon=True, dpi=150)
            ax = fig.add_subplot(111)
            plt.contourf(
                x_grid,
                y_grid,
                vorticity_field,
                levels=np.linspace(-5, 5, 100),
                extend="both",
                cmap=lab_cmap,
            )
            plt.colorbar()
            plt.plot(
                pendulum_rod.position_collection[0],
                pendulum_rod.position_collection[1],
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
                f"max_vort: {np.amax(vorticity_field):.4f}"
            )

        # compute timestep
        flow_dt = compute_advection_diffusion_timestep(
            velocity_field=velocity_field,
            CFL=CFL,
            nu=nu,
            dx=dx,
            dt_prefac=0.25,
        )
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
        full_flow_timestep(
            eul_grid_forcing_field=eul_grid_forcing_field,
            field=vorticity_field,
            velocity_field=velocity_field,
            flux_buffer=buffer_scalar_field,
            dt=flow_dt,
            forcing_prefactor=flow_dt,
            vorticity_field=vorticity_field,
            stream_func_field=buffer_scalar_field,
            field_to_penalise=vorticity_field,
        )

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

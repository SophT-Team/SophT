import elastica as ea
import matplotlib.pyplot as plt
import numpy as np
import os
from sopht.numeric.eulerian_grid_ops import (
    gen_add_fixed_val_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t
import sopht_simulator as sps


def flow_past_rod_case(
    non_dim_final_time,
    grid_size_x,
    grid_size_y,
    reynolds=100.0,
    # Kb = 1.4e6 * (2e-2**3) / 12 / (0.35**3) / 1e3 / (1.0**2)
    nondim_bending_stiffness=0.021768707482993206,
    # mass_ratio = 1e4 * 2e-2 / 1e3 / 0.35
    mass_ratio=0.5714285714285715,
    cyl_diameter_to_rod_length=(0.1 / 0.35),
    beam_aspect_ratio=(0.2 / 3.5),
    coupling_stiffness=-6.25e2,
    coupling_damping=-0.25,
    num_threads=4,
    flow_precision="single",
    save_diagnostic=False,
):
    # =================COMMON SIMULATOR STUFF=======================
    U_free_stream = 1.0
    rho_f = 1.0
    base_length = 1.0
    x_range = 5.0 * base_length
    # =================PYELASTICA STUFF BEGIN=====================
    class FlowPastRodSimulator(ea.BaseSystemCollection, ea.Constraints, ea.Forcing):
        pass

    flow_past_sim = FlowPastRodSimulator()
    # setting up test params
    n_elem = 80
    start = np.array(
        [
            1.0 * base_length,
            0.5 * x_range * grid_size_y / grid_size_x,
            0.0,
        ]
    )
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    base_radius = beam_aspect_ratio * base_length / 2.0
    base_area = np.pi * base_radius**2
    # mass_ratio = rod_line_density / (rho_f * L)
    rod_line_density = mass_ratio * rho_f * base_length
    density = rod_line_density / base_area
    moment_of_inertia = np.pi / 4 * base_radius**4
    # Kb = E I / (rho_f U^2 L^3)
    # for converting plate to rod stiffness
    # see Gilmanov et al. 2015
    # poisson_ratio_plate = 0.4
    # poisson_ratio_correction_factor = 1.0 - poisson_ratio_plate**2
    # TODO after convergence see if we really need it?
    # nondim_bending_stiffness /= poisson_ratio_correction_factor
    youngs_modulus = (
        nondim_bending_stiffness
        * (rho_f * U_free_stream**2 * base_length**3)
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
    tip_start_position = flow_past_rod.position_collection[:2, -1]
    flow_past_sim.append(flow_past_rod)
    flow_past_sim.constrain(flow_past_rod).using(
        ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )
    dl = base_length / n_elem
    rod_dt = 1e-3 * dl
    # =================PYELASTICA STUFF END=====================

    # ==================FLOW SETUP START=========================
    real_t = get_real_t(flow_precision)
    CFL = 0.1
    # Flow parameters
    # Re = U * (square_side_length) / nu
    cyl_diameter = base_length * cyl_diameter_to_rod_length
    nu = cyl_diameter * U_free_stream / reynolds
    flow_sim = sps.UnboundedFlowSimulator2D(
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
    # Initialise the top and bottom walls as fixed rods
    wall_boundary_offset = 4 * flow_sim.dx  # to avoid interpolation at boundaries
    wall_length = x_range - 2 * wall_boundary_offset
    wall_n_elem = n_elem * int(x_range / base_length)
    top_wall_start = np.array(
        [
            wall_boundary_offset,
            flow_past_rod.position_collection[1, 0] + 2.0 * cyl_diameter,
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
            flow_past_rod.position_collection[1, 0] - 2.0 * cyl_diameter,
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
    X_cm = flow_past_rod.position_collection[0, 0] - 0.5 * cyl_diameter
    Y_cm = flow_past_rod.position_collection[1, 0]
    start = np.array([X_cm, Y_cm, 0.0])
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
    flow_body_interactors = []
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=flow_past_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=2,
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
        grid_dim=2,
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
        grid_dim=2,
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
        grid_dim=2,
        forcing_grid_cls=sps.CosseratRodNodalForcingGrid,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_body_interactors.append(bottom_wall_flow_interactor)
    # ==================FLOW-ROD COMMUNICATOR SETUP END======
    # =================TIMESTEPPING====================
    flow_past_sim.finalize()
    timestepper = ea.PositionVerlet()
    do_step, stages_and_updates = ea.extend_stepper_interface(
        timestepper, flow_past_sim
    )
    time = 0.0
    foto_timer = 0.0
    timescale = base_length / U_free_stream
    final_time = non_dim_final_time * timescale
    foto_timer_limit = final_time / 50

    # setup freestream ramping
    ramp_timescale = 0.5 * timescale
    V_free_stream_perturb = 0.5 * U_free_stream

    data_timer = 0.0
    data_timer_limit = 0.05 * timescale
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
                flow_sim.x_grid,
                flow_sim.y_grid,
                flow_sim.vorticity_field,
                levels=np.linspace(-15, 15, 100),
                extend="both",
                cmap=sps.lab_cmap,
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.plot(
                flow_past_rod.position_collection[0],
                flow_past_rod.position_collection[1],
                linewidth=3,
                color="k",
            )
            for flow_body_interactor in flow_body_interactors:
                ax.scatter(
                    flow_body_interactor.forcing_grid.position_field[0],
                    flow_body_interactor.forcing_grid.position_field[1],
                    s=5,
                    color="k",
                )
            sps.save_and_clear_fig(
                fig, ax, cbar, file_name="snap_" + str("%0.4d" % (time * 100)) + ".png"
            )
            lag_grid_dev = 0.0
            for flow_body_interactor in flow_body_interactors:
                lag_grid_dev += np.linalg.norm(
                    flow_body_interactor.lag_grid_position_mismatch_field
                ) / np.sqrt(flow_body_interactor.forcing_grid.num_lag_nodes)
            print(
                f"time: {time:.2f} ({(time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"lag grid deviation: {lag_grid_dev:.8f}"
            )

        # save diagnostic data
        if data_timer > data_timer_limit or data_timer == 0.0:
            data_timer = 0.0
            tip_time.append(time / timescale)
            tip_position.append(
                (flow_past_rod.position_collection[:2, -1] - tip_start_position)
                / cyl_diameter
            )
            # print(tip_position[-1])

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.25)
        # flow_dt = rod_dt

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = time
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
        flow_sim.time_step(dt=flow_dt)

        # add freestream
        ramp_factor = 1.0 - np.exp(-time / ramp_timescale)
        # TODO merge later into flow sim
        add_fixed_val(
            sum_field=flow_sim.velocity_field,
            vector_field=flow_sim.velocity_field,
            fixed_vals=[
                U_free_stream * ramp_factor,
                V_free_stream_perturb * (1.0 - ramp_factor),
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
    plt.plot(np.array(tip_time), np.array(tip_position)[..., 0], label="X")
    plt.plot(np.array(tip_time), np.array(tip_position)[..., 1], label="Y")
    plt.legend()
    plt.xlabel("Non-dimensional time")
    plt.ylabel("Tip deflection")
    plt.savefig("tip_position_vs_time.png")

    if save_diagnostic:
        np.savetxt(
            fname="tip_position_vs_time.csv",
            X=np.c_[
                np.array(tip_time),
                np.array(tip_position)[..., 0],
                np.array(tip_position)[..., 1],
            ],
            header="time, tip_x, tip_y",
            delimiter=",",
        )


if __name__ == "__main__":
    grid_size_x = 512
    flow_past_rod_case(
        non_dim_final_time=25.0,
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_x // 2,
        save_diagnostic=True,
    )

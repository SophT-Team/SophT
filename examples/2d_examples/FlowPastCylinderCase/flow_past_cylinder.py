import elastica as ea
import matplotlib.pyplot as plt
import numpy as np
import os
from sopht.numeric.eulerian_grid_ops import (
    gen_add_fixed_val_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t
import sopht_simulator as sps


def flow_past_cylinder_boundary_forcing_case(
    grid_size_x,
    grid_size_y,
    num_threads=4,
    precision="single",
    save_diagnostic=False,
):
    """
    This example considers a simple flow past cylinder using immersed
    boundary forcing.
    """
    real_t = get_real_t(precision)
    # Flow parameters
    U_inf = real_t(1.0)
    velocity_free_stream = np.zeros(2)
    velocity_free_stream[0] = U_inf
    cyl_radius = real_t(0.03)
    Re = 200
    nu = cyl_radius * U_inf / Re
    CFL = real_t(0.1)
    x_range = 1.0

    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=(grid_size_y, grid_size_x),
        x_range=x_range,
        kinematic_viscosity=nu,
        CFL=CFL,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        num_threads=num_threads,
    )

    # Initialize fixed cylinder (elastica rigid body) with direction along Z
    X_cm = real_t(2.5) * cyl_radius
    Y_cm = real_t(0.5) * grid_size_y / grid_size_x
    start = np.array([X_cm, Y_cm, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    density = 1e3
    cylinder = ea.Cylinder(start, direction, normal, base_length, cyl_radius, density)
    # Since the cylinder is fixed, we dont add it to pyelastica simulator,
    # and directly use it for setting up the flow interactor.

    # Compile additional kernels
    # TODO put in flow sim
    add_fixed_val = gen_add_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(grid_size_y, grid_size_x),
        num_threads=num_threads,
        field_type="vector",
    )

    # ==================FLOW-BODY COMMUNICATOR SETUP START======
    num_lag_nodes = 60
    dtheta = 2.0 * np.pi / num_lag_nodes
    ds = cyl_radius * real_t(dtheta)
    virtual_boundary_stiffness_coeff = real_t(-5e4 * ds)
    virtual_boundary_damping_coeff = real_t(-2e1 * ds)
    cylinder_flow_interactor = sps.RigidBodyFlowInteraction(
        num_forcing_points=num_lag_nodes,
        rigid_body=cylinder,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=virtual_boundary_damping_coeff,
        dx=flow_sim.dx,
        grid_dim=2,
        forcing_grid_cls=sps.CircularCylinderForcingGrid,
        real_t=real_t,
    )
    # ==================FLOW-BODY COMMUNICATOR SETUP END======

    # iterate
    timescale = cyl_radius / U_inf
    t_end_hat = real_t(200.0)  # non-dimensional end time
    t_end = t_end_hat * timescale  # dimensional end time
    t = real_t(0.0)
    foto_timer = 0.0
    foto_timer_limit = t_end / 50

    data_timer = 0.0
    data_timer_limit = 0.25 * timescale
    time = []
    drag_coeffs = []
    lag_grid_deviation = []

    # create fig for plotting flow fields
    fig, ax = sps.create_figure_and_axes()

    while t < t_end:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(f"Vorticity, time: {t / timescale:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.x_grid,
                flow_sim.y_grid,
                flow_sim.vorticity_field,
                levels=np.linspace(-25, 25, 100),
                extend="both",
                cmap=sps.lab_cmap,
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.scatter(
                cylinder_flow_interactor.forcing_grid.position_field[0],
                cylinder_flow_interactor.forcing_grid.position_field[1],
                s=4,
                color="k",
            )
            sps.save_and_clear_fig(
                fig, ax, cbar, file_name="snap_" + str("%0.4d" % (t * 100)) + ".png"
            )
            lag_grid_dev = np.linalg.norm(
                cylinder_flow_interactor.lag_grid_position_mismatch_field
            ) / np.sqrt(num_lag_nodes)
            print(
                f"time: {t:.2f} ({(t/t_end*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"lag grid deviation: {lag_grid_dev:.8f}"
            )

        # track diagnostic data
        if data_timer >= data_timer_limit or data_timer == 0:
            data_timer = 0.0
            time.append(t / timescale)

            # compute lag grid deviation
            lag_grid_deviation.append(
                np.linalg.norm(
                    cylinder_flow_interactor.lag_grid_position_mismatch_field
                )
                / np.sqrt(num_lag_nodes)
            )

            # calculate drag
            F = np.sum(cylinder_flow_interactor.lag_grid_forcing_field[0, ...])
            drag_coeff = np.fabs(F) / U_inf / U_inf / cyl_radius
            drag_coeffs.append(drag_coeff)

        dt = flow_sim.compute_stable_timestep()

        # compute flow forcing and timestep forcing
        cylinder_flow_interactor.time_step(dt=dt)
        cylinder_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt)

        # add freestream
        # TODO merge later into flow sim
        add_fixed_val(
            sum_field=flow_sim.velocity_field,
            vector_field=flow_sim.velocity_field,
            fixed_vals=velocity_free_stream,
        )

        # update time
        t = t + dt
        foto_timer += dt
        data_timer += dt

    # compile video
    os.system("rm -f flow.mp4")
    os.system(
        "ffmpeg -r 10 -s 3840x2160 -f image2 -pattern_type glob -i 'snap*.png' "
        "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2'"
        " flow.mp4"
    )
    os.system("rm -f snap*.png")

    plt.figure()
    plt.plot(np.array(time), np.array(drag_coeffs))
    plt.ylim([0.7, 1.7])
    plt.xlabel("Non-dimensional time")
    plt.ylabel("Drag coefficient, Cd")
    plt.savefig("drag_vs_time.png")
    if save_diagnostic:
        np.savetxt(
            "drag_vs_time.csv",
            np.c_[np.array(time), np.array(drag_coeffs)],
            delimiter=",",
        )

    plt.figure()
    plt.plot(np.array(time), np.array(lag_grid_deviation))
    plt.xlabel("Non-dimensional time")
    plt.ylabel("Lagrangian grid deviation error")
    plt.savefig("lag_grid_dev_vs_time.png")


if __name__ == "__main__":
    grid_size_x = 512
    grid_size_y = 256
    flow_past_cylinder_boundary_forcing_case(
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        save_diagnostic=True,
    )

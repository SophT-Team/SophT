import numpy as np
import sopht.simulator as sps
import sopht.utils as spu
from elastic_fish import ElasticFishSimulator
import os


def elastic_fish_swimming_case_2d(
    non_dim_final_time: float,
    n_elem: int,
    grid_size: tuple[int, int],
    mass_ratio: float = 1.0,
    coupling_stiffness: float = -1.6e4,
    coupling_damping: float = -16,
    num_threads: int = 4,
    precision: str = "single",
    save_data: bool = False,
    muscle_torque_coefficients=None,
) -> None:
    grid_dim = 2
    grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    # rho_f = 1
    base_length = 1.0
    x_range = 4 * base_length
    y_range = grid_size_y / grid_size_x * x_range
    # =================PYELASTICA STUFF BEGIN=====================
    print(f"Running with muscle torque coeff: {muscle_torque_coefficients}")
    period = 1
    final_time = period * non_dim_final_time
    start = np.array([0.75 * x_range - 0.5 * base_length, 0.5 * y_range, 0.0])
    # rho_s = mass_ratio * rho_f
    rho_s = 1e3
    youngs_modulus = 4e5
    fish_sim = ElasticFishSimulator(
        final_time=final_time,
        period=period,
        muscle_torque_coefficients=muscle_torque_coefficients,
        n_elements=n_elem,
        rod_density=rho_s,
        youngs_modulus=youngs_modulus,
        base_length=base_length,
        start=start,
    )
    # =================PYELASTICA STUFF END=====================
    # ==================FLOW SETUP START=========================
    # Flow parameters
    vel_scale = base_length / period
    reynolds = 2000
    kinematic_viscosity = base_length * vel_scale / reynolds
    # kinematic_viscosity = 1.4e-4
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=kinematic_viscosity,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        num_threads=num_threads,
    )
    # ==================FLOW SETUP END=========================
    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    cosserat_rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=fish_sim.shearable_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=coupling_stiffness,
        virtual_boundary_damping_coeff=coupling_damping,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        num_threads=num_threads,
        forcing_grid_cls=sps.CosseratRodEdgeForcingGrid,
    )
    fish_sim.simulator.add_forcing_to(fish_sim.shearable_rod).using(
        sps.FlowForces,
        cosserat_rod_flow_interactor,
    )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======
    # =================TIMESTEPPING====================
    # Finalize the pyelastica environment
    fish_sim.finalize()

    foto_timer = 0.0
    foto_timer_limit = period / 10
    fish_vel = []

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()

    while flow_sim.time < final_time:

        # Plot solution
        if foto_timer >= foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            if save_data:
                ax.set_title(
                    f"Vorticity, time: {flow_sim.time:.2f}, "
                    f"distance: {(fish_sim.shearable_rod.position_collection[0, 0] - start[0]):.6f}"
                )
                contourf_obj = ax.contourf(
                    flow_sim.position_field[x_axis_idx],
                    flow_sim.position_field[y_axis_idx],
                    flow_sim.vorticity_field,
                    levels=np.linspace(-5, 5, 100),
                    extend="both",
                    cmap=spu.get_lab_cmap(),
                )
                cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
                element_position = 0.5 * (
                    fish_sim.shearable_rod.position_collection[:, 1:]
                    + fish_sim.shearable_rod.position_collection[:, :-1]
                )
                # for plotting rod with correct radii for reference see
                # https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot
                # -axes-scatter-markersize-by-x-scale/48174228#48174228
                scaling_factor = (
                    ax.get_window_extent().width
                    / max(flow_sim.x_range, flow_sim.y_range)
                    * 72.0
                    / fig.dpi
                )
                ax.scatter(
                    element_position[x_axis_idx],
                    element_position[y_axis_idx],
                    s=4 * (scaling_factor * fish_sim.shearable_rod.radius) ** 2,
                    c="k",
                )
                # plot rod and cylinder forcing points
                # ax.scatter(
                #     cosserat_rod_flow_interactor.forcing_grid.position_field[x_axis_idx],
                #     cosserat_rod_flow_interactor.forcing_grid.position_field[y_axis_idx],
                #     s=5,
                #     color="g",
                # )
                spu.save_and_clear_fig(
                    fig,
                    ax,
                    cbar,
                    file_name="snap_" + str("%0.4d" % (flow_sim.time * 100)) + ".png",
                )
            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"grid deviation L2 error: {cosserat_rod_flow_interactor.get_grid_deviation_error_l2_norm():.6f}, "
                f"fish pos: {fish_sim.shearable_rod.position_collection[0, 0]:.6f}, "
                f"flow_dt: {flow_sim.compute_stable_timestep(dt_prefac=0.25):.6f}"
            )

            # save diagnostics
            if flow_sim.time >= final_time - 2 * period:
                fish_vel.append(
                    fish_sim.shearable_rod.compute_velocity_center_of_mass()[0]
                )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.25)
        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, fish_sim.dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            rod_time = fish_sim.time_step(rod_time, local_rod_dt)
            # timestep the cosserat_rod_flow_interactor
            cosserat_rod_flow_interactor.time_step(dt=local_rod_dt)
        # evaluate feedback/interaction between flow and rod
        cosserat_rod_flow_interactor()

        # timestep the flow
        flow_sim.time_step(dt=flow_dt)

        # update timer
        foto_timer += flow_dt

    # compile video
    if save_data:
        spu.make_video_from_image_series(
            video_name="flow", image_series_name="snap", frame_rate=10
        )

    print(np.mean(fish_vel))
    return np.mean(fish_vel)


if __name__ == "__main__":
    nx = 256
    ny = nx // 4
    n_elem = nx // 8 * 2

    period = 1.0
    final_time = 12.0 * period

    if os.path.exists("optimized_coefficients.txt"):
        muscle_torque_coefficients = np.genfromtxt(
            "optimized_coefficients.txt", delimiter=","
        )
    elif os.path.exists("outcmaes/xrecentbest.dat"):
        muscle_torque_coefficients = np.loadtxt("outcmaes/xrecentbest.dat", skiprows=1)[
            -1, 5:
        ]
    else:
        muscle_torque_coefficients = np.array([1.51, 0.48, 5.74, 2.73, 1.44])

    elastic_fish_swimming_case_2d(
        non_dim_final_time=final_time,
        grid_size=(ny, nx),
        n_elem=n_elem,
        mass_ratio=1.0,
        save_data=True,
        muscle_torque_coefficients=muscle_torque_coefficients,
    )

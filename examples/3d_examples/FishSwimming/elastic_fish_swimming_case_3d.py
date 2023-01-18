import elastica as ea
import numpy as np
import sopht.simulator as sps
import sopht.utils as spu
import click
from elastic_fish import ElasticFishSimulator
from fish_grid import FishSurfaceForcingGrid
import os


def elastic_fish_swimming_case(
    non_dim_final_time: float,
    n_elem: int,
    grid_size: tuple[int, int, int],
    surface_grid_density_for_largest_element: int,
    mass_ratio: float = 1.0,
    coupling_stiffness: float = -2e4,
    coupling_damping: float = -1e1,
    num_threads: int = 4,
    precision: str = "single",
    save_data: bool = False,
    muscle_torque_coefficients=None,
) -> None:
    # =================COMMON SIMULATOR STUFF=======================
    grid_dim = 3
    grid_size_z, grid_size_y, grid_size_x = grid_size
    real_t = spu.get_real_t(precision)
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    # rho_f = 1
    base_length = 1.0
    x_range = 4 * base_length
    y_range = grid_size_y / grid_size_x * x_range
    z_range = grid_size_z / grid_size_x * x_range
    # =================PYELASTICA STUFF BEGIN=====================
    period = 1
    final_time = non_dim_final_time * period
    start = np.array([0.75 * x_range - 0.5 * base_length, 0.5 * y_range, 0.5 * z_range])
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
    # kinematic_viscosity = U_free_stream * base_diameter / reynolds
    vel_scale = base_length / period
    reynolds = 2000
    kinematic_viscosity = base_length * vel_scale / reynolds
    # kinematic_viscosity = 1e-4
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=kinematic_viscosity,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        num_threads=num_threads,
        filter_vorticity=True,
        filter_setting_dict={"order": 1, "type": "multiplicative"},
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
        forcing_grid_cls=FishSurfaceForcingGrid,
        surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
    )
    fish_sim.simulator.add_forcing_to(fish_sim.shearable_rod).using(
        sps.FlowForces,
        cosserat_rod_flow_interactor,
    )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======
    # =================TIMESTEPPING====================
    fish_sim.finalize()

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
        io = spu.IO(dim=grid_dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field, velocity=flow_sim.velocity_field
        )
        # Initialize sphere IO
        rod_io = spu.IO(dim=grid_dim, real_dtype=real_t)
        rod_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=cosserat_rod_flow_interactor.forcing_grid.position_field,
            lagrangian_grid_name="fish",
            vector_3d=cosserat_rod_flow_interactor.lag_grid_forcing_field,
        )

    foto_timer = 0.0
    foto_timer_limit = period / 30
    timescale = period
    time_history = []
    force_history = []

    # create fig for plotting flow fields
    fig, ax = spu.create_figure_and_axes()

    # iterate
    while flow_sim.time < final_time:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            ax.set_title(
                f"Vorticity z, time: {flow_sim.time / timescale:.2f}, "
                f"distance: {(fish_sim.shearable_rod.position_collection[0, 0] - start[0]):.6f}"
            )
            contourf_obj = ax.contourf(
                flow_sim.position_field[x_axis_idx, grid_size_z // 2, :, :],
                flow_sim.position_field[y_axis_idx, grid_size_z // 2, :, :],
                flow_sim.vorticity_field[z_axis_idx, grid_size_z // 2, :, :],
                levels=np.linspace(-25, 25, 100),
                extend="both",
                cmap=spu.get_lab_cmap(),
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            ax.scatter(
                cosserat_rod_flow_interactor.forcing_grid.position_field[x_axis_idx],
                cosserat_rod_flow_interactor.forcing_grid.position_field[y_axis_idx],
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
            forces = np.sum(cosserat_rod_flow_interactor.lag_grid_forcing_field, axis=1)
            force_history.append(forces.copy())
            print(
                f"time: {flow_sim.time:.2f} ({(flow_sim.time/final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"vort divg. L2 norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f},"
                " grid deviation L2 error: "
                f"{cosserat_rod_flow_interactor.get_grid_deviation_error_l2_norm():.6f},"
                f" fish pos: {fish_sim.shearable_rod.position_collection[0, 0]:.6f},"
                f" total force: {forces},"
                f" force norm: {np.linalg.norm(forces)}"
            )
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

        flow_sim.time_step(dt=flow_dt)

        # update timer
        foto_timer += flow_dt

    # compile video
    spu.make_video_from_image_series(
        video_name="flow", image_series_name="snap", frame_rate=30
    )

    # Save data
    np.savetxt(
        "fish_forces_vs_time.csv",
        np.c_[
            np.array(time_history),
            np.array(force_history),
            np.linalg.norm(np.array(force_history), axis=1),
        ],
        delimiter=",",
        header="time, force x, force y, force z, force norm",
    )


if __name__ == "__main__":

    @click.command()
    @click.option("--num_threads", default=4, help="Number of threads for parallelism.")
    @click.option("--nx", default=128, help="Number of grid points in x direction.")
    def simulate_fish_swimming(num_threads: int, nx: int) -> None:
        ny = nx // 4
        nz = nx // 4
        # in order Z, Y, X
        grid_size = (nz, ny, nx)
        surface_grid_density_for_largest_element = nx // 8
        # surface_grid_density_for_largest_element = 100
        n_elem = nx // 8 * 2

        period = 1.0
        final_time = 12.0 * period

        if os.path.exists("optimized_coefficients.txt"):
            muscle_torque_coefficients = np.genfromtxt(
                "optimized_coefficients.txt", delimiter=","
            )
        elif os.path.exists("outcmaes/xrecentbest.dat"):
            muscle_torque_coefficients = np.loadtxt(
                "outcmaes/xrecentbest.dat", skiprows=1
            )[-1, 5:]
        else:
            muscle_torque_coefficients = np.array([1.51, 0.48, 5.74, 2.73, 1.44])

        elastic_fish_swimming_case(
            non_dim_final_time=final_time,
            n_elem=n_elem,
            grid_size=grid_size,
            surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
            mass_ratio=1.0,
            save_data=True,
            muscle_torque_coefficients=muscle_torque_coefficients,
        )

    simulate_fish_swimming()

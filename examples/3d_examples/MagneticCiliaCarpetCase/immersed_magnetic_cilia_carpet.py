import numpy as np
from sopht.utils.precision import get_real_t
import sopht_simulator as sps
from sopht.utils.IO import IO
import os
from magnetic_cilia_carpet import MagneticCiliaCarpetSimulator


def immersed_magnetic_cilia_carpet_case(
    cilia_carpet_simulator,
    domain_range,
    grid_size_x,
    reynolds=100.0,
    coupling_stiffness=-2e4,
    coupling_damping=-1e1,
    num_threads=4,
    precision="single",
    save_data=False,
):
    # ==================FLOW SETUP START=========================
    z_range, y_range, x_range = domain_range
    grid_size_y = round(y_range / x_range * grid_size_x)
    grid_size_z = round(z_range / x_range * grid_size_x)
    # order Z, Y, X
    grid_size = (grid_size_z, grid_size_y, grid_size_x)
    print(f"Flow grid size:{grid_size}")
    dim = 3
    real_t = get_real_t(precision)
    kinematic_viscosity = (
        cilia_carpet_simulator.rod_base_length
        * cilia_carpet_simulator.velocity_scale
        / reynolds
    )
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=domain_x_range,
        kinematic_viscosity=kinematic_viscosity,
        flow_type="navier_stokes_with_forcing",
        real_t=real_t,
        num_threads=num_threads,
        navier_stokes_inertial_term_form="rotational",
        filter_vorticity=True,
        filter_setting_dict={"order": 1, "type": "multiplicative"},
    )
    # ==================FLOW SETUP END=========================

    # ==================FLOW-ROD COMMUNICATOR SETUP START======
    rod_flow_interactor_list = []
    for magnetic_rod in cilia_carpet_simulator.magnetic_rod_list:
        rod_flow_interactor = sps.CosseratRodFlowInteraction(
            cosserat_rod=magnetic_rod,
            eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
            eul_grid_velocity_field=flow_sim.velocity_field,
            virtual_boundary_stiffness_coeff=coupling_stiffness,
            virtual_boundary_damping_coeff=coupling_damping,
            dx=flow_sim.dx,
            grid_dim=dim,
            real_t=real_t,
            num_threads=num_threads,
            forcing_grid_cls=sps.CosseratRodElementCentricForcingGrid,
        )
        rod_flow_interactor_list.append(rod_flow_interactor)
        cilia_carpet_simulator.magnetic_beam_sim.add_forcing_to(magnetic_rod).using(
            sps.FlowForces,
            rod_flow_interactor,
        )
    # ==================FLOW-ROD COMMUNICATOR SETUP END======

    if save_data:
        # setup IO
        # TODO internalise this in flow simulator as dump_fields
        io_origin = np.array(
            [flow_sim.z_grid.min(), flow_sim.y_grid.min(), flow_sim.x_grid.min()]
        )
        io_dx = flow_sim.dx * np.ones(dim)
        io_grid_size = np.array(grid_size)
        io = IO(dim=dim, real_dtype=real_t)
        io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
        io.add_as_eulerian_fields_for_io(
            vorticity=flow_sim.vorticity_field, velocity=flow_sim.velocity_field
        )
        # Initialize carpet IO
        carpet_io = IO(dim=dim, real_dtype=real_t)
        rod_num_lag_nodes_list = [
            interactor.forcing_grid.num_lag_nodes
            for interactor in rod_flow_interactor_list
        ]
        carpet_num_lag_nodes = sum(rod_num_lag_nodes_list)
        carpet_lag_grid_position_field = np.zeros((dim, carpet_num_lag_nodes))
        carpet_lag_grid_forcing_field = np.zeros_like(carpet_lag_grid_position_field)

        def update_carpet_lag_grid_fields():
            """Updates the combined lag grid with individual rod grids"""
            start_idx = 0
            for interactor in rod_flow_interactor_list:
                carpet_lag_grid_position_field[
                    ..., start_idx : start_idx + interactor.forcing_grid.num_lag_nodes
                ] = interactor.forcing_grid.position_field
                carpet_lag_grid_forcing_field[
                    ..., start_idx : start_idx + interactor.forcing_grid.num_lag_nodes
                ] = interactor.lag_grid_forcing_field
                start_idx += interactor.forcing_grid.num_lag_nodes

        update_carpet_lag_grid_fields()
        # Add vector field on lagrangian grid
        carpet_io.add_as_lagrangian_fields_for_io(
            lagrangian_grid=carpet_lag_grid_position_field,
            lagrangian_grid_name="cilia",
            vector_3d=carpet_lag_grid_forcing_field,
        )
    cilia_carpet_simulator.finalize()
    # =================TIMESTEPPING====================
    time = 0.0
    foto_timer = 0.0
    foto_timer_limit = cilia_carpet_simulator.final_time / 100
    time_history = []

    # create fig for plotting flow fields
    fig, ax = sps.create_figure_and_axes()
    # iterate
    while time < cilia_carpet_simulator.final_time:
        # Save data
        if foto_timer > foto_timer_limit or foto_timer == 0:
            foto_timer = 0.0
            if save_data:
                update_carpet_lag_grid_fields()
                io.save(
                    h5_file_name="flow_" + str("%0.4d" % (time * 100)) + ".h5",
                    time=time,
                )
                carpet_io.save(
                    h5_file_name="carpet_" + str("%0.4d" % (time * 100)) + ".h5",
                    time=time,
                )
            ax.set_title(f"Velocity magnitude, time: {time:.2f}")
            contourf_obj = ax.contourf(
                flow_sim.x_grid[:, grid_size_y // 2, :],
                flow_sim.z_grid[:, grid_size_y // 2, :],
                np.linalg.norm(
                    np.mean(
                        flow_sim.velocity_field[
                            :, :, grid_size_y // 2 - 1 : grid_size_y // 2 + 1, :
                        ],
                        axis=2,
                    ),
                    axis=0,
                ),
                levels=50,
                extend="both",
                cmap="Purples",
            )
            cbar = fig.colorbar(mappable=contourf_obj, ax=ax)
            for magnetic_rod in cilia_carpet_simulator.magnetic_rod_list:
                ax.scatter(
                    magnetic_rod.position_collection[0],
                    magnetic_rod.position_collection[2],
                    s=5,
                    color="k",
                )
            sps.save_and_clear_fig(
                fig, ax, cbar, file_name="snap_" + str("%0.5d" % (time * 100)) + ".png"
            )
            time_history.append(time)
            print(
                f"time: {time:.2f} ({(time/cilia_carpet_simulator.final_time*100):2.1f}%), "
                f"max_vort: {np.amax(flow_sim.vorticity_field):.4f}, "
                f"div vorticity norm: {flow_sim.get_vorticity_divergence_l2_norm():.4f}"
            )

        # compute timestep
        flow_dt = flow_sim.compute_stable_timestep(dt_prefac=0.25)
        # flow_dt = rod_dt

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, cilia_carpet_simulator.dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = time
        for i in range(rod_time_steps):
            # timestep the cilia simulator
            rod_time = cilia_carpet_simulator.time_step(
                time=rod_time, time_step=local_rod_dt
            )
            # timestep the rod_flow_interactors
            for rod_flow_interactor in rod_flow_interactor_list:
                rod_flow_interactor.time_step(dt=local_rod_dt)

        # evaluate feedback/interaction between flow and rod
        for rod_flow_interactor in rod_flow_interactor_list:
            rod_flow_interactor()

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

    # setup the structure of the carpet
    num_rods_along_x = 9  # set >= 2
    num_rods_along_y = 4  # set >= 2
    n_elem_per_rod = 20
    rod_base_length = 1.5
    carpet_spacing = rod_base_length
    carpet_length_x = (num_rods_along_x - 1) * carpet_spacing
    carpet_length_y = (num_rods_along_y - 1) * carpet_spacing
    # get the flow domain range based on the carpet
    domain_x_range = carpet_length_x + 2 * rod_base_length
    domain_y_range = carpet_length_y + 2 * rod_base_length
    domain_z_range = 5 * rod_base_length
    carpet_base_centroid = np.array(
        [0.5 * domain_x_range, 0.5 * domain_y_range, 0.1 * domain_z_range]
    )
    cilia_carpet_simulator = MagneticCiliaCarpetSimulator(
        n_elem_per_rod=n_elem_per_rod,
        num_rods_along_x=num_rods_along_x,
        num_rods_along_y=num_rods_along_y,
        rod_base_length=rod_base_length,
        num_cycles=2.0,
        carpet_base_centroid=carpet_base_centroid,
        plot_result=False,
    )
    immersed_magnetic_cilia_carpet_case(
        cilia_carpet_simulator=cilia_carpet_simulator,
        reynolds=10.0,
        grid_size_x=128,
        domain_range=(domain_z_range, domain_y_range, domain_x_range),
    )

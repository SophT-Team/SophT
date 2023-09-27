import pytest
import sopht.utils as spu
import sopht.simulator as sps
import elastica as ea
import numpy as np
import os
from numpy.testing import assert_allclose


@pytest.mark.parametrize("grid_size_x", [64])
@pytest.mark.parametrize("precision", ["single"])
@pytest.mark.parametrize("with_free_stream", [True])
@pytest.mark.parametrize("filter_vorticity", [True])
def test_restart_simulation(precision, grid_size_x, with_free_stream, filter_vorticity):

    final_time = 0.25

    # run first half of the simulation
    _ = run_sim(
        final_time / 2,
        save_data=True,
        load_data=False,
        precision=precision,
        grid_size_x=grid_size_x,
        with_free_stream=with_free_stream,
        filter_vorticity=filter_vorticity,
    )

    # run second half of the simulation
    recorded_data_restarted = run_sim(
        final_time,
        save_data=False,
        load_data=True,
        precision=precision,
        grid_size_x=grid_size_x,
        with_free_stream=with_free_stream,
        filter_vorticity=filter_vorticity,
    )

    # run full simulation
    recorded_data_full = run_sim(
        final_time,
        save_data=False,
        load_data=False,
        precision=precision,
        grid_size_x=grid_size_x,
        with_free_stream=with_free_stream,
        filter_vorticity=filter_vorticity,
    )

    os.system("rm -f *h5 *xmf")

    for i in range(len(recorded_data_restarted)):
        assert_allclose(recorded_data_restarted[i], recorded_data_full[i])


def run_sim(
    final_time: float,
    save_data: bool,
    load_data: bool,
    precision: str,
    grid_size_x: int,
    with_free_stream: bool,
    filter_vorticity: bool,
):
    class RestartTestSimulator(
        ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.Damping
    ):
        ...

    restart_test_simulator = RestartTestSimulator()

    num_threads = 4
    grid_dim = 3
    x_range = 1.8
    nu = 3e-3
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x, int(0.25 * grid_size_x), grid_size_x)
    filter_type = "multiplicative"
    filter_order = 1

    flow_sim = sps.UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        with_forcing=True,
        with_free_stream_flow=with_free_stream,
        real_t=real_t,
        num_threads=num_threads,
        filter_vorticity=filter_vorticity,
        filter_setting_dict={"type": filter_type, "order": filter_order},
    )

    n_elems = 40
    cosserat_rod = ea.CosseratRod.straight_rod(
        n_elems,
        start=np.array([0.2 * x_range, 0.5 * x_range * 0.25, 0.75 * x_range]),
        direction=np.array([0.0, 0.0, -1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1,
        base_radius=0.045,
        density=830,
        youngs_modulus=865,
        shear_modulus=865 / 1.01,
    )

    restart_test_simulator.append(cosserat_rod)

    restart_test_simulator.constrain(cosserat_rod).using(
        ea.OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )
    # Add gravitational forces
    restart_test_simulator.add_forcing_to(cosserat_rod).using(
        ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -0.036])
    )
    # add damping
    dl = 1 / n_elems
    rod_dt = 0.01 * dl
    damping_constant = 1e-3
    restart_test_simulator.dampen(cosserat_rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=rod_dt,
    )

    forcing_grid_cls = sps.CosseratRodSurfaceForcingGrid
    rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=cosserat_rod,
        eul_grid_forcing_field=flow_sim.eul_grid_forcing_field,
        eul_grid_velocity_field=flow_sim.velocity_field,
        virtual_boundary_stiffness_coeff=-200000.0,
        virtual_boundary_damping_coeff=-100.0,
        dx=flow_sim.dx,
        grid_dim=grid_dim,
        real_t=real_t,
        forcing_grid_cls=forcing_grid_cls,
        surface_grid_density_for_largest_element=16,
    )

    restart_test_simulator.add_forcing_to(cosserat_rod).using(
        sps.FlowForces,
        rod_flow_interactor,
    )

    restart_test_simulator.finalize()

    restart_dir = "restart_data"
    free_stream_velocity = np.array([1.0, 0.0, 0.0])
    real_t = spu.get_real_t(precision)
    timestepper = ea.PositionVerlet()
    do_step, stages_and_updates = ea.extend_stepper_interface(
        timestepper, restart_test_simulator
    )

    recorded_data = []

    # setup flow IO
    io = spu.EulerianFieldIO(
        position_field=flow_sim.position_field,
        eulerian_fields_dict={
            "vorticity": flow_sim.vorticity_field,
            "velocity": flow_sim.velocity_field,
        },
    )
    # Initialize sphere IO
    rod_io = spu.CosseratRodIO(
        cosserat_rod=cosserat_rod, dim=grid_dim, real_dtype=real_t
    )
    # Initialize forcing io
    forcing_io = spu.IO(dim=grid_dim, real_dtype=real_t)
    # Add vector field on lagrangian grid
    forcing_io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=rod_flow_interactor.forcing_grid.position_field,
        lagrangian_grid_name="cosseratrod",
        vector_3d=rod_flow_interactor.lag_grid_forcing_field,
        position_mismatch=rod_flow_interactor.lag_grid_position_mismatch_field,
    )

    if load_data:
        curr_time = spu.restart_simulation(
            restart_simulator=restart_test_simulator,
            io=io,
            rod_io=rod_io,
            forcing_io=forcing_io,
            restart_dir=restart_dir,
        )
        flow_sim.time = curr_time

    while flow_sim.time < final_time:

        if save_data:
            io.save(
                h5_file_name="sopht_" + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                time=flow_sim.time,
            )
            rod_io.save(
                h5_file_name="rod_" + str("%0.4d" % (flow_sim.time * 100)) + ".h5",
                time=flow_sim.time,
            )
            forcing_io.save(
                h5_file_name="forcing_grid_"
                + str("%0.4d" % (flow_sim.time * 100))
                + ".h5",
                time=flow_sim.time,
            )
            ea.save_state(restart_test_simulator, restart_dir, flow_sim.time)

        # compute timestep
        rod_dt = 2.5e-4
        flow_dt = rod_dt

        # timestep the rod, through the flow timestep
        rod_time_steps = int(flow_dt / min(flow_dt, rod_dt))
        local_rod_dt = flow_dt / rod_time_steps
        rod_time = flow_sim.time
        for i in range(rod_time_steps):
            rod_time = do_step(
                timestepper,
                stages_and_updates,
                restart_test_simulator,
                rod_time,
                local_rod_dt,
            )
            # timestep the cosserat_rod_flow_interactor
            rod_flow_interactor.time_step(dt=local_rod_dt)
        # evaluate feedback/interaction between flow and rod
        rod_flow_interactor()

        flow_sim.time_step(
            dt=flow_dt,
            free_stream_velocity=free_stream_velocity,
        )

    recorded_data.append(flow_sim.velocity_field)
    recorded_data.append(flow_sim.vorticity_field)
    recorded_data.append(flow_sim.position_field)
    recorded_data.append(np.array([flow_sim.time]))
    recorded_data.append(rod_flow_interactor.lag_grid_forcing_field)
    recorded_data.append(rod_flow_interactor.forcing_grid.position_field)
    recorded_data.append(cosserat_rod.position_collection)
    recorded_data.append(cosserat_rod.velocity_collection)

    return recorded_data

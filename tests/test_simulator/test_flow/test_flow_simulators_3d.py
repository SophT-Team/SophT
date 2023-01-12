import numpy as np
import pytest
import sopht.utils as spu
import sopht.simulator as sps


@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("with_free_stream", [True, False])
@pytest.mark.parametrize("filter_vorticity", [True, False])
def test_flow_sim_3d_navier_stokes_with_forcing_timestep(
    grid_size_x,
    precision,
    with_free_stream,
    filter_vorticity,
):
    num_threads = 4
    grid_dim = 3
    x_range = 1.0
    nu = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dt = 2.0
    free_stream_velocity = np.array([3.0, 4.0, 5.0])
    init_time = 1.0
    filter_type = "convolution"
    filter_order = 2
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=with_free_stream,
        real_t=real_t,
        num_threads=num_threads,
        time=init_time,
        filter_vorticity=filter_vorticity,
        filter_setting_dict={"type": filter_type, "order": filter_order},
    )
    # initialise flow sim state (vorticity and forcing)
    flow_sim.vorticity_field[...] = np.random.rand(
        *flow_sim.vorticity_field.shape
    ).astype(real_t)
    flow_sim.velocity_field[...] = 0 * np.random.rand(
        *flow_sim.velocity_field.shape
    ).astype(real_t)
    flow_sim.eul_grid_forcing_field[...] = np.random.rand(
        *flow_sim.eul_grid_forcing_field.shape
    ).astype(real_t)
    # generate reference simulator
    ref_flow_sim = sps.UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        with_forcing=True,
        with_free_stream_flow=with_free_stream,
        real_t=real_t,
        num_threads=num_threads,
        time=init_time,
        filter_vorticity=filter_vorticity,
        filter_setting_dict={"type": filter_type, "order": filter_order},
    )
    ref_flow_sim.vorticity_field[...] = flow_sim.vorticity_field.copy()
    ref_flow_sim.velocity_field[...] = flow_sim.velocity_field.copy()
    ref_flow_sim.eul_grid_forcing_field[...] = flow_sim.eul_grid_forcing_field.copy()
    flow_sim.time_step(dt=dt, free_stream_velocity=free_stream_velocity)
    ref_flow_sim.time_step(dt=dt, free_stream_velocity=free_stream_velocity)

    assert flow_sim.time == ref_flow_sim.time
    np.testing.assert_allclose(flow_sim.eul_grid_forcing_field, 0.0)
    np.testing.assert_allclose(flow_sim.vorticity_field, ref_flow_sim.vorticity_field)
    np.testing.assert_allclose(flow_sim.velocity_field, ref_flow_sim.velocity_field)


@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_flow_sim_3d_compute_stable_timestep(grid_size_x, precision):
    num_threads = 4
    grid_dim = 3
    x_range = 1.0
    nu = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    cfl = 0.2
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        CFL=cfl,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_sim.velocity_field[...] = 2.0
    dt_prefac = 0.5
    sim_dt = flow_sim.compute_stable_timestep(dt_prefac=dt_prefac)
    # generate reference simulator
    ref_flow_sim = sps.UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        cfl=cfl,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
    )
    ref_flow_sim.velocity_field[...] = 2.0
    ref_dt = ref_flow_sim.compute_stable_timestep(dt_prefac=dt_prefac)
    assert ref_dt == sim_dt


@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_flow_sim_3d_get_vorticity_divergence_l2_norm(grid_size_x, precision):
    num_threads = 4
    grid_dim = 3
    x_range = 1.0
    nu = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    cfl = 0.2
    flow_sim = sps.UnboundedFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        CFL=cfl,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_sim.vorticity_field[...] = np.random.rand(grid_dim, *grid_size)
    vorticity_divergence_l2_norm = flow_sim.get_vorticity_divergence_l2_norm()
    # generate reference simulator
    ref_flow_sim = sps.UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        cfl=cfl,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
    )
    ref_flow_sim.vorticity_field[...] = flow_sim.vorticity_field.copy()
    ref_vorticity_divg_l2_norm = ref_flow_sim.get_vorticity_divergence_l2_norm()
    assert vorticity_divergence_l2_norm == ref_vorticity_divg_l2_norm

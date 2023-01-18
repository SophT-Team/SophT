import numpy as np
import pytest
import sopht.simulator as sps
from sopht.utils.precision import get_real_t


@pytest.mark.parametrize("grid_size_x", [8, 16])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("with_free_stream", [True, False])
def test_flow_sim_2d_navier_stokes_with_forcing_timestep(
    grid_size_x, precision, with_free_stream
):
    num_threads = 4
    grid_dim = 2
    x_range = 1.0
    nu = 1.0
    real_t = get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dt = 2.0
    free_stream_velocity = np.array([3.0, 4.0])
    init_time = 1.0
    flow_sim = sps.UnboundedFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        flow_type="navier_stokes_with_forcing",
        with_free_stream_flow=with_free_stream,
        real_t=real_t,
        num_threads=num_threads,
        time=init_time,
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
    ref_flow_sim = sps.UnboundedNavierStokesFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        with_forcing=True,
        with_free_stream_flow=with_free_stream,
        real_t=real_t,
        num_threads=num_threads,
        time=init_time,
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


@pytest.mark.parametrize("grid_size_x", [8, 16])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_flow_sim_2d_compute_stable_timestep(grid_size_x, precision):
    num_threads = 4
    grid_dim = 2
    x_range = 1.0
    nu = 1e-2
    real_t = get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    cfl = 0.2
    flow_sim = sps.UnboundedFlowSimulator2D(
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
    ref_flow_sim = sps.UnboundedNavierStokesFlowSimulator2D(
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

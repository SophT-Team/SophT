import numpy as np
import pytest
import sopht.utils as spu
import sopht.simulator as sps
import sopht.numeric.eulerian_grid_ops as spne


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("field_type", ["scalar", "vector"])
def test_passive_transport_flow_simulator_time_step(
    grid_dim,
    field_type,
    grid_size_x,
    precision,
):
    num_threads = 4
    x_range = 1.0
    nu = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    dt = 2.0
    init_time = 1.0
    # TODO support this option in future
    if grid_dim == 2 and field_type == "vector":
        return
    flow_sim = sps.PassiveTransportFlowSimulator(
        kinematic_viscosity=nu,
        grid_dim=grid_dim,
        grid_size=grid_size,
        x_range=x_range,
        real_t=real_t,
        num_threads=num_threads,
        time=init_time,
        field_type=field_type,
    )
    ref_time = init_time + dt
    # initialise flow sim state (passive fields)
    flow_sim.velocity_field[...] = np.random.rand(
        *flow_sim.velocity_field.shape
    ).astype(real_t)
    flow_sim.primary_field[...] = np.random.rand(*flow_sim.primary_field.shape).astype(
        real_t
    )
    ref_primary_field = flow_sim.primary_field.copy()
    flow_sim.time_step(dt=dt)

    # next we setup the reference timestep manually
    # first we compile necessary kernels
    advection_timestep = None
    diffusion_timestep = None
    match grid_dim:
        case 2:
            diffusion_timestep = (
                spne.gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
                    real_t=real_t,
                    fixed_grid_size=grid_size,
                    num_threads=num_threads,
                )
            )
            advection_timestep = spne.gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
                real_t=real_t,
                fixed_grid_size=grid_size,
                num_threads=num_threads,
            )
        case 3:
            diffusion_timestep = (
                spne.gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
                    real_t=real_t,
                    fixed_grid_size=grid_size,
                    num_threads=num_threads,
                    field_type=field_type,
                )
            )
            advection_timestep = spne.gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                real_t=real_t,
                fixed_grid_size=grid_size,
                num_threads=num_threads,
                field_type=field_type,
            )
    # manually timestep
    advection_timestep(
        ref_primary_field,
        advection_flux=np.zeros_like(flow_sim.buffer_scalar_field),
        velocity=flow_sim.velocity_field,
        dt_by_dx=real_t(dt / dx),
    )
    diffusion_timestep(
        ref_primary_field,
        diffusion_flux=np.zeros_like(flow_sim.buffer_scalar_field),
        nu_dt_by_dx2=real_t(nu * dt / dx / dx),
    )
    assert flow_sim.time == ref_time
    np.testing.assert_allclose(flow_sim.primary_field, ref_primary_field)


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_passive_transport_flow_sim_compute_stable_time_step(
    grid_dim, grid_size_x, precision
):
    num_threads = 4
    x_range = 1.0
    nu = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    cfl = 0.2
    flow_sim = sps.PassiveTransportFlowSimulator(
        kinematic_viscosity=nu,
        cfl=cfl,
        grid_dim=grid_dim,
        grid_size=grid_size,
        x_range=x_range,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_sim.velocity_field[...] = 2.0
    dt_prefac = 0.5
    sim_dt = flow_sim.compute_stable_time_step(dt_prefac=dt_prefac)
    # next compute reference value
    tol = 10 * np.finfo(real_t).eps
    advection_limit_dt = (
        cfl * dx / (grid_dim * 2.0 + tol)
    )  # max(sum(abs(velocity_field)))
    diffusion_limit_dt = 0.9 * dx**2 / (2 * grid_dim) / nu
    ref_dt = dt_prefac * min(advection_limit_dt, diffusion_limit_dt)
    assert ref_dt == sim_dt

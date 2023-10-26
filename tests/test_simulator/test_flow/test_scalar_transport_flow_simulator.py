import numpy as np
import pytest
import sopht.utils as spu
import sopht.simulator as sps
import sopht.numeric.eulerian_grid_ops as spne


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("field_type", ["scalar"])
def test_passive_transport_scalar_field_flow_simulator_time_step(
    grid_dim,
    field_type,
    grid_size_x,
    precision,
):
    num_threads = 4
    x_range = 1.0
    alpha = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    dt = 2.0
    init_time = 1.0
    penalty_zone_width = 2

    velocity_field = np.zeros((grid_dim, *grid_size), dtype=real_t)

    flow_sim = sps.PassiveTransportScalarFieldFlowSimulator(
        diffusivity_constant=alpha,
        grid_dim=grid_dim,
        grid_size=grid_size,
        x_range=x_range,
        real_t=real_t,
        num_threads=num_threads,
        time=init_time,
        field_type=field_type,
        velocity_field=velocity_field,
        with_forcing=False,
        penalty_zone_width=penalty_zone_width,
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
            _penalise_field_towards_boundary = (
                spne.gen_penalise_field_boundary_pyst_kernel_2d(
                    width=penalty_zone_width,
                    dx=dx,
                    x_grid_field=flow_sim.position_field[spu.VectorField.x_axis_idx()],
                    y_grid_field=flow_sim.position_field[spu.VectorField.y_axis_idx()],
                    real_t=real_t,
                    num_threads=num_threads,
                    fixed_grid_size=grid_size,
                )
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
            _penalise_field_towards_boundary = (
                spne.gen_penalise_field_boundary_pyst_kernel_3d(
                    width=penalty_zone_width,
                    dx=dx,
                    x_grid_field=flow_sim.position_field[spu.VectorField.x_axis_idx()],
                    y_grid_field=flow_sim.position_field[spu.VectorField.y_axis_idx()],
                    z_grid_field=flow_sim.position_field[spu.VectorField.z_axis_idx()],
                    real_t=real_t,
                    num_threads=num_threads,
                    fixed_grid_size=grid_size,
                    field_type=field_type,
                )
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
        nu_dt_by_dx2=real_t(alpha * dt / dx / dx),
    )
    _penalise_field_towards_boundary(ref_primary_field)
    assert flow_sim.time == ref_time
    np.testing.assert_allclose(flow_sim.primary_field, ref_primary_field)


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("field_type", ["scalar"])
def test_passive_transport_scalar_field_flow_simulator_with_forcing_time_step(
    grid_dim,
    field_type,
    grid_size_x,
    precision,
):
    num_threads = 4
    x_range = 1.0
    alpha = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    dt = 2.0
    init_time = 1.0
    penalty_zone_width = 2

    velocity_field = np.zeros((grid_dim, *grid_size), dtype=real_t)

    flow_sim = sps.PassiveTransportScalarFieldFlowSimulator(
        diffusivity_constant=alpha,
        grid_dim=grid_dim,
        grid_size=grid_size,
        x_range=x_range,
        real_t=real_t,
        num_threads=num_threads,
        time=init_time,
        field_type=field_type,
        velocity_field=velocity_field,
        with_forcing=True,
        penalty_zone_width=penalty_zone_width,
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
    flow_sim.eul_grid_forcing_field[...] = np.random.rand(*flow_sim.primary_field.shape)
    ref_eul_grid_forcing_field = flow_sim.eul_grid_forcing_field.copy()
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
            _penalise_field_towards_boundary = (
                spne.gen_penalise_field_boundary_pyst_kernel_2d(
                    width=penalty_zone_width,
                    dx=dx,
                    x_grid_field=flow_sim.position_field[spu.VectorField.x_axis_idx()],
                    y_grid_field=flow_sim.position_field[spu.VectorField.y_axis_idx()],
                    real_t=real_t,
                    num_threads=num_threads,
                    fixed_grid_size=grid_size,
                )
            )
            _update_passive_field_from_forcing = (
                spne.gen_update_passive_field_from_forcing_pyst_kernel_2d(
                    real_t=real_t,
                    fixed_grid_size=grid_size,
                    num_threads=num_threads,
                )
            )
            _set_field = spne.gen_set_fixed_val_pyst_kernel_2d(
                real_t=real_t,
                num_threads=num_threads,
                field_type=field_type,
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
            _penalise_field_towards_boundary = (
                spne.gen_penalise_field_boundary_pyst_kernel_3d(
                    width=penalty_zone_width,
                    dx=dx,
                    x_grid_field=flow_sim.position_field[spu.VectorField.x_axis_idx()],
                    y_grid_field=flow_sim.position_field[spu.VectorField.y_axis_idx()],
                    z_grid_field=flow_sim.position_field[spu.VectorField.z_axis_idx()],
                    real_t=real_t,
                    num_threads=num_threads,
                    fixed_grid_size=grid_size,
                    field_type=field_type,
                )
            )
            _update_passive_field_from_forcing = (
                spne.gen_update_passive_field_from_forcing_pyst_kernel_3d(
                    real_t=real_t,
                    fixed_grid_size=grid_size,
                    num_threads=num_threads,
                )
            )
            _set_field = spne.gen_set_fixed_val_pyst_kernel_3d(
                real_t=real_t,
                num_threads=num_threads,
                field_type=field_type,
            )
    # manually timestep
    _update_passive_field_from_forcing(
        passive_field=ref_primary_field,
        forcing_field=ref_eul_grid_forcing_field,
        prefactor=real_t(dt),
    )
    advection_timestep(
        ref_primary_field,
        advection_flux=np.zeros_like(flow_sim.buffer_scalar_field),
        velocity=flow_sim.velocity_field,
        dt_by_dx=real_t(dt / dx),
    )
    diffusion_timestep(
        ref_primary_field,
        diffusion_flux=np.zeros_like(flow_sim.buffer_scalar_field),
        nu_dt_by_dx2=real_t(alpha * dt / dx / dx),
    )
    _penalise_field_towards_boundary(ref_primary_field)
    _set_field(ref_eul_grid_forcing_field, 0.0)
    assert flow_sim.time == ref_time
    np.testing.assert_allclose(flow_sim.primary_field, ref_primary_field)
    np.testing.assert_allclose(
        flow_sim.eul_grid_forcing_field, ref_eul_grid_forcing_field
    )


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_passive_transport_scalar_field_flow_sim_compute_stable_timestep(
    grid_dim, grid_size_x, precision
):
    num_threads = 4
    x_range = 1.0
    alpha = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    cfl = 0.2

    velocity_field = np.zeros((grid_dim, *grid_size), dtype=real_t)
    flow_sim = sps.PassiveTransportScalarFieldFlowSimulator(
        diffusivity_constant=alpha,
        cfl=cfl,
        grid_dim=grid_dim,
        grid_size=grid_size,
        x_range=x_range,
        real_t=real_t,
        num_threads=num_threads,
        velocity_field=velocity_field,
    )
    flow_sim.velocity_field[...] = 2.0
    dt_prefac = 0.5
    sim_dt = flow_sim.compute_stable_timestep(dt_prefac=dt_prefac)
    # next compute reference value
    tol = 10 * np.finfo(real_t).eps
    advection_limit_dt = (
        cfl * dx / (grid_dim * 2.0 + tol)
    )  # max(sum(abs(velocity_field)))
    diffusion_limit_dt = 0.9 * dx**2 / (2 * grid_dim) / alpha
    ref_dt = dt_prefac * min(advection_limit_dt, diffusion_limit_dt)
    assert ref_dt == sim_dt

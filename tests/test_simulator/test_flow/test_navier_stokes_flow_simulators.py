import numpy as np
import pytest
import sopht.simulator as sps
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu


@pytest.mark.parametrize("grid_size_x", [8, 16])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("with_free_stream", [True, False])
def test_navier_stokes_flow_sim_2d_with_forcing_timestep(
    grid_size_x, precision, with_free_stream
):
    num_threads = 4
    grid_dim = 2
    x_range = 1.0
    nu = 1.0
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    dt = 2.0
    free_stream_velocity = np.array([3.0, 4.0])
    init_time = 1.0
    flow_sim = sps.UnboundedNavierStokesFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
        time=init_time,
        with_forcing=True,
        with_free_stream_flow=with_free_stream,
    )
    ref_time = init_time + dt
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
    ref_vorticity_field = flow_sim.vorticity_field.copy()
    ref_velocity_field = flow_sim.velocity_field.copy()
    ref_eul_grid_forcing_field = flow_sim.eul_grid_forcing_field.copy()
    flow_sim.time_step(dt=dt, free_stream_velocity=free_stream_velocity)

    # next we setup the reference timestep manually
    # first we compile necessary kernels
    diffusion_timestep = spne.gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=grid_size,
        num_threads=num_threads,
    )
    advection_timestep = (
        spne.gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=grid_size,
            num_threads=num_threads,
        )
    )
    grid_size_y, grid_size_x = grid_size
    unbounded_poisson_solver = spne.UnboundedPoissonSolverPYFFTW2D(
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        x_range=x_range,
        real_t=real_t,
        num_threads=num_threads,
    )
    curl = spne.gen_outplane_field_curl_pyst_kernel_2d(
        real_t=real_t,
        num_threads=num_threads,
        fixed_grid_size=grid_size,
    )
    penalise_field_towards_boundary = spne.gen_penalise_field_boundary_pyst_kernel_2d(
        width=flow_sim.penalty_zone_width,
        dx=dx,
        x_grid_field=flow_sim.position_field[spu.VectorField.x_axis_idx()],
        y_grid_field=flow_sim.position_field[spu.VectorField.y_axis_idx()],
        real_t=real_t,
        num_threads=num_threads,
        fixed_grid_size=grid_size,
    )
    update_vorticity_from_velocity_forcing = (
        spne.gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=grid_size,
            num_threads=num_threads,
        )
    )

    # manually timestep
    update_vorticity_from_velocity_forcing(
        vorticity_field=ref_vorticity_field,
        velocity_forcing_field=ref_eul_grid_forcing_field,
        prefactor=real_t(dt / (2 * dx)),
    )
    flux_buffer = np.zeros(grid_size, dtype=real_t)
    advection_timestep(
        field=ref_vorticity_field,
        advection_flux=flux_buffer,
        velocity=ref_velocity_field,
        dt_by_dx=real_t(dt / dx),
    )
    diffusion_timestep(
        field=ref_vorticity_field,
        diffusion_flux=flux_buffer,
        nu_dt_by_dx2=real_t(nu * dt / dx / dx),
    )
    penalise_field_towards_boundary(field=ref_vorticity_field)
    stream_func_field = np.zeros(grid_size, dtype=real_t)
    unbounded_poisson_solver.solve(
        solution_field=stream_func_field,
        rhs_field=ref_vorticity_field,
    )
    curl(
        curl=ref_velocity_field,
        field=stream_func_field,
        prefactor=real_t(0.5 / dx),
    )
    if with_free_stream:
        ref_velocity_field[...] += free_stream_velocity.reshape((grid_dim, 1, 1))

    assert flow_sim.time == ref_time
    np.testing.assert_allclose(flow_sim.eul_grid_forcing_field, 0.0)
    np.testing.assert_allclose(flow_sim.vorticity_field, ref_vorticity_field)
    np.testing.assert_allclose(flow_sim.velocity_field, ref_velocity_field)


@pytest.mark.parametrize("grid_size_x", [8, 16])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_navier_stokes_flow_sim_2d_compute_stable_timestep(grid_size_x, precision):
    num_threads = 4
    grid_dim = 2
    x_range = 1.0
    nu = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    cfl = 0.2
    flow_sim = sps.UnboundedNavierStokesFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        cfl=cfl,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_sim.velocity_field[...] = 2.0
    dt_prefac = 0.5
    sim_dt = flow_sim.compute_stable_timestep(dt_prefac=dt_prefac)
    # next compute reference value
    tol = 10 * np.finfo(real_t).eps
    advection_limit_dt = (
        cfl * dx / (grid_dim * 2.0 + tol)
    )  # max(sum(abs(velocity_field)))
    diffusion_limit_dt = 0.9 * dx**2 / (2 * grid_dim) / nu
    ref_dt = dt_prefac * min(advection_limit_dt, diffusion_limit_dt)
    assert ref_dt == sim_dt


@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("with_free_stream", [True, False])
@pytest.mark.parametrize("filter_vorticity", [True, False])
def test_navier_stokes_flow_sim_3d_with_forcing_timestep(
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
    dx = real_t(x_range / grid_size_x)
    dt = 2.0
    free_stream_velocity = np.array([3.0, 4.0, 5.0])
    init_time = 1.0
    filter_type = "convolution"
    filter_order = 2
    flow_sim = sps.UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
        time=init_time,
        with_forcing=True,
        with_free_stream_flow=with_free_stream,
        filter_vorticity=filter_vorticity,
        filter_setting_dict={"type": filter_type, "order": filter_order},
    )
    ref_time = init_time + dt
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
    ref_vorticity_field = flow_sim.vorticity_field.copy()
    ref_velocity_field = flow_sim.velocity_field.copy()
    ref_eul_grid_forcing_field = flow_sim.eul_grid_forcing_field.copy()
    flow_sim.time_step(dt=dt, free_stream_velocity=free_stream_velocity)
    # next we setup the reference timestep manually
    # first we compile necessary kernels
    diffusion_timestep = spne.gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=grid_size,
        num_threads=num_threads,
        field_type="vector",
    )
    elementwise_cross_product = spne.gen_elementwise_cross_product_pyst_kernel_3d(
        real_t=real_t,
        num_threads=num_threads,
        fixed_grid_size=grid_size,
    )
    grid_size_z, grid_size_y, grid_size_x = grid_size
    unbounded_poisson_solver = spne.UnboundedPoissonSolverPYFFTW3D(
        grid_size_z=grid_size_z,
        grid_size_y=grid_size_y,
        grid_size_x=grid_size_x,
        x_range=x_range,
        real_t=real_t,
        num_threads=num_threads,
    )
    curl = spne.gen_curl_pyst_kernel_3d(
        real_t=real_t,
        num_threads=num_threads,
        fixed_grid_size=grid_size,
    )
    penalise_field_towards_boundary = spne.gen_penalise_field_boundary_pyst_kernel_3d(
        width=flow_sim.penalty_zone_width,
        dx=dx,
        x_grid_field=flow_sim.position_field[spu.VectorField.x_axis_idx()],
        y_grid_field=flow_sim.position_field[spu.VectorField.y_axis_idx()],
        z_grid_field=flow_sim.position_field[spu.VectorField.z_axis_idx()],
        real_t=real_t,
        num_threads=num_threads,
        fixed_grid_size=grid_size,
        field_type="vector",
    )
    update_vorticity_from_velocity_forcing = (
        spne.gen_update_vorticity_from_velocity_forcing_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=grid_size,
            num_threads=num_threads,
        )
    )
    if filter_vorticity:
        filter_vector_field = spne.gen_laplacian_filter_kernel_3d(
            filter_flux_buffer=np.zeros_like(ref_vorticity_field[0]),
            field_buffer=np.zeros_like(ref_vorticity_field[0]),
            real_t=real_t,
            num_threads=num_threads,
            fixed_grid_size=grid_size,
            field_type="vector",
            filter_order=filter_order,
            filter_type=filter_type,
        )
    else:

        def filter_vector_field(vector_field):
            ...

    # manually timestep
    update_vorticity_from_velocity_forcing(
        vorticity_field=ref_vorticity_field,
        velocity_forcing_field=ref_eul_grid_forcing_field,
        prefactor=real_t(dt / (2 * dx)),
    )
    velocity_cross_vorticity = np.zeros_like(ref_vorticity_field)
    elementwise_cross_product(
        result_field=velocity_cross_vorticity,
        field_1=ref_velocity_field,
        field_2=ref_vorticity_field,
    )
    update_vorticity_from_velocity_forcing(
        vorticity_field=ref_vorticity_field,
        velocity_forcing_field=velocity_cross_vorticity,
        prefactor=real_t(dt / (2 * dx)),
    )
    flux_buffer = np.zeros(grid_size, dtype=real_t)
    diffusion_timestep(
        vector_field=ref_vorticity_field,
        diffusion_flux=flux_buffer,
        nu_dt_by_dx2=real_t(nu * dt / dx / dx),
    )
    filter_vector_field(vector_field=ref_vorticity_field)
    penalise_field_towards_boundary(vector_field=ref_vorticity_field)
    stream_func_field = np.zeros_like(ref_vorticity_field)
    unbounded_poisson_solver.vector_field_solve(
        solution_vector_field=stream_func_field,
        rhs_vector_field=ref_vorticity_field,
    )
    curl(
        curl=ref_velocity_field,
        field=stream_func_field,
        prefactor=real_t(0.5 / dx),
    )
    if with_free_stream:
        ref_velocity_field[...] += free_stream_velocity.reshape((grid_dim, 1, 1, 1))

    assert flow_sim.time == ref_time
    np.testing.assert_allclose(flow_sim.eul_grid_forcing_field, 0.0)
    np.testing.assert_allclose(flow_sim.vorticity_field, ref_vorticity_field)
    np.testing.assert_allclose(flow_sim.velocity_field, ref_velocity_field)


@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_navier_stokes_flow_sim_3d_compute_stable_timestep(grid_size_x, precision):
    num_threads = 4
    grid_dim = 3
    x_range = 1.0
    nu = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    cfl = 0.2
    flow_sim = sps.UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
        cfl=cfl,
    )
    flow_sim.velocity_field[...] = 2.0
    dt_prefac = 0.5
    sim_dt = flow_sim.compute_stable_timestep(dt_prefac=dt_prefac)
    # next compute reference value
    tol = 10 * np.finfo(real_t).eps
    advection_limit_dt = (
        cfl * dx / (grid_dim * 2.0 + tol)
    )  # max(sum(abs(velocity_field)))
    diffusion_limit_dt = 0.9 * dx**2 / (2 * grid_dim) / nu
    ref_dt = dt_prefac * min(advection_limit_dt, diffusion_limit_dt)
    assert ref_dt == sim_dt


@pytest.mark.parametrize("grid_size_x", [4, 8])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_navier_stokes_flow_sim_3d_get_vorticity_divergence_l2_norm(
    grid_size_x, precision
):
    num_threads = 4
    grid_dim = 3
    x_range = 1.0
    nu = 1e-2
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    dx = real_t(x_range / grid_size_x)
    flow_sim = sps.UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=nu,
        real_t=real_t,
        num_threads=num_threads,
    )
    flow_sim.vorticity_field[...] = np.random.rand(grid_dim, *grid_size)
    vorticity_divergence_l2_norm = flow_sim.get_vorticity_divergence_l2_norm()

    # compute manually
    compute_divergence = spne.gen_divergence_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=grid_size,
        num_threads=num_threads,
    )
    divergence_field = np.zeros_like(flow_sim.vorticity_field[0])
    compute_divergence(
        divergence=divergence_field,
        field=flow_sim.vorticity_field,
        inv_dx=(1.0 / dx),
    )
    ref_vorticity_divg_l2_norm = np.linalg.norm(divergence_field) * dx ** (
        grid_dim / 2.0
    )
    assert vorticity_divergence_l2_norm == ref_vorticity_divg_l2_norm

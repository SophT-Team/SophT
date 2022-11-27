import numpy as np

import psutil

import pytest

from sopht.numeric.immersed_boundary_ops import VirtualBoundaryForcing
from sopht.utils.precision import get_real_t, get_test_tol


class MockVirtualBoundaryForcingSolution:
    """Mock solution test class for virtual boundary forcing."""

    def __init__(self, grid_size, grid_dim=2, precision="single"):
        """Class initialiser."""
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.real_t = real_t
        self.virtual_boundary_stiffness_coeff = real_t(1e3)
        self.virtual_boundary_damping_coeff = real_t(1e1)
        self.dx = real_t(1.0 / grid_size)
        self.eul_grid_coord_shift = real_t(self.dx / 2)
        self.num_lag_nodes = 3
        self.interp_kernel_width = 2
        self.grid_dim = grid_dim
        self.grid_size = grid_size
        self.time = 0.0

    def compute_interaction_step_solution(
        self,
    ):
        """Compute solution fields necessary for interaction step."""
        eul_grid_velocity_shape = (self.grid_dim,) + (self.grid_size,) * self.grid_dim
        self.eul_grid_velocity_field = self.real_t(5.0) * np.ones(
            eul_grid_velocity_shape, dtype=self.real_t
        )
        self.lag_grid_velocity_field = self.real_t(2.0) * np.ones(
            (self.grid_dim, self.num_lag_nodes), dtype=self.real_t
        )
        self.nearest_eul_grid_index_to_lag_grid = np.empty(
            (self.grid_dim, self.num_lag_nodes), dtype=int
        )
        # init lag. grid near domain center
        self.nearest_eul_grid_index_to_lag_grid[...] = np.arange(
            self.grid_size // 3, self.grid_size // 3 + self.num_lag_nodes
        ).reshape(-1, self.num_lag_nodes)
        self.lag_grid_position_field = (
            self.nearest_eul_grid_index_to_lag_grid * self.dx
            + self.eul_grid_coord_shift
        )

        # interaction step solution computation
        self.lag_grid_flow_velocity_field = self.real_t(5.0) * np.ones(
            (self.grid_dim, self.num_lag_nodes), dtype=self.real_t
        )
        self.lag_grid_velocity_mismatch_field = (
            self.lag_grid_flow_velocity_field - self.lag_grid_velocity_field
        )
        self.dt = self.real_t(0.5) * self.dx
        self.lag_grid_position_mismatch_field = np.zeros_like(
            self.lag_grid_velocity_mismatch_field
        )
        self.lag_grid_forcing_field = (
            self.virtual_boundary_stiffness_coeff
            * self.lag_grid_position_mismatch_field
            + self.virtual_boundary_damping_coeff
            * self.lag_grid_velocity_mismatch_field
        )
        # max number of Eulerian grid indices affected by Lagrangian grid
        self.max_num_of_eul_grid_idx_impacted_by_lag_grid = (
            self.grid_dim
            * self.num_lag_nodes
            * (2 * self.interp_kernel_width) ** self.grid_dim
        )

    def check_lag_grid_interaction_solution(self, virtual_boundary_forcing):
        """Check solution for lag grid forcing in the interaction step."""
        np.testing.assert_allclose(
            virtual_boundary_forcing.nearest_eul_grid_index_to_lag_grid,
            self.nearest_eul_grid_index_to_lag_grid,
            atol=self.test_tol,
        )
        np.testing.assert_allclose(
            virtual_boundary_forcing.lag_grid_flow_velocity_field,
            self.lag_grid_flow_velocity_field,
            atol=self.test_tol,
        )
        np.testing.assert_allclose(
            virtual_boundary_forcing.lag_grid_velocity_mismatch_field,
            self.lag_grid_velocity_mismatch_field,
            atol=self.test_tol,
        )
        np.testing.assert_allclose(
            virtual_boundary_forcing.lag_grid_forcing_field,
            self.lag_grid_forcing_field,
            atol=self.test_tol,
        )

    def check_eul_grid_interaction_solution(self, eul_grid_forcing_field):
        """Check solution for eul grid forcing in the interaction step."""
        # check if force is added on Eulerian grid within max impact zone
        assert self.max_num_of_eul_grid_idx_impacted_by_lag_grid >= np.sum(
            eul_grid_forcing_field > 0
        )
        # force conservation test
        eul_forcing_grid_integral = np.sum(eul_grid_forcing_field) * (
            self.dx**self.grid_dim
        )
        np.testing.assert_allclose(
            eul_forcing_grid_integral,
            np.sum(self.lag_grid_forcing_field),
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_virtual_boundary_forcing_init(grid_dim, n_values, precision):
    mock_soln = MockVirtualBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=mock_soln.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=mock_soln.virtual_boundary_damping_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        num_lag_nodes=mock_soln.num_lag_nodes,
        real_t=mock_soln.real_t,
        start_time=mock_soln.time,
    )

    assert virtual_boundary_forcing.time == mock_soln.time
    assert (
        virtual_boundary_forcing.virtual_boundary_stiffness_coeff
        == mock_soln.virtual_boundary_stiffness_coeff
    )
    assert (
        virtual_boundary_forcing.virtual_boundary_damping_coeff
        == mock_soln.virtual_boundary_damping_coeff
    )
    assert virtual_boundary_forcing.nearest_eul_grid_index_to_lag_grid.dtype == int
    assert virtual_boundary_forcing.nearest_eul_grid_index_to_lag_grid.shape == (
        grid_dim,
        mock_soln.num_lag_nodes,
    )
    # figure out a clean way for this!
    if grid_dim == 2:
        assert virtual_boundary_forcing.local_eul_grid_support_of_lag_grid.shape == (
            grid_dim,
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mock_soln.num_lag_nodes,
        )
        assert virtual_boundary_forcing.interp_weights.shape == (
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mock_soln.num_lag_nodes,
        )
    elif grid_dim == 3:
        assert virtual_boundary_forcing.local_eul_grid_support_of_lag_grid.shape == (
            grid_dim,
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mock_soln.num_lag_nodes,
        )
        assert virtual_boundary_forcing.interp_weights.shape == (
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mock_soln.num_lag_nodes,
        )
    assert virtual_boundary_forcing.lag_grid_flow_velocity_field.shape == (
        grid_dim,
        mock_soln.num_lag_nodes,
    )
    assert virtual_boundary_forcing.lag_grid_velocity_mismatch_field.shape == (
        grid_dim,
        mock_soln.num_lag_nodes,
    )
    assert virtual_boundary_forcing.lag_grid_position_mismatch_field.shape == (
        grid_dim,
        mock_soln.num_lag_nodes,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_compute_lag_grid_velocity_mismatch_field(grid_dim, n_values, precision):
    real_t = get_real_t(precision)
    mock_soln = MockVirtualBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=mock_soln.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=mock_soln.virtual_boundary_damping_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        num_lag_nodes=mock_soln.num_lag_nodes,
        real_t=mock_soln.real_t,
    )

    lag_grid_velocity_field = np.ones((grid_dim, mock_soln.num_lag_nodes), dtype=real_t)
    virtual_boundary_forcing.lag_grid_flow_velocity_field = (
        real_t(3) * lag_grid_velocity_field
    )
    virtual_boundary_forcing.compute_lag_grid_velocity_mismatch_field(
        lag_grid_velocity_mismatch_field=virtual_boundary_forcing.lag_grid_velocity_mismatch_field,
        lag_grid_flow_velocity_field=virtual_boundary_forcing.lag_grid_flow_velocity_field,
        lag_grid_body_velocity_field=lag_grid_velocity_field,
    )
    # 3 - 1 = 2
    ref_lag_grid_velocity_mismatch_field = 2 * lag_grid_velocity_field
    np.testing.assert_allclose(
        ref_lag_grid_velocity_mismatch_field,
        virtual_boundary_forcing.lag_grid_velocity_mismatch_field,
        atol=mock_soln.test_tol,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_update_lag_grid_position_mismatch_field_via_euler_forward(
    grid_dim, n_values, precision
):
    real_t = get_real_t(precision)
    mock_soln = MockVirtualBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=mock_soln.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=mock_soln.virtual_boundary_damping_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        num_lag_nodes=mock_soln.num_lag_nodes,
        real_t=mock_soln.real_t,
    )

    virtual_boundary_forcing.lag_grid_position_mismatch_field[...] = real_t(1.0)
    dt = real_t(0.1)
    virtual_boundary_forcing.lag_grid_velocity_mismatch_field[...] = real_t(2.0)
    virtual_boundary_forcing.update_lag_grid_position_mismatch_field_via_euler_forward(
        lag_grid_position_mismatch_field=virtual_boundary_forcing.lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field=virtual_boundary_forcing.lag_grid_velocity_mismatch_field,
        dt=dt,
    )
    # pos = pos + dt * vel = 1.0 + 0.1 * 2.0 = 1.2
    ref_lag_grid_position_mismatch_field = real_t(1.2) * np.ones_like(
        virtual_boundary_forcing.lag_grid_position_mismatch_field
    )
    np.testing.assert_allclose(
        ref_lag_grid_position_mismatch_field,
        virtual_boundary_forcing.lag_grid_position_mismatch_field,
        atol=mock_soln.test_tol,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_compute_lag_grid_forcing_field(grid_dim, n_values, precision):
    real_t = get_real_t(precision)
    mock_soln = MockVirtualBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=mock_soln.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=mock_soln.virtual_boundary_damping_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        num_lag_nodes=mock_soln.num_lag_nodes,
        real_t=mock_soln.real_t,
    )

    virtual_boundary_forcing.lag_grid_position_mismatch_field[...] = real_t(1.0)
    virtual_boundary_forcing.lag_grid_velocity_mismatch_field[...] = real_t(2.0)
    virtual_boundary_forcing.compute_lag_grid_forcing_field(
        lag_grid_forcing_field=virtual_boundary_forcing.lag_grid_forcing_field,
        lag_grid_position_mismatch_field=virtual_boundary_forcing.lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field=virtual_boundary_forcing.lag_grid_velocity_mismatch_field,
        virtual_boundary_stiffness_coeff=virtual_boundary_forcing.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=virtual_boundary_forcing.virtual_boundary_damping_coeff,
    )
    ref_lag_grid_forcing_field = (
        mock_soln.virtual_boundary_stiffness_coeff
        + real_t(2.0) * mock_soln.virtual_boundary_damping_coeff
    ) * np.ones_like(virtual_boundary_forcing.lag_grid_position_mismatch_field)
    np.testing.assert_allclose(
        ref_lag_grid_forcing_field,
        virtual_boundary_forcing.lag_grid_forcing_field,
        atol=mock_soln.test_tol,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_compute_interaction_force_on_lag_grid(grid_dim, n_values, precision):
    mock_soln = MockVirtualBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    mock_soln.compute_interaction_step_solution()
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=mock_soln.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=mock_soln.virtual_boundary_damping_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        num_lag_nodes=mock_soln.num_lag_nodes,
        real_t=mock_soln.real_t,
        enable_eul_grid_forcing_reset=False,
    )
    virtual_boundary_forcing.compute_interaction_force_on_lag_grid(
        eul_grid_velocity_field=mock_soln.eul_grid_velocity_field,
        lag_grid_position_field=mock_soln.lag_grid_position_field,
        lag_grid_velocity_field=mock_soln.lag_grid_velocity_field,
    )
    mock_soln.check_lag_grid_interaction_solution(
        virtual_boundary_forcing=virtual_boundary_forcing
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_compute_interaction_without_eul_grid_forcing_reset(
    grid_dim, n_values, precision
):
    real_t = get_real_t(precision)
    mock_soln = MockVirtualBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    mock_soln.compute_interaction_step_solution()
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=mock_soln.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=mock_soln.virtual_boundary_damping_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        num_lag_nodes=mock_soln.num_lag_nodes,
        real_t=mock_soln.real_t,
        enable_eul_grid_forcing_reset=False,
    )
    eul_grid_velocity_shape = (grid_dim,) + (n_values,) * grid_dim
    eul_grid_forcing_field = np.zeros(eul_grid_velocity_shape, dtype=real_t)
    virtual_boundary_forcing.compute_interaction_forcing(
        eul_grid_forcing_field=eul_grid_forcing_field,
        eul_grid_velocity_field=mock_soln.eul_grid_velocity_field,
        lag_grid_position_field=mock_soln.lag_grid_position_field,
        lag_grid_velocity_field=mock_soln.lag_grid_velocity_field,
    )
    mock_soln.check_lag_grid_interaction_solution(
        virtual_boundary_forcing=virtual_boundary_forcing
    )
    mock_soln.check_eul_grid_interaction_solution(
        eul_grid_forcing_field=eul_grid_forcing_field
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_compute_interaction_with_eul_grid_forcing_reset(grid_dim, n_values, precision):
    real_t = get_real_t(precision)
    mock_soln = MockVirtualBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    mock_soln.compute_interaction_step_solution()
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=mock_soln.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=mock_soln.virtual_boundary_damping_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        num_lag_nodes=mock_soln.num_lag_nodes,
        real_t=mock_soln.real_t,
        enable_eul_grid_forcing_reset=True,
        num_threads=psutil.cpu_count(logical=False),
    )
    eul_grid_velocity_shape = (grid_dim,) + (n_values,) * grid_dim
    eul_grid_forcing_field = np.random.rand(*eul_grid_velocity_shape).astype(real_t)
    virtual_boundary_forcing.compute_interaction_forcing(
        eul_grid_forcing_field=eul_grid_forcing_field,
        eul_grid_velocity_field=mock_soln.eul_grid_velocity_field,
        lag_grid_position_field=mock_soln.lag_grid_position_field,
        lag_grid_velocity_field=mock_soln.lag_grid_velocity_field,
    )
    mock_soln.check_lag_grid_interaction_solution(
        virtual_boundary_forcing=virtual_boundary_forcing
    )
    mock_soln.check_eul_grid_interaction_solution(
        eul_grid_forcing_field=eul_grid_forcing_field
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_time_step(grid_dim, n_values, precision):
    real_t = get_real_t(precision)
    mock_soln = MockVirtualBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    virtual_boundary_forcing = VirtualBoundaryForcing(
        virtual_boundary_stiffness_coeff=mock_soln.virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff=mock_soln.virtual_boundary_damping_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        num_lag_nodes=mock_soln.num_lag_nodes,
        real_t=mock_soln.real_t,
    )

    virtual_boundary_forcing.lag_grid_position_mismatch_field[...] = real_t(1.0)
    dt = real_t(0.1)
    virtual_boundary_forcing.lag_grid_velocity_mismatch_field[...] = real_t(2.0)
    virtual_boundary_forcing.time_step(dt=dt)
    # pos = pos + dt * vel = 1.0 + 0.1 * 2.0 = 1.2
    ref_lag_grid_position_mismatch_field = real_t(1.2) * np.ones_like(
        virtual_boundary_forcing.lag_grid_position_mismatch_field
    )
    ref_time = dt
    np.testing.assert_allclose(
        ref_lag_grid_position_mismatch_field,
        virtual_boundary_forcing.lag_grid_position_mismatch_field,
        atol=mock_soln.test_tol,
    )
    np.testing.assert_allclose(
        ref_time,
        virtual_boundary_forcing.time,
        atol=mock_soln.test_tol,
    )

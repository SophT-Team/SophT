import numpy as np

import pytest

from sopht.numeric.immersed_boundary_ops import BrinkmannBoundaryForcing
from sopht.utils.precision import get_real_t, get_test_tol


class MockBrinkmannBoundaryForcingSolution:
    """Mock solution test class for Brinkmann boundary forcing."""

    def __init__(self, grid_size, grid_dim=2, precision="single"):
        """Class initialiser."""
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.real_t = real_t
        self.brinkmann_coeff = real_t(1e3)
        self.dx = real_t(1.0 / grid_size)
        self.eul_grid_coord_shift = real_t(self.dx / 2)
        self.num_lag_nodes = 3
        self.interp_kernel_width = 2
        self.grid_dim = grid_dim
        self.grid_size = grid_size
        self.dt = real_t(2)

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
        self.dt = self.real_t(0.5) * self.dx
        self.lag_grid_penalised_velocity_field = (
            self.lag_grid_flow_velocity_field
            + self.brinkmann_coeff * self.dt * self.lag_grid_velocity_field
        ) / (1 + self.brinkmann_coeff * self.dt)
        self.lag_grid_penalisation_flux = (
            self.lag_grid_penalised_velocity_field - self.lag_grid_flow_velocity_field
        )
        self.lag_grid_penalisation_forcing = (
            (self.dx**self.grid_dim) * (self.lag_grid_penalisation_flux) / self.dt
        )
        # max number of Eulerian grid indices affected by Lagrangian grid
        self.max_num_of_eul_grid_idx_impacted_by_lag_grid = (
            self.grid_dim
            * self.num_lag_nodes
            * (2 * self.interp_kernel_width) ** self.grid_dim
        )

    def check_interaction_step_solution(
        self, brinkmann_boundary_forcing, eul_grid_penalisation_flux
    ):
        """Check solution for substeps in the interaction step."""
        np.testing.assert_allclose(
            brinkmann_boundary_forcing.nearest_eul_grid_index_to_lag_grid,
            self.nearest_eul_grid_index_to_lag_grid,
            atol=self.test_tol,
        )
        np.testing.assert_allclose(
            brinkmann_boundary_forcing.lag_grid_flow_velocity_field,
            self.lag_grid_flow_velocity_field,
            atol=self.test_tol,
        )
        np.testing.assert_allclose(
            brinkmann_boundary_forcing.lag_grid_penalised_velocity_field,
            self.lag_grid_penalised_velocity_field,
            atol=self.test_tol,
        )
        np.testing.assert_allclose(
            brinkmann_boundary_forcing.lag_grid_penalisation_flux,
            self.lag_grid_penalisation_flux,
            atol=self.test_tol,
        )
        np.testing.assert_allclose(
            brinkmann_boundary_forcing.lag_grid_penalisation_forcing,
            self.lag_grid_penalisation_forcing,
            atol=self.test_tol,
        )
        # check if force is added on Eulerian grid within max impact zone
        assert self.max_num_of_eul_grid_idx_impacted_by_lag_grid >= np.sum(
            eul_grid_penalisation_flux > 0
        )
        # momentum conservation test
        eul_grid_penalisation_flux_sum = np.sum(eul_grid_penalisation_flux)
        np.testing.assert_allclose(
            eul_grid_penalisation_flux_sum,
            np.sum(self.lag_grid_penalisation_flux),
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_brinkmann_boundary_forcing_init(grid_dim, n_values, precision):
    mock_soln = MockBrinkmannBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    brinkmann_boundary_forcing = BrinkmannBoundaryForcing(
        brinkmann_coeff=mock_soln.brinkmann_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        num_lag_nodes=mock_soln.num_lag_nodes,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
    )

    assert brinkmann_boundary_forcing.grid_dim == grid_dim
    assert brinkmann_boundary_forcing.dx == mock_soln.dx
    assert brinkmann_boundary_forcing.brinkmann_coeff == mock_soln.brinkmann_coeff
    assert brinkmann_boundary_forcing.nearest_eul_grid_index_to_lag_grid.dtype == int
    assert brinkmann_boundary_forcing.nearest_eul_grid_index_to_lag_grid.shape == (
        grid_dim,
        mock_soln.num_lag_nodes,
    )
    # figure out a clean way for this!
    if grid_dim == 2:
        assert brinkmann_boundary_forcing.local_eul_grid_support_of_lag_grid.shape == (
            grid_dim,
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mock_soln.num_lag_nodes,
        )
        assert brinkmann_boundary_forcing.interp_weights.shape == (
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mock_soln.num_lag_nodes,
        )
    elif grid_dim == 3:
        assert brinkmann_boundary_forcing.local_eul_grid_support_of_lag_grid.shape == (
            grid_dim,
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mock_soln.num_lag_nodes,
        )
        assert brinkmann_boundary_forcing.interp_weights.shape == (
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            2 * mock_soln.interp_kernel_width,
            mock_soln.num_lag_nodes,
        )
    assert brinkmann_boundary_forcing.lag_grid_flow_velocity_field.shape == (
        grid_dim,
        mock_soln.num_lag_nodes,
    )
    assert brinkmann_boundary_forcing.lag_grid_penalised_velocity_field.shape == (
        grid_dim,
        mock_soln.num_lag_nodes,
    )
    assert brinkmann_boundary_forcing.lag_grid_penalisation_flux.shape == (
        grid_dim,
        mock_soln.num_lag_nodes,
    )
    assert brinkmann_boundary_forcing.lag_grid_penalisation_forcing.shape == (
        grid_dim,
        mock_soln.num_lag_nodes,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_brinkmann_penalise_lag_grid_velocity_field(grid_dim, n_values, precision):
    real_t = get_real_t(precision)
    mock_soln = MockBrinkmannBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    brinkmann_boundary_forcing = BrinkmannBoundaryForcing(
        brinkmann_coeff=mock_soln.brinkmann_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        num_lag_nodes=mock_soln.num_lag_nodes,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
    )

    lag_grid_velocity_field = np.ones((grid_dim, mock_soln.num_lag_nodes), dtype=real_t)
    brinkmann_boundary_forcing.lag_grid_flow_velocity_field = (
        real_t(3) * lag_grid_velocity_field
    )
    brinkmann_boundary_forcing.brinkmann_penalise_lag_grid_velocity_field(
        lag_grid_penalised_velocity_field=brinkmann_boundary_forcing.lag_grid_penalised_velocity_field,
        lag_grid_flow_velocity_field=brinkmann_boundary_forcing.lag_grid_flow_velocity_field,
        lag_grid_body_velocity_field=lag_grid_velocity_field,
        brinkmann_coeff=brinkmann_boundary_forcing.brinkmann_coeff,
        dt=mock_soln.dt,
    )
    # V_pen = (V_flow + brink_coeff * dt * V_body) / (1 + brink_coeff * dt)
    ref_lag_grid_penalised_velocity_field = (
        3 + 1 * brinkmann_boundary_forcing.brinkmann_coeff * mock_soln.dt
    ) / (1 + brinkmann_boundary_forcing.brinkmann_coeff * mock_soln.dt)
    np.testing.assert_allclose(
        ref_lag_grid_penalised_velocity_field,
        brinkmann_boundary_forcing.lag_grid_penalised_velocity_field,
        atol=mock_soln.test_tol,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_compute_lag_grid_penalisation_flux(grid_dim, n_values, precision):
    real_t = get_real_t(precision)
    mock_soln = MockBrinkmannBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    brinkmann_boundary_forcing = BrinkmannBoundaryForcing(
        brinkmann_coeff=mock_soln.brinkmann_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        num_lag_nodes=mock_soln.num_lag_nodes,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
    )

    brinkmann_boundary_forcing.lag_grid_flow_velocity_field[...] = real_t(1.0)
    brinkmann_boundary_forcing.lag_grid_penalised_velocity_field[...] = real_t(2.0)
    brinkmann_boundary_forcing.compute_lag_grid_penalisation_flux(
        lag_grid_penalisation_flux=brinkmann_boundary_forcing.lag_grid_penalisation_flux,
        lag_grid_penalised_velocity_field=brinkmann_boundary_forcing.lag_grid_penalised_velocity_field,
        lag_grid_flow_velocity_field=brinkmann_boundary_forcing.lag_grid_flow_velocity_field,
    )
    # 2 - 1 = 1
    ref_lag_grid_penalisation_flux = real_t(1.0)
    np.testing.assert_allclose(
        ref_lag_grid_penalisation_flux,
        brinkmann_boundary_forcing.lag_grid_penalisation_flux,
        atol=mock_soln.test_tol,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_compute_lag_grid_penalisation_forcing(grid_dim, n_values, precision):
    real_t = get_real_t(precision)
    mock_soln = MockBrinkmannBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    brinkmann_boundary_forcing = BrinkmannBoundaryForcing(
        brinkmann_coeff=mock_soln.brinkmann_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        num_lag_nodes=mock_soln.num_lag_nodes,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
    )

    brinkmann_boundary_forcing.lag_grid_penalisation_flux[...] = real_t(2.0)
    brinkmann_boundary_forcing.compute_lag_grid_penalisation_forcing(mock_soln.dt)
    ref_lag_grid_penalisation_forcing = (
        (mock_soln.dx**mock_soln.grid_dim) * real_t(2.0) / mock_soln.dt
    )
    np.testing.assert_allclose(
        ref_lag_grid_penalisation_forcing,
        brinkmann_boundary_forcing.lag_grid_penalisation_forcing,
        atol=mock_soln.test_tol,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_compute_interaction_without_eul_grid_flux_reset(grid_dim, n_values, precision):
    real_t = get_real_t(precision)
    mock_soln = MockBrinkmannBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    mock_soln.compute_interaction_step_solution()
    brinkmann_boundary_forcing = BrinkmannBoundaryForcing(
        brinkmann_coeff=mock_soln.brinkmann_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        num_lag_nodes=mock_soln.num_lag_nodes,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
        enable_eul_grid_flux_reset=False,
    )

    eul_grid_velocity_shape = (grid_dim,) + (n_values,) * grid_dim
    eul_grid_penalisation_flux = np.zeros(eul_grid_velocity_shape, dtype=real_t)
    brinkmann_boundary_forcing.compute_interaction_forcing(
        eul_grid_penalisation_flux=eul_grid_penalisation_flux,
        eul_grid_velocity_field=mock_soln.eul_grid_velocity_field,
        lag_grid_position_field=mock_soln.lag_grid_position_field,
        lag_grid_velocity_field=mock_soln.lag_grid_velocity_field,
        dt=mock_soln.dt,
    )
    mock_soln.check_interaction_step_solution(
        brinkmann_boundary_forcing=brinkmann_boundary_forcing,
        eul_grid_penalisation_flux=eul_grid_penalisation_flux,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_values", [16])
def test_compute_interaction_with_eul_grid_flux_reset(grid_dim, n_values, precision):
    real_t = get_real_t(precision)
    mock_soln = MockBrinkmannBoundaryForcingSolution(
        grid_size=n_values,
        grid_dim=grid_dim,
        precision=precision,
    )
    mock_soln.compute_interaction_step_solution()
    brinkmann_boundary_forcing = BrinkmannBoundaryForcing(
        brinkmann_coeff=mock_soln.brinkmann_coeff,
        grid_dim=mock_soln.grid_dim,
        dx=mock_soln.dx,
        eul_grid_coord_shift=mock_soln.eul_grid_coord_shift,
        num_lag_nodes=mock_soln.num_lag_nodes,
        interp_kernel_width=mock_soln.interp_kernel_width,
        real_t=mock_soln.real_t,
        enable_eul_grid_flux_reset=True,
    )
    eul_grid_velocity_shape = (grid_dim,) + (n_values,) * grid_dim
    eul_grid_penalisation_flux = np.random.rand(*eul_grid_velocity_shape).astype(real_t)
    brinkmann_boundary_forcing.compute_interaction_forcing(
        eul_grid_penalisation_flux=eul_grid_penalisation_flux,
        eul_grid_velocity_field=mock_soln.eul_grid_velocity_field,
        lag_grid_position_field=mock_soln.lag_grid_position_field,
        lag_grid_velocity_field=mock_soln.lag_grid_velocity_field,
        dt=mock_soln.dt,
    )
    mock_soln.check_interaction_step_solution(
        brinkmann_boundary_forcing=brinkmann_boundary_forcing,
        eul_grid_penalisation_flux=eul_grid_penalisation_flux,
    )

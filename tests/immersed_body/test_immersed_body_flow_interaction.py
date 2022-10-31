import logging
import numpy as np
import pytest
from sopht.utils.precision import get_real_t
import sopht_simulator as sps
from tests.immersed_body.rigid_body.test_rigid_body_forcing_grids import (
    mock_2d_cylinder,
)


def mock_2d_cylinder_flow_interactor(num_forcing_points=16):
    """Returns a mock 2D cylinder flow interactor and related fields for testing"""
    grid_dim = 2
    cylinder = mock_2d_cylinder()
    grid_size = (16,) * grid_dim
    eul_grid_velocity_field = np.random.rand(grid_dim, *grid_size)
    eul_grid_forcing_field = np.zeros_like(eul_grid_velocity_field)
    # chosen so that cylinder lies within domain
    dx = cylinder.length / 4.0
    cylinder_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=cylinder,
        eul_grid_forcing_field=eul_grid_forcing_field,
        eul_grid_velocity_field=eul_grid_velocity_field,
        virtual_boundary_stiffness_coeff=1.0,
        virtual_boundary_damping_coeff=1.0,
        dx=dx,
        grid_dim=grid_dim,
        real_t=get_real_t(),
        forcing_grid_cls=sps.CircularCylinderForcingGrid,
        num_forcing_points=num_forcing_points,
    )
    return cylinder_flow_interactor, eul_grid_forcing_field, eul_grid_velocity_field, dx


@pytest.mark.parametrize("num_forcing_points", [1, 4, 64])
def test_immersed_body_interactor_warnings(num_forcing_points, caplog):
    with caplog.at_level(logging.WARNING):
        cylinder_flow_interactor, _, _, dx = mock_2d_cylinder_flow_interactor(
            num_forcing_points
        )
    max_lag_grid_dx = (
        cylinder_flow_interactor.forcing_grid.get_maximum_lagrangian_grid_spacing()
    )
    if max_lag_grid_dx > 2 * dx:
        warning_message = (
            f"Eulerian grid spacing (dx): {dx}"
            f"\nMax Lagrangian grid spacing: {max_lag_grid_dx} > 2 * dx"
            "\nThe Lagrangian grid of the body is too coarse relative to"
            "\nthe Eulerian grid of the flow, which can lead to unexpected"
            "\nconvergence. Please make the Lagrangian grid finer."
        )
    elif max_lag_grid_dx < 0.5 * dx:
        warning_message = (
            "==========================================================\n"
            f"Eulerian grid spacing (dx): {dx}"
            f"\nMax Lagrangian grid spacing: {max_lag_grid_dx} < 0.5 * dx"
            "\nThe Lagrangian grid of the body is too fine relative to"
            "\nthe Eulerian grid of the flow, which corresponds to redundant"
            "\nforcing points. Please make the Lagrangian grid coarser."
        )
    else:
        warning_message = (
            "Lagrangian grid is resolved almost the same as the Eulerian"
            "\ngrid of the flow."
        )
    assert warning_message in caplog.text


def test_immersed_body_interactor_call_method():
    (
        cylinder_flow_interactor,
        eul_grid_forcing_field,
        eul_grid_velocity_field,
        _,
    ) = mock_2d_cylinder_flow_interactor()
    cylinder_flow_interactor()

    ref_eul_grid_forcing_field = np.zeros_like(eul_grid_forcing_field)
    forcing_grid = cylinder_flow_interactor.forcing_grid
    # correct procedure
    forcing_grid.compute_lag_grid_position_field()
    forcing_grid.compute_lag_grid_velocity_field()
    cylinder_flow_interactor.compute_interaction_forcing(
        eul_grid_forcing_field=ref_eul_grid_forcing_field,
        eul_grid_velocity_field=eul_grid_velocity_field,
        lag_grid_position_field=forcing_grid.position_field,
        lag_grid_velocity_field=forcing_grid.velocity_field,
    )

    np.testing.assert_allclose(ref_eul_grid_forcing_field, eul_grid_forcing_field)


def test_immersed_body_interactor_compute_flow_forces_and_torques():
    (
        cylinder_flow_interactor,
        eul_grid_forcing_field,
        eul_grid_velocity_field,
        _,
    ) = mock_2d_cylinder_flow_interactor()
    cylinder_flow_interactor.compute_flow_forces_and_torques()

    ref_body_flow_forces = np.zeros_like(cylinder_flow_interactor.body_flow_forces)
    ref_body_flow_torques = np.zeros_like(cylinder_flow_interactor.body_flow_torques)
    forcing_grid = cylinder_flow_interactor.forcing_grid
    # correct procedure
    forcing_grid.compute_lag_grid_position_field()
    forcing_grid.compute_lag_grid_velocity_field()
    cylinder_flow_interactor.compute_interaction_force_on_lag_grid(
        eul_grid_velocity_field=eul_grid_velocity_field,
        lag_grid_position_field=forcing_grid.position_field,
        lag_grid_velocity_field=forcing_grid.velocity_field,
    )
    forcing_grid.transfer_forcing_from_grid_to_body(
        body_flow_forces=ref_body_flow_forces,
        body_flow_torques=ref_body_flow_torques,
        lag_grid_forcing_field=cylinder_flow_interactor.lag_grid_forcing_field,
    )
    np.testing.assert_allclose(
        ref_body_flow_forces, cylinder_flow_interactor.body_flow_forces
    )
    np.testing.assert_allclose(
        ref_body_flow_torques, cylinder_flow_interactor.body_flow_torques
    )


def test_immersed_body_interactor_get_grid_deviation_error_l2_norm():
    cylinder_flow_interactor, _, _, _ = mock_2d_cylinder_flow_interactor()
    fixed_val = 2.0
    cylinder_flow_interactor.lag_grid_position_mismatch_field[...] = fixed_val
    grid_dev_error_l2_norm = cylinder_flow_interactor.get_grid_deviation_error_l2_norm()
    grid_dim = cylinder_flow_interactor.grid_dim
    ref_grid_dev_error_l2_norm = fixed_val * np.sqrt(grid_dim)
    np.testing.assert_allclose(grid_dev_error_l2_norm, ref_grid_dev_error_l2_norm)

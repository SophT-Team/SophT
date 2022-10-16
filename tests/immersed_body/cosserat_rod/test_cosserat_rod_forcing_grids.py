import elastica as ea
import numpy as np
import pytest
import sopht_simulator as sps


def mock_straight_rod(n_elems):
    """Returns a straight rod aligned x = y = z plane for testing."""
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 1.0, 1.0])
    normal = np.array([0.0, -1.0, 1.0])
    rod_length = 1.0
    base_radius = 0.05
    staight_rod = ea.CosseratRod.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        rod_length,
        base_radius,
        density=1e3,
        nu=0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus=1e6,
        shear_modulus=1e6 / (0.5 + 1.0),
    )
    n_nodes = n_elems + 1
    staight_rod.velocity_collection[...] = np.linspace(1, n_nodes, n_nodes)
    return staight_rod


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_element_centric_grid_grid_kinematics(grid_dim, n_elems):
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodElementCentricForcingGrid(
        grid_dim=grid_dim, cosserat_rod=straight_rod
    )

    # check if setup is correct
    assert rod_forcing_grid.cosserat_rod is straight_rod
    assert rod_forcing_grid.num_lag_nodes == straight_rod.n_elems
    assert rod_forcing_grid.position_field.shape == (grid_dim, n_elems)
    assert rod_forcing_grid.velocity_field.shape == (grid_dim, n_elems)

    # check if position is correct; the rod is a straight one with same
    # element size
    grid_start = np.mean(straight_rod.position_collection[..., :2], axis=1)
    grid_end = np.mean(straight_rod.position_collection[..., -2:], axis=1)
    correct_position_field = np.zeros_like(rod_forcing_grid.position_field)
    for axis in range(grid_dim):
        correct_position_field[axis] = np.linspace(
            grid_start[axis], grid_end[axis], n_elems
        )
    np.testing.assert_allclose(rod_forcing_grid.position_field, correct_position_field)

    # check if velocities are correct, in mock rod they are initialised as
    # linearly increasing along the rod
    grid_start_velocity = np.mean(straight_rod.velocity_collection[..., :2], axis=1)
    grid_end_velocity = np.mean(straight_rod.velocity_collection[..., -2:], axis=1)
    correct_velocity_field = np.zeros_like(rod_forcing_grid.velocity_field)
    for axis in range(grid_dim):
        correct_velocity_field[axis] = np.linspace(
            grid_start_velocity[axis], grid_end_velocity[axis], n_elems
        )
    np.testing.assert_allclose(rod_forcing_grid.velocity_field, correct_velocity_field)


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_element_centric_grid_force_transfer(grid_dim, n_elems):
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodElementCentricForcingGrid(
        grid_dim=grid_dim, cosserat_rod=straight_rod
    )
    rod_dim = 3
    body_flow_forces = np.zeros((rod_dim, n_elems + 1))
    body_flow_torques = np.zeros((rod_dim, n_elems))
    lag_grid_forcing_field = np.zeros((grid_dim, rod_forcing_grid.num_lag_nodes))
    uniform_forcing = np.linspace(1.0, grid_dim, grid_dim).reshape(-1, 1)
    lag_grid_forcing_field[...] = uniform_forcing
    rod_forcing_grid.transfer_forcing_from_grid_to_body(
        body_flow_forces=body_flow_forces,
        body_flow_torques=body_flow_torques,
        lag_grid_forcing_field=lag_grid_forcing_field,
    )

    # check if forces are correct
    correct_body_flow_forces = np.zeros_like(body_flow_forces)
    correct_body_flow_forces[:grid_dim, ...] = -uniform_forcing
    correct_body_flow_forces[:grid_dim, (0, -1)] *= 0.5
    np.testing.assert_allclose(body_flow_forces, correct_body_flow_forces)

    # torques stay 0 for this grid
    np.testing.assert_allclose(body_flow_torques, 0.0)


@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_element_centric_grid_spacing(n_elems):
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodElementCentricForcingGrid(
        grid_dim=3, cosserat_rod=straight_rod
    )
    max_grid_spacing = rod_forcing_grid.get_maximum_lagrangian_grid_spacing()
    # rod with same element sizes so max is one of any lengths
    correct_max_grid_spacing = straight_rod.lengths[0]
    assert correct_max_grid_spacing == max_grid_spacing

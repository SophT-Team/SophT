import elastica as ea
import numpy as np
import pytest
import sopht_simulator as sps
from elastica.interaction import node_to_element_velocity, elements_to_nodes_inplace


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
    staight_rod.omega_collection[...] = np.linspace(1, n_elems, n_elems)
    return staight_rod


@pytest.mark.parametrize("n_elems", [8, 16])
def test_pyelastica_node_to_element_velocity_func_validity(n_elems):
    """
    Testing validity of node to element function of pyelastica
    """
    straight_rod = mock_straight_rod(n_elems)

    element_velocity = node_to_element_velocity(
        straight_rod.mass, straight_rod.velocity_collection
    )

    correct_velocity = 0.5 * (
        straight_rod.velocity_collection[:, 1:]
        + straight_rod.velocity_collection[:, :-1]
    )
    correct_velocity[..., 0] = (
        straight_rod.velocity_collection[:, 0]
        + 2 * straight_rod.velocity_collection[:, 1]
    ) / 3
    correct_velocity[..., -1] = (
        straight_rod.velocity_collection[:, -1]
        + 2 * straight_rod.velocity_collection[:, -2]
    ) / 3

    np.testing.assert_allclose(element_velocity, correct_velocity)


@pytest.mark.parametrize("n_elems", [8, 16])
def test_pyelastica_elements_to_nodes_inplace(n_elems):
    """
    Testing validity of elements to nodes inplace function of pyelastica
    """
    n_nodes = n_elems + 1
    mock_vector = np.random.random((3, n_elems))
    test_vector = np.zeros((3, n_nodes))
    correct_vector = np.zeros((3, n_nodes))
    correct_vector[:, 1:] += 0.5 * mock_vector
    correct_vector[:, :-1] += 0.5 * mock_vector

    elements_to_nodes_inplace(mock_vector, test_vector)

    np.testing.assert_allclose(test_vector, correct_vector)


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
        # Special treatment at the end elements, to conserve momentum
        correct_velocity_field[axis, 0] = (
            straight_rod.velocity_collection[axis, 0]
            + 2 * straight_rod.velocity_collection[axis, 1]
        ) / 3
        correct_velocity_field[axis, -1] = (
            straight_rod.velocity_collection[axis, -1]
            + 2 * straight_rod.velocity_collection[axis, -2]
        ) / 3

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


# Edge Forcing Grid tests
@pytest.mark.parametrize("grid_dim", [0, 1, 3, 4])
@pytest.mark.parametrize("n_elems", [8, 16])
@pytest.mark.xfail(raises=ValueError)
def test_rod_edge_grid_grid_kinematics(grid_dim, n_elems):
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodEdgeForcingGrid(
        grid_dim=grid_dim, cosserat_rod=straight_rod
    )


class MockEdgeForcingGrid:
    def __init__(self, n_elems):

        self.num_lag_nodes = 3 * n_elems
        self.z_vector = np.zeros((3, n_elems))
        self.z_vector[-1, :] = 1.0
        self.start_idx_elems = 0
        self.end_idx_elems = n_elems
        self.start_idx_left_edge_nodes = n_elems
        self.end_idx_left_edge_nodes = 2 * n_elems
        self.start_idx_right_edge_nodes = 2 * n_elems
        self.end_idx_right_edge_nodes = 3 * n_elems

        self.position_field = np.zeros((2, self.num_lag_nodes))
        self.velocity_field = np.zeros((2, self.num_lag_nodes))
        self.moment_arm = np.zeros((3, n_elems))


@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_edge_grid_grid_setup(n_elems):
    grid_dim = 2
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodEdgeForcingGrid(
        grid_dim=grid_dim, cosserat_rod=straight_rod
    )

    correct_forcing_grid = MockEdgeForcingGrid(n_elems)

    # check if setup is correct
    assert rod_forcing_grid.cosserat_rod is straight_rod
    assert rod_forcing_grid.num_lag_nodes == correct_forcing_grid.num_lag_nodes
    assert (
        rod_forcing_grid.position_field.shape
        == correct_forcing_grid.position_field.shape
    )
    assert (
        rod_forcing_grid.velocity_field.shape
        == correct_forcing_grid.velocity_field.shape
    )
    assert rod_forcing_grid.moment_arm.shape == correct_forcing_grid.moment_arm.shape

    np.testing.assert_allclose(
        rod_forcing_grid.start_idx_elems, correct_forcing_grid.start_idx_elems
    )
    np.testing.assert_allclose(
        rod_forcing_grid.end_idx_elems, correct_forcing_grid.end_idx_elems
    )
    np.testing.assert_allclose(
        rod_forcing_grid.start_idx_left_edge_nodes,
        correct_forcing_grid.start_idx_left_edge_nodes,
    )
    np.testing.assert_allclose(
        rod_forcing_grid.end_idx_left_edge_nodes,
        correct_forcing_grid.end_idx_left_edge_nodes,
    )
    np.testing.assert_allclose(
        rod_forcing_grid.start_idx_right_edge_nodes,
        correct_forcing_grid.start_idx_right_edge_nodes,
    )
    np.testing.assert_allclose(
        rod_forcing_grid.end_idx_right_edge_nodes,
        correct_forcing_grid.end_idx_right_edge_nodes,
    )


@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_edge_grid_grid_kinematics(n_elems):
    grid_dim = 2
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodEdgeForcingGrid(
        grid_dim=grid_dim, cosserat_rod=straight_rod
    )
    correct_forcing_grid = MockEdgeForcingGrid(n_elems)

    # Compute the correct moment arm first
    tangents = np.array([1.0, 1.0, 1.0])
    tangents /= np.linalg.norm(tangents)
    normal = np.cross(np.array([0, 0, 1.0]), tangents)
    correct_forcing_grid.moment_arm[:] = straight_rod.radius * normal.reshape(3, 1)

    # Check if moment arm is correct
    np.testing.assert_allclose(
        rod_forcing_grid.moment_arm, correct_forcing_grid.moment_arm
    )

    # Compute the correct grid position
    grid_start = np.mean(straight_rod.position_collection[..., :2], axis=1)
    grid_end = np.mean(straight_rod.position_collection[..., -2:], axis=1)
    for axis in range(grid_dim):
        element_position = np.linspace(grid_start[axis], grid_end[axis], n_elems)
        correct_forcing_grid.position_field[
            axis,
            correct_forcing_grid.start_idx_elems : correct_forcing_grid.end_idx_elems,
        ] = element_position
        correct_forcing_grid.position_field[
            axis,
            correct_forcing_grid.start_idx_left_edge_nodes : correct_forcing_grid.end_idx_left_edge_nodes,
        ] = (
            element_position + correct_forcing_grid.moment_arm[axis]
        )
        correct_forcing_grid.position_field[
            axis,
            correct_forcing_grid.start_idx_right_edge_nodes : correct_forcing_grid.end_idx_right_edge_nodes,
        ] = (
            element_position - correct_forcing_grid.moment_arm[axis]
        )

    # Check if moment arm is correct
    np.testing.assert_allclose(
        rod_forcing_grid.position_field, correct_forcing_grid.position_field
    )

    # Compute the correct grid velocity
    # check if velocities are correct, in mock rod they are initialised as
    # linearly increasing along the rod
    grid_start_velocity = np.mean(straight_rod.velocity_collection[..., :2], axis=1)
    grid_end_velocity = np.mean(straight_rod.velocity_collection[..., -2:], axis=1)

    # Compute omega x moment arm
    director_transpose = straight_rod.director_collection[:, :, 0].T
    omega = straight_rod.omega_collection.copy()
    omega_cross_moment_arm = np.zeros((3, n_elems))
    for i in range(n_elems):
        omega_in_lab_frame = director_transpose @ omega[:, i]
        omega_cross_moment_arm[:, i] = np.cross(
            omega_in_lab_frame, correct_forcing_grid.moment_arm[:, i]
        )

    for axis in range(grid_dim):
        element_velocity = np.linspace(
            grid_start_velocity[axis], grid_end_velocity[axis], n_elems
        )
        # Special treatment at the end elements, to conserve momentum
        element_velocity[0] = (
            straight_rod.velocity_collection[axis, 0]
            + 2 * straight_rod.velocity_collection[axis, 1]
        ) / 3
        element_velocity[-1] = (
            straight_rod.velocity_collection[axis, -1]
            + 2 * straight_rod.velocity_collection[axis, -2]
        ) / 3

        correct_forcing_grid.velocity_field[
            axis,
            correct_forcing_grid.start_idx_elems : correct_forcing_grid.end_idx_elems,
        ] = element_velocity
        correct_forcing_grid.velocity_field[
            axis,
            correct_forcing_grid.start_idx_left_edge_nodes : correct_forcing_grid.end_idx_left_edge_nodes,
        ] = (
            element_velocity + omega_cross_moment_arm[axis, :]
        )
        correct_forcing_grid.velocity_field[
            axis,
            correct_forcing_grid.start_idx_right_edge_nodes : correct_forcing_grid.end_idx_right_edge_nodes,
        ] = (
            element_velocity - omega_cross_moment_arm[axis, :]
        )

    np.testing.assert_allclose(
        rod_forcing_grid.velocity_field[
            :, correct_forcing_grid.start_idx_elems : correct_forcing_grid.end_idx_elems
        ],
        correct_forcing_grid.velocity_field[
            :, correct_forcing_grid.start_idx_elems : correct_forcing_grid.end_idx_elems
        ],
    )


@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_edge_grid_force_transfer(n_elems):
    grid_dim = 2
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodEdgeForcingGrid(
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
    correct_body_flow_forces[:grid_dim, ...] = -3 * uniform_forcing
    correct_body_flow_forces[:grid_dim, (0, -1)] *= 0.5
    np.testing.assert_allclose(body_flow_forces, correct_body_flow_forces)

    # torques stay 0 for this loading
    np.testing.assert_allclose(body_flow_torques, 0.0)


@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_edge_grid_spacing(n_elems):
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodEdgeForcingGrid(
        grid_dim=2, cosserat_rod=straight_rod
    )
    max_grid_spacing = rod_forcing_grid.get_maximum_lagrangian_grid_spacing()
    # rod with same element sizes so max is one of any lengths
    correct_max_grid_spacing = straight_rod.lengths[0]
    assert correct_max_grid_spacing == max_grid_spacing

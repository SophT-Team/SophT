import elastica as ea
import numpy as np
import pytest
import sopht.simulator as sps
from elastica.interaction import node_to_element_velocity, elements_to_nodes_inplace
from sopht.utils.precision import get_test_tol


def mock_straight_rod(n_elems, **kwargs):
    """Returns a straight rod aligned x = y = z plane for testing."""
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 1.0, 1.0])
    normal = np.array([0.0, -1.0, 1.0])
    rod_length = 1.0
    base_radius = kwargs.get("base_radius", 0.05)
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
def test_rod_nodal_grid_kinematics(grid_dim, n_elems):
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodNodalForcingGrid(
        grid_dim=grid_dim, cosserat_rod=straight_rod
    )

    # check if setup is correct
    n_nodes = straight_rod.n_elems + 1
    assert rod_forcing_grid.cosserat_rod is straight_rod
    assert rod_forcing_grid.num_lag_nodes == n_nodes
    assert rod_forcing_grid.position_field.shape == (grid_dim, n_nodes)
    assert rod_forcing_grid.velocity_field.shape == (grid_dim, n_nodes)

    # check if position is correct; the rod is a straight one with same
    # element size
    correct_position_field = straight_rod.position_collection[:grid_dim]
    np.testing.assert_allclose(rod_forcing_grid.position_field, correct_position_field)

    # check if velocities are correct, in mock rod they are initialised as
    # linearly increasing along the rod
    correct_velocity_field = straight_rod.velocity_collection[:grid_dim]
    np.testing.assert_allclose(rod_forcing_grid.velocity_field, correct_velocity_field)


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_nodal_grid_force_transfer(grid_dim, n_elems):
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodNodalForcingGrid(
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
    np.testing.assert_allclose(body_flow_forces, correct_body_flow_forces)

    # torques stay 0 for this grid
    correct_body_flow_torques = np.zeros_like(body_flow_torques)
    # endpoint corrections
    moment_arm = (
        straight_rod.position_collection[..., 1:]
        - straight_rod.position_collection[..., :-1]
    ) / 2.0
    correct_body_flow_torques[..., -1] += straight_rod.director_collection[..., -1] @ (
        np.cross(
            moment_arm[..., -1],
            correct_body_flow_forces[..., -1],
        )
        / 2.0
    )
    correct_body_flow_torques[..., 0] -= straight_rod.director_collection[..., 0] @ (
        np.cross(
            moment_arm[..., 0],
            correct_body_flow_forces[..., 0],
        )
        / 2.0
    )
    np.testing.assert_allclose(body_flow_torques, correct_body_flow_torques)


@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_nodal_grid_spacing(n_elems):
    straight_rod = mock_straight_rod(n_elems)
    rod_forcing_grid = sps.CosseratRodNodalForcingGrid(
        grid_dim=3, cosserat_rod=straight_rod
    )
    max_grid_spacing = rod_forcing_grid.get_maximum_lagrangian_grid_spacing()
    # rod with same element sizes so max is one of any lengths
    correct_max_grid_spacing = straight_rod.lengths[0]
    assert correct_max_grid_spacing == max_grid_spacing


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
class MockEdgeForcingGrid:
    def __init__(self, n_elems):

        self.num_lag_nodes = 3 * n_elems
        self.rod_dim = 3
        self.z_vector = np.zeros((self.rod_dim, n_elems))
        self.z_vector[-1, :] = 1.0
        self.start_idx_elems = 0
        self.end_idx_elems = n_elems
        self.start_idx_left_edge_nodes = n_elems
        self.end_idx_left_edge_nodes = 2 * n_elems
        self.start_idx_right_edge_nodes = 2 * n_elems
        self.end_idx_right_edge_nodes = 3 * n_elems

        self.grid_dim = 2
        self.position_field = np.zeros((self.grid_dim, self.num_lag_nodes))
        self.velocity_field = np.zeros((self.grid_dim, self.num_lag_nodes))
        self.moment_arm = np.zeros((self.rod_dim, n_elems))


@pytest.mark.parametrize("grid_dim", [0, 1, 3, 4])
@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_edge_grid_dimension(grid_dim, n_elems):
    straight_rod = mock_straight_rod(n_elems)
    with pytest.raises(ValueError) as exc_info:
        _ = sps.CosseratRodEdgeForcingGrid(grid_dim=grid_dim, cosserat_rod=straight_rod)
    error_msg = (
        "Invalid grid dimensions. Cosserat rod edge forcing grid is only "
        "defined for grid_dim=2"
    )
    assert exc_info.value.args[0] == error_msg


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
    rod_dim = 3
    omega_cross_moment_arm = np.zeros((rod_dim, n_elems))
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
    # 3 = 1 (center) + 1 (left edge) + 1 (right edge)
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


# Surface Forcing Grid tests
class MockSurfaceForcingGrid:
    def __init__(self, n_elems, grid_density, base_radius, with_cap):

        self.surface_grid_density_for_largest_element = grid_density
        self.surface_grid_points = np.zeros((n_elems), dtype=int)
        self.surface_point_rotation_angle_list = []
        self.grid_point_radius_ratio = []
        self.start_idx = np.zeros((n_elems), dtype=int)
        self.end_idx = np.zeros((n_elems), dtype=int)

        idx_temp = 0
        for i in range(n_elems):
            n_point_per_elem = base_radius[i] / np.max(base_radius) * grid_density
            n_point_per_elem = round(n_point_per_elem)

            if n_point_per_elem < 3:
                n_point_per_elem = 1
                self.surface_point_rotation_angle_list.append(np.array([]))
                self.grid_point_radius_ratio.append(np.ones(n_point_per_elem))

            else:
                self.surface_point_rotation_angle_list.append(
                    np.array(
                        [
                            2 * np.pi / n_point_per_elem * i
                            for i in range(n_point_per_elem)
                        ]
                    )
                )
                self.grid_point_radius_ratio.append(np.ones(n_point_per_elem))

                # compute grid points for rod end caps if needed
                if with_cap and i in [0, n_elems - 1]:
                    grid_angular_spacing = 2.0 * np.pi / n_point_per_elem
                    end_elem_surface_grid_radial_spacing = (
                        base_radius[i] * grid_angular_spacing
                    )
                    end_elem_radial_grid_density = max(
                        int(base_radius[i] // end_elem_surface_grid_radial_spacing), 1
                    )

                    end_elem_radius_ratio = np.array(
                        [
                            base_radius[i] / end_elem_radial_grid_density * j
                            for j in range(end_elem_radial_grid_density)
                        ]
                    )
                    end_elem_surface_grid_points = np.array(
                        [
                            (n_point_per_elem // end_elem_radial_grid_density - 1) * j
                            + 1
                            for j in range(end_elem_radial_grid_density)
                        ]
                    ).astype(int)

                    n_point_per_elem += end_elem_surface_grid_points.sum()
                    self.surface_point_rotation_angle_list[i] = np.append(
                        self.surface_point_rotation_angle_list[i],
                        (
                            [
                                np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                                for num_points in end_elem_surface_grid_points
                            ]
                        ),
                    )
                    self.surface_point_rotation_angle_list[i] = np.hstack(
                        self.surface_point_rotation_angle_list[i]
                    )

                    # add the radius ratio for inner grid points
                    self.grid_point_radius_ratio[i] = np.append(
                        self.grid_point_radius_ratio[i],
                        np.hstack(
                            [
                                np.ones((num_grid_points)) * end_elem_radius_ratio[j]
                                for j, num_grid_points in enumerate(
                                    end_elem_surface_grid_points
                                )
                            ]
                        ),
                    )

            self.surface_grid_points[i] = n_point_per_elem
            self.start_idx[i] = idx_temp
            idx_temp += n_point_per_elem
            self.end_idx[i] = idx_temp

        self.num_lag_nodes = self.surface_grid_points.sum()
        self.n_elems = n_elems
        grid_dim = 3
        self.position_field = np.zeros((grid_dim, self.num_lag_nodes))
        self.velocity_field = np.zeros((grid_dim, self.num_lag_nodes))
        self.moment_arm = np.zeros((grid_dim, self.num_lag_nodes))
        self.local_frame_surface_points = np.zeros_like(self.position_field)

        for i, angle in enumerate(self.surface_point_rotation_angle_list):
            if angle.size == 0:
                self.local_frame_surface_points[
                    :, self.start_idx[i] : self.end_idx[i]
                ] = 0.0
            else:
                self.local_frame_surface_points[
                    0, self.start_idx[i] : self.end_idx[i]
                ] = np.cos(angle)
                self.local_frame_surface_points[
                    1, self.start_idx[i] : self.end_idx[i]
                ] = np.sin(angle)


@pytest.mark.parametrize("grid_dim", [0, 1, 2, 4])
@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_surface_grid_dimension(grid_dim, n_elems):
    straight_rod = mock_straight_rod(n_elems)
    with pytest.raises(ValueError) as exc_info:
        _ = sps.CosseratRodSurfaceForcingGrid(
            grid_dim=grid_dim,
            cosserat_rod=straight_rod,
            surface_grid_density_for_largest_element=1,
        )
    error_msg = (
        "Invalid grid dimensions. Cosserat rod surface forcing grid is only "
        "defined for grid_dim=3"
    )
    assert exc_info.value.args[0] == error_msg


@pytest.mark.parametrize("n_elems", [8, 16])
@pytest.mark.parametrize("largest_element_grid_density", [16, 12, 8, 4])
@pytest.mark.parametrize("taper_ratio", [1, 2, 5, 10])
@pytest.mark.parametrize("with_cap", [True, False])
def test_rod_surface_grid_setup(
    n_elems, largest_element_grid_density, taper_ratio, with_cap
):
    base_radius = np.linspace(1, 1 / taper_ratio, n_elems)
    straight_rod = mock_straight_rod(n_elems, base_radius=base_radius)

    rod_forcing_grid = sps.CosseratRodSurfaceForcingGrid(
        grid_dim=3,
        cosserat_rod=straight_rod,
        surface_grid_density_for_largest_element=largest_element_grid_density,
        with_cap=with_cap,
    )

    correct_forcing_grid = MockSurfaceForcingGrid(
        n_elems,
        largest_element_grid_density,
        base_radius,
        with_cap=with_cap,
    )

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
        rod_forcing_grid.surface_grid_points, correct_forcing_grid.surface_grid_points
    )

    for i in range(n_elems):
        np.testing.assert_allclose(
            rod_forcing_grid.surface_point_rotation_angle_list[i],
            correct_forcing_grid.surface_point_rotation_angle_list[i],
            atol=get_test_tol(precision="double"),
        )

    np.testing.assert_allclose(
        rod_forcing_grid.start_idx, correct_forcing_grid.start_idx
    )
    np.testing.assert_allclose(rod_forcing_grid.end_idx, correct_forcing_grid.end_idx)
    np.testing.assert_allclose(
        rod_forcing_grid.local_frame_surface_points,
        correct_forcing_grid.local_frame_surface_points,
    )


@pytest.mark.parametrize("n_elems", [8, 16])
@pytest.mark.parametrize("largest_element_grid_density", [16, 12, 8, 4])
@pytest.mark.parametrize("taper_ratio", [1, 2, 5, 10])
@pytest.mark.parametrize("with_cap", [True, False])
def test_rod_surface_grid_grid_kinematics(
    n_elems, largest_element_grid_density, taper_ratio, with_cap
):
    base_radius = np.linspace(1, 1 / taper_ratio, n_elems)
    straight_rod = mock_straight_rod(n_elems, base_radius=base_radius)

    rod_forcing_grid = sps.CosseratRodSurfaceForcingGrid(
        grid_dim=3,
        cosserat_rod=straight_rod,
        surface_grid_density_for_largest_element=largest_element_grid_density,
        with_cap=with_cap,
    )
    correct_forcing_grid = MockSurfaceForcingGrid(
        n_elems,
        largest_element_grid_density,
        base_radius,
        with_cap=with_cap,
    )

    # Compute the correct grid position
    for i in range(n_elems):
        element_pos = 0.5 * (
            straight_rod.position_collection[:, i]
            + straight_rod.position_collection[:, i + 1]
        )
        director_transpose = straight_rod.director_collection[:, :, i].T
        rod_radius = straight_rod.radius[i]
        rod_radius_ratio = correct_forcing_grid.grid_point_radius_ratio[i]

        for j in range(correct_forcing_grid.surface_grid_points[i]):
            grid_idx = correct_forcing_grid.start_idx[i] + j
            correct_forcing_grid.moment_arm[:, grid_idx] = (
                rod_radius
                * rod_radius_ratio[j]
                * director_transpose
                @ correct_forcing_grid.local_frame_surface_points[:, grid_idx]
            )
            correct_forcing_grid.position_field[:, grid_idx] = (
                element_pos + correct_forcing_grid.moment_arm[:, grid_idx]
            )

    np.testing.assert_allclose(
        rod_forcing_grid.moment_arm,
        correct_forcing_grid.moment_arm,
        atol=get_test_tol(precision="double"),
    )
    np.testing.assert_allclose(
        rod_forcing_grid.position_field, correct_forcing_grid.position_field
    )

    # Compute the correct grid velocity
    for i in range(n_elems):
        element_velocity = (
            straight_rod.velocity_collection[:, i] * straight_rod.mass[i]
            + straight_rod.velocity_collection[:, i + 1] * straight_rod.mass[i + 1]
        ) / (straight_rod.mass[i] + straight_rod.mass[i + 1])
        director_transpose = straight_rod.director_collection[:, :, i].T
        omega_in_lab_frame = director_transpose @ straight_rod.omega_collection[:, i]

        for j in range(correct_forcing_grid.surface_grid_points[i]):
            grid_idx = correct_forcing_grid.start_idx[i] + j
            correct_forcing_grid.velocity_field[
                :, grid_idx
            ] = element_velocity + np.cross(
                omega_in_lab_frame, correct_forcing_grid.moment_arm[:, grid_idx]
            )

    np.testing.assert_allclose(
        rod_forcing_grid.velocity_field, correct_forcing_grid.velocity_field
    )


@pytest.mark.parametrize("n_elems", [8, 16])
@pytest.mark.parametrize("largest_element_grid_density", [16, 12, 8, 4])
@pytest.mark.parametrize("taper_ratio", [1, 2, 5, 10])
@pytest.mark.parametrize("with_cap", [True, False])
def test_rod_surface_grid_force_transfer(
    n_elems, largest_element_grid_density, taper_ratio, with_cap
):
    grid_dim = 3
    base_radius = np.linspace(1, 1 / taper_ratio, n_elems)
    straight_rod = mock_straight_rod(n_elems, base_radius=base_radius)

    rod_forcing_grid = sps.CosseratRodSurfaceForcingGrid(
        grid_dim=grid_dim,
        cosserat_rod=straight_rod,
        surface_grid_density_for_largest_element=largest_element_grid_density,
        with_cap=with_cap,
    )
    correct_forcing_grid = MockSurfaceForcingGrid(
        n_elems, largest_element_grid_density, base_radius, with_cap
    )

    body_flow_forces = np.zeros((grid_dim, n_elems + 1))
    body_flow_torques = np.zeros((grid_dim, n_elems))
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
    for i in range(n_elems):
        correct_body_flow_forces[:, i] -= (
            0.5 * uniform_forcing[:, 0] * correct_forcing_grid.surface_grid_points[i]
        )
        correct_body_flow_forces[:, i + 1] -= (
            0.5 * uniform_forcing[:, 0] * correct_forcing_grid.surface_grid_points[i]
        )
    np.testing.assert_allclose(body_flow_forces, correct_body_flow_forces)

    # torques stay 0 for this loading
    np.testing.assert_allclose(
        body_flow_torques, 0.0, atol=get_test_tol(precision="double")
    )


@pytest.mark.parametrize("n_elems", [8, 16])
@pytest.mark.parametrize("largest_element_grid_density", [16, 12, 8, 4])
@pytest.mark.parametrize("taper_ratio", [1, 2, 5, 10])
def test_rod_surface_grid_spacing(n_elems, largest_element_grid_density, taper_ratio):
    base_radius = np.linspace(1, 1 / taper_ratio, n_elems)
    straight_rod = mock_straight_rod(n_elems, base_radius=base_radius)

    rod_forcing_grid = sps.CosseratRodSurfaceForcingGrid(
        grid_dim=3,
        cosserat_rod=straight_rod,
        surface_grid_density_for_largest_element=largest_element_grid_density,
    )

    max_grid_spacing = rod_forcing_grid.get_maximum_lagrangian_grid_spacing()
    # rod with same element sizes so max is one of any lengths
    correct_max_grid_spacing = max(
        straight_rod.lengths[0],
        np.max(base_radius) * (2 * np.pi / largest_element_grid_density),
    )

    np.testing.assert_allclose(correct_max_grid_spacing, max_grid_spacing)

import elastica as ea
import numpy as np
import pytest
import sopht_simulator as sps
from sopht.utils.precision import get_test_tol


def mock_2d_cylinder():
    """Returns a mock 2D cylinder (from elastica) for testing"""
    cyl_radius = 0.1
    X_cm = 1.0
    Y_cm = 2.0
    start = np.array([X_cm, Y_cm, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    density = 1e3
    cylinder = ea.Cylinder(start, direction, normal, base_length, cyl_radius, density)
    cylinder.velocity_collection[...] = 3.0
    cylinder.omega_collection[...] = 4.0
    return cylinder


@pytest.mark.parametrize("grid_dim", [2, 3])
def test_circular_cylinder_grid_invalid_dim(grid_dim):
    num_forcing_points = 8
    cylinder = mock_2d_cylinder()
    if grid_dim != 2:
        with pytest.raises(ValueError) as exc_info:
            _ = sps.CircularCylinderForcingGrid(
                grid_dim=grid_dim,
                rigid_body=cylinder,
                num_forcing_points=num_forcing_points,
            )
        error_msg = (
            "Invalid grid dimensions. 2D cylinder forcing grid is only "
            "defined for grid_dim=2"
        )
        assert exc_info.value.args[0] == error_msg


@pytest.mark.parametrize("num_forcing_points", [8, 16])
def test_circular_cylinder_grid_kinematics(num_forcing_points):
    cylinder = mock_2d_cylinder()
    grid_dim = 2
    circ_cyl_forcing_grid = sps.CircularCylinderForcingGrid(
        grid_dim=grid_dim,
        rigid_body=cylinder,
        num_forcing_points=num_forcing_points,
    )
    assert circ_cyl_forcing_grid.cylinder is cylinder
    assert circ_cyl_forcing_grid.position_field.shape == (grid_dim, num_forcing_points)
    assert circ_cyl_forcing_grid.velocity_field.shape == (grid_dim, num_forcing_points)

    # check if grid position consistent
    cylinder_com = cylinder.position_collection[:grid_dim]
    moment_arm = circ_cyl_forcing_grid.position_field - cylinder_com
    grid_distance_from_center = np.linalg.norm(moment_arm, axis=0)
    # check if all points are at distance = radius from center
    np.testing.assert_allclose(grid_distance_from_center, cylinder.radius)

    # check if location of points on circumference is correct
    d_theta = 2 * np.pi / num_forcing_points
    correct_angular_grid = np.linspace(
        d_theta / 2.0, 2 * np.pi - d_theta / 2.0, num_forcing_points
    )
    x_axis = 0
    y_axis = 1
    sine_of_angular_grid = (
        circ_cyl_forcing_grid.position_field[y_axis] - cylinder_com[y_axis]
    )
    cos_of_angular_grid = (
        circ_cyl_forcing_grid.position_field[x_axis] - cylinder_com[x_axis]
    )
    # since arctan2 gives values in range (-pi, pi)
    test_angular_grid = (
        np.arctan2(sine_of_angular_grid, cos_of_angular_grid) + 2 * np.pi
    ) % (2 * np.pi)
    np.testing.assert_allclose(test_angular_grid, correct_angular_grid)

    # check if velocities are correct
    correct_velocity_field = np.zeros_like(circ_cyl_forcing_grid.velocity_field)
    cylinder_com_velocity = cylinder.velocity_collection[:grid_dim, 0]
    z_axis = 2
    # vel = v_com + omega cross r
    correct_velocity_field[x_axis] = (
        cylinder_com_velocity[x_axis]
        - moment_arm[y_axis] * cylinder.omega_collection[z_axis, 0]
    )
    correct_velocity_field[y_axis] = (
        cylinder_com_velocity[y_axis]
        + moment_arm[x_axis] * cylinder.omega_collection[z_axis, 0]
    )
    np.testing.assert_allclose(
        correct_velocity_field, circ_cyl_forcing_grid.velocity_field
    )


@pytest.mark.parametrize("num_forcing_points", [8, 16])
def test_circular_cylinder_grid_force_transfer(num_forcing_points):
    cylinder = mock_2d_cylinder()
    grid_dim = 2
    circ_cyl_forcing_grid = sps.CircularCylinderForcingGrid(
        grid_dim=grid_dim,
        rigid_body=cylinder,
        num_forcing_points=num_forcing_points,
    )
    cyl_dim = 3
    body_flow_forces = np.zeros((cyl_dim, 1))
    body_flow_torques = np.zeros_like(body_flow_forces)
    lag_grid_forcing_field = np.zeros((grid_dim, num_forcing_points))
    x_axis = 0
    y_axis = 1
    uniform_forcing = (2.0, 3.0)
    lag_grid_forcing_field[x_axis] = uniform_forcing[x_axis]
    lag_grid_forcing_field[y_axis] = uniform_forcing[y_axis]
    circ_cyl_forcing_grid.transfer_forcing_from_grid_to_body(
        body_flow_forces=body_flow_forces,
        body_flow_torques=body_flow_torques,
        lag_grid_forcing_field=lag_grid_forcing_field,
    )
    correct_body_flow_forces = np.zeros_like(body_flow_forces)
    # negative sum
    correct_body_flow_forces[x_axis] = -uniform_forcing[x_axis] * num_forcing_points
    correct_body_flow_forces[y_axis] = -uniform_forcing[y_axis] * num_forcing_points
    np.testing.assert_allclose(body_flow_forces, correct_body_flow_forces)

    cylinder_com = cylinder.position_collection[:grid_dim]
    moment_arm = circ_cyl_forcing_grid.position_field - cylinder_com
    z_axis = 2
    correct_body_flow_torques = np.zeros_like(correct_body_flow_forces)
    correct_body_flow_torques[z_axis] = -np.sum(
        moment_arm[x_axis] * uniform_forcing[y_axis]
        - moment_arm[y_axis] * uniform_forcing[x_axis]
    )
    np.testing.assert_allclose(
        body_flow_torques,
        correct_body_flow_torques,
        atol=get_test_tol(precision="double"),
    )


@pytest.mark.parametrize("num_forcing_points", [8, 16])
def test_circular_cylinder_grid_spacing(num_forcing_points):
    cylinder = mock_2d_cylinder()
    grid_dim = 2
    circ_cyl_forcing_grid = sps.CircularCylinderForcingGrid(
        grid_dim=grid_dim,
        rigid_body=cylinder,
        num_forcing_points=num_forcing_points,
    )
    max_grid_spacing = circ_cyl_forcing_grid.get_maximum_lagrangian_grid_spacing()
    cylinder_circumference = 2 * np.pi * cylinder.radius
    correct_max_grid_spacing = cylinder_circumference / num_forcing_points
    assert correct_max_grid_spacing == max_grid_spacing

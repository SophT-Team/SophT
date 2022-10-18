import elastica as ea
import numpy as np
import pytest
import sopht_simulator as sps
from sopht.utils.precision import get_test_tol
from tests.immersed_body.rigid_body.test_derived_rigid_bodies import mock_xy_plane


def mock_2d_cylinder():
    """Returns a mock 2D cylinder (from elastica) for testing"""
    cyl_radius = 0.1
    start = np.array([1.0, 2.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    base_length = 1.0
    cylinder = ea.Cylinder(
        start, direction, normal, base_length, cyl_radius, density=1e3
    )
    cylinder.velocity_collection[...] = 3.0
    cylinder.omega_collection[...] = 4.0
    return cylinder


@pytest.mark.parametrize("grid_dim", [2, 3])
def test_circular_cylinder_grid_invalid_dim(grid_dim):
    cylinder = mock_2d_cylinder()
    if grid_dim != 2:
        with pytest.raises(ValueError) as exc_info:
            _ = sps.CircularCylinderForcingGrid(
                grid_dim=grid_dim,
                rigid_body=cylinder,
                num_forcing_points=8,
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


def mock_3d_sphere():
    """Returns a mock 3D sphere (from elastica) for testing"""
    sphere_radius = 0.1
    sphere_com = np.array([1.0, 2.0, 3.0])
    sphere = ea.Sphere(center=sphere_com, base_radius=sphere_radius, density=1e3)
    sphere.velocity_collection[...] = 3.0
    sphere.omega_collection[...] = 4.0
    return sphere


@pytest.mark.parametrize("grid_dim", [2, 3])
def test_sphere_grid_invalid_dim(grid_dim):
    sphere = mock_3d_sphere()
    if grid_dim != 3:
        with pytest.raises(ValueError) as exc_info:
            _ = sps.SphereForcingGrid(
                grid_dim=grid_dim,
                rigid_body=sphere,
                num_forcing_points_along_equator=8,
            )
        error_msg = (
            "Invalid grid dimensions. 3D Rigid body forcing grid is only "
            "defined for grid_dim=3"
        )
        assert exc_info.value.args[0] == error_msg


@pytest.mark.parametrize("num_forcing_points_along_equator", [8, 16])
def test_sphere_grid_kinematics(num_forcing_points_along_equator):
    sphere = mock_3d_sphere()
    grid_dim = 3
    sphere_forcing_grid = sps.SphereForcingGrid(
        grid_dim=grid_dim,
        rigid_body=sphere,
        num_forcing_points_along_equator=num_forcing_points_along_equator,
    )
    assert sphere_forcing_grid.rigid_body is sphere
    # number of forcing grid points is dynamic so cant test here
    assert sphere_forcing_grid.position_field.shape[0] == grid_dim
    assert sphere_forcing_grid.velocity_field.shape[0] == grid_dim
    polar_angle_grid = np.linspace(0, np.pi, num_forcing_points_along_equator // 2)
    num_forcing_points_along_latitudes = (
        np.rint(num_forcing_points_along_equator * np.sin(polar_angle_grid)).astype(int)
        + 1
    )
    assert sphere_forcing_grid.num_lag_nodes == sum(num_forcing_points_along_latitudes)

    # check if grid position consistent
    sphere_com = sphere.position_collection
    moment_arm = sphere_forcing_grid.position_field - sphere_com
    grid_distance_from_center = np.linalg.norm(moment_arm, axis=0)
    # check if all points are at distance = radius from center
    np.testing.assert_allclose(grid_distance_from_center, sphere.radius)
    grid_centroid = np.mean(sphere_forcing_grid.position_field, axis=1)
    np.testing.assert_allclose(grid_centroid, sphere_com[..., 0])

    # check if angular locations are correct
    x_axis = 0
    y_axis = 1
    z_axis = 2
    num_lag_nodes_idx = 0
    test_tol = get_test_tol(precision="double")
    for num_forcing_points_along_latitude, polar_angle in zip(
        num_forcing_points_along_latitudes, polar_angle_grid
    ):
        azimuthal_angle_grid = np.linspace(
            0.0, 2 * np.pi, num_forcing_points_along_latitude, endpoint=False
        )
        np.testing.assert_allclose(
            moment_arm[
                x_axis,
                num_lag_nodes_idx : num_lag_nodes_idx
                + num_forcing_points_along_latitude,
            ],
            sphere.radius * np.sin(polar_angle) * np.cos(azimuthal_angle_grid),
            atol=test_tol,
        )
        np.testing.assert_allclose(
            moment_arm[
                y_axis,
                num_lag_nodes_idx : num_lag_nodes_idx
                + num_forcing_points_along_latitude,
            ],
            sphere.radius * np.sin(polar_angle) * np.sin(azimuthal_angle_grid),
            atol=test_tol,
        )
        np.testing.assert_allclose(
            moment_arm[
                z_axis,
                num_lag_nodes_idx : num_lag_nodes_idx
                + num_forcing_points_along_latitude,
            ],
            sphere.radius * np.cos(polar_angle),
            atol=test_tol,
        )
        num_lag_nodes_idx += num_forcing_points_along_latitude

        # check if velocities are correct
        correct_velocity_field = np.zeros_like(sphere_forcing_grid.velocity_field)
        sphere_com_velocity = sphere.velocity_collection[..., 0]
        for axis in range(grid_dim):
            # vel = v_com + omega cross r
            omega_cross_r = (
                sphere.omega_collection[(axis + 1) % grid_dim]
                * moment_arm[(axis + 2) % grid_dim]
                - sphere.omega_collection[(axis + 2) % grid_dim]
                * moment_arm[(axis + 1) % grid_dim]
            )
            correct_velocity_field[axis] = sphere_com_velocity[x_axis] + omega_cross_r
        np.testing.assert_allclose(
            correct_velocity_field, sphere_forcing_grid.velocity_field
        )


@pytest.mark.parametrize("num_forcing_points_along_equator", [8, 16])
def test_sphere_grid_force_transfer(num_forcing_points_along_equator):
    sphere = mock_3d_sphere()
    grid_dim = 3
    sphere_forcing_grid = sps.SphereForcingGrid(
        grid_dim=grid_dim,
        rigid_body=sphere,
        num_forcing_points_along_equator=num_forcing_points_along_equator,
    )
    body_flow_forces = np.zeros((grid_dim, 1))
    body_flow_torques = np.zeros_like(body_flow_forces)
    lag_grid_forcing_field = np.zeros((grid_dim, sphere_forcing_grid.num_lag_nodes))
    uniform_forcing = np.random.rand(grid_dim, 1)
    lag_grid_forcing_field[...] = uniform_forcing
    sphere_forcing_grid.transfer_forcing_from_grid_to_body(
        body_flow_forces=body_flow_forces,
        body_flow_torques=body_flow_torques,
        lag_grid_forcing_field=lag_grid_forcing_field,
    )
    correct_body_flow_forces = np.zeros_like(body_flow_forces)
    # negative sum
    correct_body_flow_forces[...] = -uniform_forcing * sphere_forcing_grid.num_lag_nodes
    np.testing.assert_allclose(body_flow_forces, correct_body_flow_forces)

    sphere_com = sphere.position_collection
    moment_arm = sphere_forcing_grid.position_field - sphere_com
    correct_body_flow_torques = np.zeros_like(correct_body_flow_forces)
    for axis in range(grid_dim):
        correct_body_flow_torques[axis] = -np.sum(
            moment_arm[(axis + 1) % grid_dim] * uniform_forcing[(axis + 2) % grid_dim]
            - moment_arm[(axis + 2) % grid_dim] * uniform_forcing[(axis + 1) % grid_dim]
        )
    np.testing.assert_allclose(
        body_flow_torques,
        correct_body_flow_torques,
        atol=get_test_tol(precision="double"),
    )


@pytest.mark.parametrize("num_forcing_points_along_equator", [8, 16])
def test_sphere_grid_spacing(num_forcing_points_along_equator):
    sphere = mock_3d_sphere()
    grid_dim = 3
    sphere_forcing_grid = sps.SphereForcingGrid(
        grid_dim=grid_dim,
        rigid_body=sphere,
        num_forcing_points_along_equator=num_forcing_points_along_equator,
    )

    max_grid_spacing = sphere_forcing_grid.get_maximum_lagrangian_grid_spacing()
    sphere_equator_circumference = 2 * np.pi * sphere.radius
    correct_max_grid_spacing = (
        sphere_equator_circumference / num_forcing_points_along_equator
    )
    assert correct_max_grid_spacing == max_grid_spacing


@pytest.mark.parametrize("num_forcing_points_along_length", [8, 16])
def test_rectangular_plane_grid_kinematics(num_forcing_points_along_length):
    plane = mock_xy_plane()
    grid_dim = 3
    rect_plane_forcing_grid = sps.RectangularPlaneForcingGrid(
        grid_dim=grid_dim,
        rigid_body=plane,
        num_forcing_points_along_length=num_forcing_points_along_length,
    )
    assert rect_plane_forcing_grid.rigid_body is plane
    num_forcing_points_along_breadth = int(
        num_forcing_points_along_length * plane.breadth / plane.length
    )
    correct_num_forcing_lag_nodes = (
        num_forcing_points_along_length * num_forcing_points_along_breadth
    )
    assert rect_plane_forcing_grid.num_lag_nodes == correct_num_forcing_lag_nodes
    assert rect_plane_forcing_grid.position_field.shape == (
        grid_dim,
        correct_num_forcing_lag_nodes,
    )
    assert rect_plane_forcing_grid.velocity_field.shape == (
        grid_dim,
        correct_num_forcing_lag_nodes,
    )

    # check if positions are correct
    z_axis = 2
    np.testing.assert_allclose(
        rect_plane_forcing_grid.position_field[z_axis],
        plane.position_collection[z_axis, 0],
    )
    x_axis = 0
    x_axis_grid_range = np.linspace(
        plane.position_collection[x_axis, 0] - 0.5 * plane.length,
        plane.position_collection[x_axis, 0] + 0.5 * plane.length,
        num_forcing_points_along_length,
    )
    y_axis = 1
    y_axis_grid_range = np.linspace(
        plane.position_collection[y_axis, 0] - 0.5 * plane.breadth,
        plane.position_collection[y_axis, 0] + 0.5 * plane.breadth,
        num_forcing_points_along_breadth,
    )
    correct_x_axis_grid, correct_y_axis_grid = np.meshgrid(
        x_axis_grid_range, y_axis_grid_range
    )
    np.testing.assert_allclose(
        rect_plane_forcing_grid.position_field[x_axis],
        correct_x_axis_grid.reshape(
            -1,
        ),
    )
    np.testing.assert_allclose(
        rect_plane_forcing_grid.position_field[y_axis],
        correct_y_axis_grid.reshape(
            -1,
        ),
    )

    # velocity computation is tested in parent 3D rigid body class
    # hence leaving only a check against 0 here
    np.testing.assert_allclose(rect_plane_forcing_grid.velocity_field, 0.0)


@pytest.mark.parametrize("num_forcing_points_along_length", [8, 16])
def test_rectangular_plane_grid_spacing(num_forcing_points_along_length):
    plane = mock_xy_plane()
    rect_plane_forcing_grid = sps.RectangularPlaneForcingGrid(
        grid_dim=3,
        rigid_body=plane,
        num_forcing_points_along_length=num_forcing_points_along_length,
    )
    max_grid_spacing = rect_plane_forcing_grid.get_maximum_lagrangian_grid_spacing()
    correct_max_grid_spacing = plane.length / num_forcing_points_along_length
    assert correct_max_grid_spacing == max_grid_spacing

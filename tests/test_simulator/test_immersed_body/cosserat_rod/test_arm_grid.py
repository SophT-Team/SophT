import numpy as np
import pytest
import elastica as ea
import sopht.simulator as sps
from sopht.utils.precision import get_test_tol


def mock_arm_straight_rods(n_elems, **kwargs):
    rod_list = []
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    rod_length = 5
    taper_ratio = kwargs.get("taper_ratio", 1)
    base_radius = kwargs.get("base_radius", np.linspace(1, 1 / taper_ratio, n_elems))
    straight_rod = ea.CosseratRod.straight_rod(
        n_elems,
        start,
        direction,
        normal,
        rod_length,
        base_radius,
        density=1e3,
        youngs_modulus=1e6,
        shear_modulus=1e6 / (0.5 + 1.0),
    )
    n_nodes = n_elems + 1
    straight_rod.velocity_collection[...] = np.linspace(1, n_nodes, n_nodes)
    straight_rod.omega_collection[...] = np.linspace(1, n_elems, n_elems)
    offset = straight_rod.position_collection[1, 1] / 2
    rod_list.append(straight_rod)

    radius_outer_rods = base_radius[0] / (2 / (np.sqrt(2 - np.sqrt(2))) - 1)
    angles = np.linspace(0, 2 * np.pi - np.pi / 4, 8)
    n_elems_outer = n_elems * 2 - 1
    radius_outer_rods = np.linspace(
        radius_outer_rods, radius_outer_rods / taper_ratio, n_elems_outer
    )
    radius_tot = radius_outer_rods[0] + base_radius[0]

    for i in range(8):
        start = np.array(
            [
                0.0 + radius_tot * np.cos(angles[i]),
                0.0 + offset,
                0.0 + radius_tot * np.sin(angles[i]),
            ]
        )
        outer_rod = ea.CosseratRod.straight_rod(
            n_elems_outer,
            start,
            direction,
            normal,
            rod_length - offset,
            radius_outer_rods,
            density=1e3,
            youngs_modulus=1e6,
            shear_modulus=1e6 / (0.5 + 1.0),
        )
        n_nodes = n_elems_outer + 1
        outer_rod.velocity_collection[...] = np.linspace(1, n_nodes, n_nodes)
        outer_rod.omega_collection[...] = np.linspace(1, n_elems_outer, n_elems_outer)
        rod_list.append(outer_rod)

    return rod_list


# Surface Forcing Grid tests
class MockArmSurfaceForcingGrid:
    def __init__(self, n_elems, grid_density, base_radius, with_cap):

        self.surface_grid_density_for_largest_element = grid_density
        self.surface_grid_points = np.zeros(n_elems, dtype=int)
        self.surface_point_rotation_angle_list = []
        self.grid_point_radius_ratio = []
        self.start_idx = np.zeros(n_elems, dtype=int)
        self.end_idx = np.zeros(n_elems, dtype=int)

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

                    for num_points in end_elem_surface_grid_points:
                        self.surface_point_rotation_angle_list[i] = np.append(
                            self.surface_point_rotation_angle_list[i],
                            ([np.linspace(0, 2 * np.pi, num_points, endpoint=False)]),
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

                    sorted_indices = np.argsort(
                        self.surface_point_rotation_angle_list[i], kind="stable"
                    )
                    self.surface_point_rotation_angle_list[
                        i
                    ] = self.surface_point_rotation_angle_list[i][sorted_indices]
                    self.grid_point_radius_ratio[i] = self.grid_point_radius_ratio[i][
                        sorted_indices
                    ]

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
                    2, self.start_idx[i] : self.end_idx[i]
                ] = np.sin(angle)


class MockBlock:
    def __init__(self, grid_dim, num_points_pos, num_points_rad):
        self.position_collection = np.zeros((grid_dim, num_points_pos))
        self.radius = np.zeros((1, num_points_rad))


@pytest.mark.parametrize("grid_dim", [0, 1, 2, 4])
@pytest.mark.parametrize("n_elems", [8, 16])
def test_rod_surface_grid_dimension(grid_dim, n_elems):
    rod_list = mock_arm_straight_rods(n_elems)
    rod_numbers = [0, 0, 8]
    surface_grid_density_for_largest_element = 8
    memory_block = []

    with pytest.raises(ValueError) as exc_info:
        _ = sps.OctopusArmSurfaceForcingGrid(
            memory_block_list=memory_block,
            grid_dim=grid_dim,
            rod_list=rod_list,
            rod_numbers=rod_numbers,
            surface_grid_density_for_largest_element=surface_grid_density_for_largest_element,
        )
    error_msg = (
        "Invalid grid dimensions. Octopus arm surface forcing grid is only "
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
    rod_numbers = [0, 0, 8]
    base_radius = np.linspace(1, 1 / taper_ratio, n_elems)
    rod_list = mock_arm_straight_rods(
        n_elems, base_radius=base_radius, taper_ratio=taper_ratio
    )
    num_of_rods = len(rod_list)
    num_points_pos = n_elems + 1 + (n_elems * 2) * (num_of_rods - 1) + num_of_rods - 1
    num_points_rad = (
        n_elems + (n_elems * 2 - 1) * (num_of_rods - 1) + (num_of_rods - 1) * 2
    )
    block_temp = MockBlock(
        grid_dim=3, num_points_pos=num_points_pos, num_points_rad=num_points_rad
    )
    memory_block = [[block_temp]]
    memory_block[0][0].position_collection[:, : n_elems + 1] = rod_list[
        0
    ].position_collection
    memory_block[0][0].radius[:, :n_elems] = rod_list[0].radius
    temp_idx = n_elems + 2

    for i in range(num_of_rods - 1):
        memory_block[0][0].position_collection[
            :, temp_idx : temp_idx + 2 * n_elems
        ] = rod_list[1 + i].position_collection
        memory_block[0][0].radius[:, temp_idx : temp_idx + 2 * n_elems - 1] = rod_list[
            i + 1
        ].radius
        temp_idx += 2 + 2 * n_elems - 1

    rod_forcing_grid = sps.OctopusArmSurfaceForcingGrid(
        grid_dim=3,
        memory_block_list=memory_block,
        rod_list=rod_list,
        rod_numbers=rod_numbers,
        surface_grid_density_for_largest_element=largest_element_grid_density,
        with_cap=with_cap,
    )

    correct_forcing_grid = MockArmSurfaceForcingGrid(
        n_elems,
        largest_element_grid_density,
        base_radius,
        with_cap=with_cap,
    )

    # check if setup is correct
    assert rod_forcing_grid.rod_list is rod_list
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
        if len(rod_forcing_grid.surface_point_rotation_angle_list[i]) == 0:
            np.testing.assert_allclose(
                rod_forcing_grid.surface_point_rotation_angle_list[i],
                correct_forcing_grid.surface_point_rotation_angle_list[i],
                atol=get_test_tol(precision="double"),
            )
        else:
            np.testing.assert_allclose(
                rod_forcing_grid.surface_point_rotation_angle_list[i][0],
                correct_forcing_grid.surface_point_rotation_angle_list[i],
                atol=get_test_tol(precision="double"),
            )

    np.testing.assert_allclose(
        rod_forcing_grid.start_idx, correct_forcing_grid.start_idx
    )
    np.testing.assert_allclose(rod_forcing_grid.end_idx, correct_forcing_grid.end_idx)


@pytest.mark.parametrize("n_elems", [8, 16])
@pytest.mark.parametrize("largest_element_grid_density", [16, 12, 8, 4])
@pytest.mark.parametrize("taper_ratio", [1, 2, 5, 10])
@pytest.mark.parametrize("with_cap", [True, False])
def test_rod_surface_grid_grid_kinematics(
    n_elems, largest_element_grid_density, taper_ratio, with_cap
):
    base_radius = np.linspace(1, 1 / taper_ratio, n_elems)
    rod_list = mock_arm_straight_rods(
        n_elems, base_radius=base_radius, taper_ratio=taper_ratio
    )
    rod_numbers = [0, 0, 8]
    num_of_rods = len(rod_list)
    num_points_pos = n_elems + 1 + (n_elems * 2) * (num_of_rods - 1) + num_of_rods - 1
    num_points_rad = (
        n_elems + (n_elems * 2 - 1) * (num_of_rods - 1) + (num_of_rods - 1) * 2
    )
    block_temp = MockBlock(
        grid_dim=3, num_points_pos=num_points_pos, num_points_rad=num_points_rad
    )
    memory_block = [[block_temp]]
    memory_block[0][0].position_collection[:, : n_elems + 1] = rod_list[
        0
    ].position_collection
    memory_block[0][0].radius[:, :n_elems] = rod_list[0].radius
    temp_idx = n_elems + 2

    for i in range(num_of_rods - 1):
        memory_block[0][0].position_collection[
            :, temp_idx : temp_idx + 2 * n_elems
        ] = rod_list[1 + i].position_collection
        memory_block[0][0].radius[:, temp_idx : temp_idx + 2 * n_elems - 1] = rod_list[
            i + 1
        ].radius
        temp_idx += 2 + 2 * n_elems - 1

    rod_forcing_grid = sps.OctopusArmSurfaceForcingGrid(
        grid_dim=3,
        memory_block_list=memory_block,
        rod_list=rod_list,
        rod_numbers=rod_numbers,
        surface_grid_density_for_largest_element=largest_element_grid_density,
        with_cap=with_cap,
    )
    correct_forcing_grid = MockArmSurfaceForcingGrid(
        n_elems,
        largest_element_grid_density,
        base_radius,
        with_cap=with_cap,
    )

    # Compute the correct grid position
    for i in range(n_elems):
        element_pos = 0.5 * (
            rod_list[0].position_collection[:, i]
            + rod_list[0].position_collection[:, i + 1]
        )
        director_transpose = rod_list[0].director_collection[:, :, i].T
        rod_radius = (
            rod_list[0].radius[0]
            + rod_list[1].radius[[2 * i]]
            + rod_list[1].radius[[0]]
        )
        rod_radius_ratio = correct_forcing_grid.grid_point_radius_ratio[i]

        for j in range(correct_forcing_grid.surface_grid_points[i]):
            grid_idx = correct_forcing_grid.start_idx[i] + j
            correct_forcing_grid.moment_arm[:, grid_idx] = (
                rod_radius
                * rod_radius_ratio[j]
                * correct_forcing_grid.local_frame_surface_points[:, grid_idx]
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
        rod_forcing_grid.position_field,
        correct_forcing_grid.position_field,
        atol=get_test_tol(precision="double"),
    )

    # Compute the correct grid velocity
    for i in range(n_elems):
        element_velocity = (
            rod_list[0].velocity_collection[:, i] * rod_list[0].mass[i]
            + rod_list[0].velocity_collection[:, i + 1] * rod_list[0].mass[i + 1]
        ) / (rod_list[0].mass[i] + rod_list[0].mass[i + 1])
        director_transpose = rod_list[0].director_collection[:, :, i].T
        omega_in_lab_frame = director_transpose @ rod_list[0].omega_collection[:, i]

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
    rod_list = mock_arm_straight_rods(
        n_elems, base_radius=base_radius, taper_ratio=taper_ratio
    )
    rod_numbers = [0, 0, 8]
    num_of_rods = len(rod_list)
    num_points_pos = n_elems + 1 + (n_elems * 2) * (num_of_rods - 1) + num_of_rods - 1
    num_points_rad = (
        n_elems + (n_elems * 2 - 1) * (num_of_rods - 1) + (num_of_rods - 1) * 2
    )
    block_temp = MockBlock(
        grid_dim=grid_dim, num_points_pos=num_points_pos, num_points_rad=num_points_rad
    )
    memory_block = [[block_temp]]
    memory_block[0][0].position_collection[:, : n_elems + 1] = rod_list[
        0
    ].position_collection
    memory_block[0][0].radius[:, :n_elems] = rod_list[0].radius
    temp_idx = n_elems + 2

    for i in range(num_of_rods - 1):
        memory_block[0][0].position_collection[
            :, temp_idx : temp_idx + 2 * n_elems
        ] = rod_list[1 + i].position_collection
        memory_block[0][0].radius[:, temp_idx : temp_idx + 2 * n_elems - 1] = rod_list[
            i + 1
        ].radius
        temp_idx += 2 + 2 * n_elems - 1

    rod_forcing_grid = sps.OctopusArmSurfaceForcingGrid(
        grid_dim=grid_dim,
        memory_block_list=memory_block,
        rod_list=rod_list,
        rod_numbers=rod_numbers,
        surface_grid_density_for_largest_element=largest_element_grid_density,
        with_cap=with_cap,
    )
    correct_forcing_grid = MockArmSurfaceForcingGrid(
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
    rod_list = mock_arm_straight_rods(
        n_elems, base_radius=base_radius, taper_ratio=taper_ratio
    )
    rod_numbers = [0, 0, 8]
    num_of_rods = len(rod_list)
    num_points_pos = n_elems + 1 + (n_elems * 2) * (num_of_rods - 1) + num_of_rods - 1
    num_points_rad = (
        n_elems + (n_elems * 2 - 1) * (num_of_rods - 1) + (num_of_rods - 1) * 2
    )
    block_temp = MockBlock(
        grid_dim=3, num_points_pos=num_points_pos, num_points_rad=num_points_rad
    )
    memory_block = [[block_temp]]
    memory_block[0][0].position_collection[:, : n_elems + 1] = rod_list[
        0
    ].position_collection
    memory_block[0][0].radius[:, :n_elems] = rod_list[0].radius
    temp_idx = n_elems + 2

    for i in range(num_of_rods - 1):
        memory_block[0][0].position_collection[
            :, temp_idx : temp_idx + 2 * n_elems
        ] = rod_list[1 + i].position_collection
        memory_block[0][0].radius[:, temp_idx : temp_idx + 2 * n_elems - 1] = rod_list[
            i + 1
        ].radius
        temp_idx += 2 + 2 * n_elems - 1

    rod_forcing_grid = sps.OctopusArmSurfaceForcingGrid(
        grid_dim=3,
        memory_block_list=memory_block,
        rod_list=rod_list,
        rod_numbers=rod_numbers,
        surface_grid_density_for_largest_element=largest_element_grid_density,
    )

    radius = base_radius[0] + 2 * rod_list[1].radius[0]
    max_grid_spacing = rod_forcing_grid.get_maximum_lagrangian_grid_spacing()
    # rod with same element sizes so max is one of any lengths
    correct_max_grid_spacing = max(
        rod_list[0].lengths[0],
        radius * (2 * np.pi / largest_element_grid_density),
    )

    np.testing.assert_allclose(correct_max_grid_spacing, max_grid_spacing)

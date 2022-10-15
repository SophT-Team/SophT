from elastica import Cylinder, RigidBodyBase, Sphere
from elastica._linalg import _batch_cross
import numpy as np
from sopht_simulator.immersed_body import ImmersedBodyForcingGrid


class TwoDimensionalCylinderForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid of a 2D cylinder with cross-section
    in XY plane.

    """

    def __init__(self, grid_dim: int, num_lag_nodes: int, rigid_body: type(Cylinder)):
        if grid_dim != 2:
            raise ValueError(
                "Invalid grid dimensions. 2D cylinder forcing grid is only "
                "defined for grid_dim=2"
            )
        self.cylinder = rigid_body
        super().__init__(grid_dim, num_lag_nodes)
        self.local_frame_relative_position_field = np.zeros_like(self.position_field)
        self.global_frame_relative_position_field = np.zeros_like(self.position_field)

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the cylinder boundary"""
        self.global_frame_relative_position_field[...] = np.dot(
            self.cylinder.director_collection[: self.grid_dim, : self.grid_dim, 0].T,
            self.local_frame_relative_position_field,
        )
        self.position_field[...] = (
            self.cylinder.position_collection[: self.grid_dim]
            + self.global_frame_relative_position_field
        )

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the cylinder boundary"""
        # d3 aligned along Z while d1 and d2 along XY plane...
        # Can be shown that omega local and global lie along d3 (Z axis)
        global_frame_omega_z = (
            self.cylinder.director_collection[self.grid_dim, self.grid_dim, 0]
            * self.cylinder.omega_collection[self.grid_dim, 0]
        )
        self.velocity_field[0] = (
            self.cylinder.velocity_collection[0]
            - global_frame_omega_z * self.global_frame_relative_position_field[1]
        )
        self.velocity_field[1] = (
            self.cylinder.velocity_collection[1]
            + global_frame_omega_z * self.global_frame_relative_position_field[0]
        )

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cylinder"""
        # negative sign due to Newtons third law
        body_flow_forces[: self.grid_dim] = -np.sum(
            lag_grid_forcing_field, axis=1
        ).reshape(-1, 1)

        # torque from grid forcing
        # Q @ (0, 0, torque) = d3 dot (0, 0, torque) = Q[2, 2] * (0, 0, torque)
        body_flow_torques[self.grid_dim] = self.cylinder.director_collection[
            self.grid_dim, self.grid_dim, 0
        ] * np.sum(
            -self.global_frame_relative_position_field[0] * lag_grid_forcing_field[1]
            + self.global_frame_relative_position_field[1] * lag_grid_forcing_field[0]
        )

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""


class CircularCylinderForcingGrid(TwoDimensionalCylinderForcingGrid):
    """Class for forcing grid of a 2D circular cylinder with cross-section
    in XY plane.

    """

    def __init__(
        self, grid_dim: int, rigid_body: type(Cylinder), num_forcing_points: int
    ):
        super().__init__(
            grid_dim=grid_dim, num_lag_nodes=num_forcing_points, rigid_body=rigid_body
        )

        dtheta = 2.0 * np.pi / self.num_lag_nodes
        theta = np.linspace(
            0 + dtheta / 2.0, 2.0 * np.pi - dtheta / 2.0, self.num_lag_nodes
        )
        self.local_frame_relative_position_field[0, :] = self.cylinder.radius * np.cos(
            theta
        )
        self.local_frame_relative_position_field[1, :] = self.cylinder.radius * np.sin(
            theta
        )

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""
        # ds = radius * dtheta
        return self.cylinder.radius * (2.0 * np.pi / self.num_lag_nodes)


class SquareCylinderForcingGrid(TwoDimensionalCylinderForcingGrid):
    """Class for forcing grid of a 2D square shaped cylinder with cross-section
    in XY plane.

    """

    def __init__(
        self, grid_dim: int, rigid_body: type(Cylinder), num_forcing_points: int
    ):
        super().__init__(
            grid_dim=grid_dim, num_lag_nodes=num_forcing_points, rigid_body=rigid_body
        )

        if self.num_lag_nodes % 4 != 0:
            raise ValueError(
                "Provide number of forcing nodes as a integer multiple of 4!"
            )
        num_lag_nodes_per_side = self.num_lag_nodes // 4
        side_length = 2 * self.cylinder.radius
        ds = side_length / num_lag_nodes_per_side
        side_coordinates_range = np.linspace(
            -0.5 * side_length + 0.5 * ds,
            0.5 * side_length - 0.5 * ds,
            num_lag_nodes_per_side,
        )
        # top boundary
        self.local_frame_relative_position_field[
            0, :num_lag_nodes_per_side
        ] = side_coordinates_range
        self.local_frame_relative_position_field[1, :num_lag_nodes_per_side] = (
            0.5 * side_length
        )
        # right boundary
        self.local_frame_relative_position_field[
            0, num_lag_nodes_per_side : 2 * num_lag_nodes_per_side
        ] = (0.5 * side_length)
        self.local_frame_relative_position_field[
            1, num_lag_nodes_per_side : 2 * num_lag_nodes_per_side
        ] = side_coordinates_range
        # bottom boundary
        self.local_frame_relative_position_field[
            0, 2 * num_lag_nodes_per_side : 3 * num_lag_nodes_per_side
        ] = side_coordinates_range
        self.local_frame_relative_position_field[
            1, 2 * num_lag_nodes_per_side : 3 * num_lag_nodes_per_side
        ] = (-0.5 * side_length)
        # left boundary
        self.local_frame_relative_position_field[
            0, 3 * num_lag_nodes_per_side : 4 * num_lag_nodes_per_side
        ] = (-0.5 * side_length)
        self.local_frame_relative_position_field[
            1, 3 * num_lag_nodes_per_side : 4 * num_lag_nodes_per_side
        ] = side_coordinates_range

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""
        num_lag_nodes_per_side = self.num_lag_nodes // 4
        side_length = 2 * self.cylinder.radius
        return side_length / num_lag_nodes_per_side


class ThreeDimensionalRigidBodyForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid of a 3D rigid body with cross-section."""

    def __init__(
        self, grid_dim: int, num_lag_nodes: int, rigid_body: type(RigidBodyBase)
    ):
        if grid_dim != 3:
            raise ValueError(
                "Invalid grid dimensions. 3D Rigid body forcing grid is only "
                "defined for grid_dim=3"
            )
        self.rigid_body = rigid_body
        super().__init__(grid_dim, num_lag_nodes)
        self.local_frame_relative_position_field = np.zeros_like(self.position_field)
        self.global_frame_relative_position_field = np.zeros_like(self.position_field)

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the rigid body boundary"""
        self.global_frame_relative_position_field[...] = np.dot(
            self.rigid_body.director_collection[:, :, 0].T,
            self.local_frame_relative_position_field,
        )
        self.position_field[...] = (
            self.rigid_body.position_collection
            + self.global_frame_relative_position_field
        )

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the rigid body boundary"""
        global_frame_omega = np.dot(
            self.rigid_body.director_collection[:, :, 0].T,
            self.rigid_body.omega_collection,
        )
        self.velocity_field[...] = self.rigid_body.velocity_collection + _batch_cross(
            global_frame_omega * np.ones(self.num_lag_nodes),
            self.global_frame_relative_position_field,
        )

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the rigid body"""
        # negative sign due to Newtons third law
        body_flow_forces[...] = -np.sum(lag_grid_forcing_field, axis=1).reshape(-1, 1)

        # torque from grid forcing
        body_flow_torques[...] = -np.dot(
            self.rigid_body.director_collection[:, :, 0],
            np.sum(
                _batch_cross(
                    self.global_frame_relative_position_field, lag_grid_forcing_field
                ),
                axis=1,
            ).reshape(-1, 1),
        )

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""


class OpenEndCircularCylinderForcingGrid(ThreeDimensionalRigidBodyForcingGrid):
    """Class for forcing grid of a 3D circular cylinder with open ends i.e. no
    forcing at the base and top (more like a pipe)"""

    def __init__(
        self,
        grid_dim: int,
        rigid_body: type(Cylinder),
        num_forcing_points_along_length: int,
    ):
        self.num_forcing_points_along_length = num_forcing_points_along_length
        cylinder_circumference = 2 * np.pi * rigid_body.radius
        # keep same density of points along surface
        self.num_forcing_points_along_circumference = int(
            self.num_forcing_points_along_length
            * cylinder_circumference
            / rigid_body.length
        )
        if self.num_forcing_points_along_circumference == 0:
            raise RuntimeError(
                "Too thin cylinder! Use a line source forcing grid instead!"
            )
        num_lag_nodes = (
            self.num_forcing_points_along_length
            * self.num_forcing_points_along_circumference
        )
        super().__init__(
            grid_dim=grid_dim, num_lag_nodes=num_lag_nodes, rigid_body=rigid_body
        )

        dtheta = 2.0 * np.pi / self.num_forcing_points_along_circumference
        theta = np.linspace(
            0 + dtheta / 2.0,
            2.0 * np.pi - dtheta / 2.0,
            self.num_forcing_points_along_circumference,
        )
        length_grid = np.linspace(
            -0.5 * self.rigid_body.length,
            0.5 * self.rigid_body.length,
            self.num_forcing_points_along_length,
        )
        for idx in range(
            0, self.num_lag_nodes, self.num_forcing_points_along_circumference
        ):
            self.local_frame_relative_position_field[
                0, idx : idx + self.num_forcing_points_along_circumference
            ] = self.rigid_body.radius * np.cos(theta)
            self.local_frame_relative_position_field[
                1, idx : idx + self.num_forcing_points_along_circumference
            ] = self.rigid_body.radius * np.sin(theta)
            self.local_frame_relative_position_field[
                2, idx : idx + self.num_forcing_points_along_circumference
            ] = length_grid[idx // self.num_forcing_points_along_circumference]

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""
        # ds = radius * dtheta
        return max(
            self.rigid_body.radius
            * (2.0 * np.pi / self.num_forcing_points_along_circumference),
            self.rigid_body.length / self.num_forcing_points_along_length,
        )


class SphereForcingGrid(ThreeDimensionalRigidBodyForcingGrid):
    """Class for forcing grid of a 3D sphere"""

    def __init__(
        self,
        grid_dim: int,
        rigid_body: type(Sphere),
        num_forcing_points_along_equator: int,
    ):
        self.num_forcing_points_along_equator = num_forcing_points_along_equator
        polar_angle_grid = np.linspace(
            0, np.pi, self.num_forcing_points_along_equator // 2
        )
        num_forcing_points_along_latitudes = (
            np.rint(num_forcing_points_along_equator * np.sin(polar_angle_grid)).astype(
                int
            )
            + 1
        )
        num_lag_nodes = sum(num_forcing_points_along_latitudes)
        super().__init__(
            grid_dim=grid_dim, num_lag_nodes=num_lag_nodes, rigid_body=rigid_body
        )
        global_frame_relative_position_x = np.array([])
        global_frame_relative_position_y = np.array([])
        global_frame_relative_position_z = np.array([])
        for num_forcing_points_along_latitude, polar_angle in zip(
            num_forcing_points_along_latitudes, polar_angle_grid
        ):
            azimuthal_angle_grid = np.linspace(
                0.0, 2 * np.pi, num_forcing_points_along_latitude, endpoint=False
            )
            global_frame_relative_position_x = np.append(
                global_frame_relative_position_x,
                self.rigid_body.radius
                * np.sin(polar_angle)
                * np.cos(azimuthal_angle_grid),
            )
            global_frame_relative_position_y = np.append(
                global_frame_relative_position_y,
                self.rigid_body.radius
                * np.sin(polar_angle)
                * np.sin(azimuthal_angle_grid),
            )
            global_frame_relative_position_z = np.append(
                global_frame_relative_position_z,
                self.rigid_body.radius
                * np.cos(polar_angle)
                * np.ones(num_forcing_points_along_latitude),
            )
        self.global_frame_relative_position_field[0] = np.array(
            global_frame_relative_position_x
        )
        self.global_frame_relative_position_field[1] = np.array(
            global_frame_relative_position_y
        )
        self.global_frame_relative_position_field[2] = np.array(
            global_frame_relative_position_z
        )

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""
        # ds = radius * dtheta
        return self.rigid_body.radius * (
            2 * np.pi / self.num_forcing_points_along_equator
        )

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the rigid sphere.

        Since this is a sphere, the local frame concept is redundant,
        and the global relative fram can be directly used.
        Hence overloading the method here.
        """
        self.position_field[...] = (
            self.rigid_body.position_collection
            + self.global_frame_relative_position_field
        )

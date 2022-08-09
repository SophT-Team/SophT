from elastica import Cylinder

import numpy as np

from sopht_simulator.immersed_body import ImmersedBodyForcingGrid


class TwoDimensionalCylinderForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid of a 2D cylinder with cross-section
    in XY plane.

    """

    def __init__(self, grid_dim, num_forcing_points, cylinder: type(Cylinder)):
        self.num_lag_nodes = num_forcing_points
        self.cylinder = cylinder
        super().__init__(grid_dim)
        self.local_frame_position_field = np.zeros_like(self.position_field)

    def update_local_frame_lag_grid_position_field(self, time=0.0):
        """Update the local frame forcing grid positions"""

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the cylinder boundary"""
        self.position_field[...] = (
            self.cylinder.position_collection[: self.grid_dim]
            + self.local_frame_position_field
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
            - global_frame_omega_z * self.local_frame_position_field[1]
        )
        self.velocity_field[1] = (
            self.cylinder.velocity_collection[1]
            + global_frame_omega_z * self.local_frame_position_field[0]
        )

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cylinder"""
        # negative sign due to Newtons third law
        body_flow_forces[: self.grid_dim] = -np.sum(lag_grid_forcing_field)

        # torque from grid forcing
        # Q @ (0, 0, torque) = d3 dot (0, 0, torque) = Q[2, 2] * (0, 0, torque)
        body_flow_torques[self.grid_dim] = self.cylinder.director_collection[
            self.grid_dim, self.grid_dim, 0
        ] * np.sum(
            -self.local_frame_position_field[0] * lag_grid_forcing_field[1]
            + self.local_frame_position_field[1] * lag_grid_forcing_field[0]
        )


class CircularCylinderForcingGrid(TwoDimensionalCylinderForcingGrid):
    """Class for forcing grid of a 2D circular cylinder with cross-section
    in XY plane.

    """

    def __init__(self, grid_dim, num_forcing_points, cylinder: type(Cylinder)):
        super().__init__(grid_dim, num_forcing_points, cylinder)

        dtheta = 2.0 * np.pi / self.num_lag_nodes
        theta = np.linspace(
            0 + dtheta / 2.0, 2.0 * np.pi - dtheta / 2.0, self.num_lag_nodes
        )
        self.local_frame_position_field[0, :] = self.cylinder.radius * np.cos(theta)
        self.local_frame_position_field[1, :] = self.cylinder.radius * np.sin(theta)

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()


class SquareCylinderForcingGrid(TwoDimensionalCylinderForcingGrid):
    """Class for forcing grid of a 2D square shaped cylinder with cross-section
    in XY plane.

    """

    def __init__(self, grid_dim, num_forcing_points, cylinder: type(Cylinder)):
        super().__init__(grid_dim, num_forcing_points, cylinder)

        if self.num_lag_nodes % 4 != 0:
            raise ValueError(
                "Provide number of forcing nodes as a integer multiple of 4!"
            )
        num_lag_nodes_per_side = self.num_lag_nodes // 4
        side_length = 2 * self.cylinder.radius
        ds = 2 * side_length / num_lag_nodes_per_side
        side_coordinates_range = np.linspace(
            -0.5 * side_length + 0.5 * ds,
            0.5 * side_length - 0.5 * ds,
            num_lag_nodes_per_side,
        )
        # top boundary
        self.local_frame_position_field[
            0, :num_lag_nodes_per_side
        ] = side_coordinates_range
        self.local_frame_position_field[1, :num_lag_nodes_per_side] = 0.5 * side_length
        # right boundary
        self.local_frame_position_field[
            0, num_lag_nodes_per_side : 2 * num_lag_nodes_per_side
        ] = (0.5 * side_length)
        self.local_frame_position_field[
            1, num_lag_nodes_per_side : 2 * num_lag_nodes_per_side
        ] = side_coordinates_range
        # bottom boundary
        self.local_frame_position_field[
            0, 2 * num_lag_nodes_per_side : 3 * num_lag_nodes_per_side
        ] = side_coordinates_range
        self.local_frame_position_field[
            1, 2 * num_lag_nodes_per_side : 3 * num_lag_nodes_per_side
        ] = (-0.5 * side_length)
        # left boundary
        self.local_frame_position_field[
            0, 3 * num_lag_nodes_per_side : 4 * num_lag_nodes_per_side
        ] = (-0.5 * side_length)
        self.local_frame_position_field[
            1, 3 * num_lag_nodes_per_side : 4 * num_lag_nodes_per_side
        ] = side_coordinates_range

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

from elastica import Cylinder

import numpy as np

from sopht_simulator.immersed_body import ImmersedBodyForcingGrid


class CircularCylinderForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid of a 2D circular cylinder with cross-section
    in XY plane.

    """

    def __init__(self, grid_dim, num_forcing_points, cylinder: type(Cylinder)):
        self.num_lag_nodes = num_forcing_points
        self.cylinder = cylinder
        super().__init__(grid_dim)

        self.local_forcing_grid_position = np.zeros_like(self.position_field)
        dtheta = 2.0 * np.pi / self.num_lag_nodes
        theta = np.linspace(
            0 + dtheta / 2.0, 2.0 * np.pi - dtheta / 2.0, self.num_lag_nodes
        )
        self.local_forcing_grid_position[0, :] = self.cylinder.radius * np.cos(theta)
        self.local_forcing_grid_position[1, :] = self.cylinder.radius * np.sin(theta)

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the circular cylinder boundary"""
        self.position_field[...] = (
            self.cylinder.position_collection[: self.grid_dim]
            + self.local_forcing_grid_position
        )

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the circular cylinder boundary"""
        # d3 aligned along Z while d1 and d2 along XY plane...
        # Can be shown that omega local and global lie along d3 (Z axis)
        global_frame_omega_z = (
            self.cylinder.director_collection[self.grid_dim, self.grid_dim, 0]
            * self.cylinder.omega_collection[self.grid_dim, 0]
        )
        self.velocity_field[0] = (
            self.cylinder.velocity_collection[0]
            - global_frame_omega_z * self.local_forcing_grid_position[1]
        )
        self.velocity_field[1] = (
            self.cylinder.velocity_collection[1]
            + global_frame_omega_z * self.local_forcing_grid_position[0]
        )

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the circular cylinder"""
        # negative sign due to Newtons third law
        body_flow_forces[: self.grid_dim] = -np.sum(lag_grid_forcing_field)

        # torque from grid forcing
        # Q @ (0, 0, torque) = d3 dot (0, 0, torque) = Q[2, 2] * (0, 0, torque)
        body_flow_torques[self.grid_dim] = self.cylinder.director_collection[
            self.grid_dim, self.grid_dim, 0
        ] * np.sum(
            -self.local_forcing_grid_position[0] * lag_grid_forcing_field[1]
            + self.local_forcing_grid_position[1] * lag_grid_forcing_field[0]
        )

from elastica._linalg import _batch_cross, _batch_matvec
from elastica.rod.cosserat_rod import CosseratRod

import numpy as np

from sopht_simulator.immersed_body import ImmersedBodyForcingGrid


class CosseratRodNodalForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid at Cosserat rod nodes"""

    def __init__(self, grid_dim, cosserat_rod: CosseratRod):
        self.num_lag_nodes = cosserat_rod.n_elems + 1
        self.cosserat_rod = cosserat_rod
        super().__init__(grid_dim)
        self.moment_arm = np.zeros((3, cosserat_rod.n_elems))

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the Cosserat rod"""
        self.position_field[...] = self.cosserat_rod.position_collection[
            : self.grid_dim
        ]

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the Cosserat rod"""
        self.velocity_field[...] = self.cosserat_rod.velocity_collection[
            : self.grid_dim
        ]

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        # negative sign due to Newtons third law
        body_flow_forces[: self.grid_dim] = -lag_grid_forcing_field

        # torque from grid forcing
        self.moment_arm[...] = (
            self.cosserat_rod.position_collection[..., 1:]
            - self.cosserat_rod.position_collection[..., :-1]
        ) / 2.0
        body_flow_torques[...] = _batch_cross(
            self.moment_arm,
            (body_flow_forces[..., 1:] - body_flow_forces[..., :-1]) / 2.0,
        )
        # end element corrections
        body_flow_torques[..., -1] += (
            np.cross(
                self.moment_arm[..., -1],
                body_flow_forces[..., -1],
            )
            / 2.0
        )
        body_flow_torques[..., 0] -= (
            np.cross(
                self.moment_arm[..., 0],
                body_flow_forces[..., 0],
            )
            / 2.0
        )
        # convert global to local frame
        body_flow_torques[...] = _batch_matvec(
            self.cosserat_rod.director_collection,
            body_flow_torques,
        )


class CosseratRodElementCentricForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid at Cosserat rod element centers"""

    def __init__(self, grid_dim, cosserat_rod: CosseratRod):
        self.num_lag_nodes = cosserat_rod.n_elems
        self.cosserat_rod = cosserat_rod
        super().__init__(grid_dim)

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the Cosserat rod"""
        self.position_field[...] = (
            self.cosserat_rod.position_collection[: self.grid_dim, 1:]
            + self.cosserat_rod.position_collection[: self.grid_dim, :-1]
        ) / 2.0

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the Cosserat rod"""
        self.velocity_field[...] = (
            self.cosserat_rod.velocity_collection[: self.grid_dim, 1:]
            + self.cosserat_rod.velocity_collection[: self.grid_dim, :-1]
        ) / 2.0

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        # negative sign due to Newtons third law
        body_flow_forces[...] = 0.0
        body_flow_forces[: self.grid_dim, 1:] -= 0.5 * lag_grid_forcing_field
        body_flow_forces[: self.grid_dim, :-1] -= 0.5 * lag_grid_forcing_field

        # torque from grid forcing (don't modify since set = 0 at initialisation)
        # because no torques acting on element centers

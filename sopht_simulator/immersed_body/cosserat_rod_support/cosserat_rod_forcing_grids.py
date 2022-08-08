from elastica._linalg import _batch_cross, _batch_matvec
from elastica.rod.cosserat_rod import CosseratRod

import logging

import numpy as np


class CosseratRodNoForcingGrid:
    """
    This is the base class for forcing grid in Cosserat rod-flow coupling.

    Notes
    -----
    Every new forcing grid class must be derived
    from CosseratRodNoForcingGrid class.

    """

    num_lag_nodes: int

    def __init__(self, grid_dim, cosserat_rod: CosseratRod):
        self.grid_dim = grid_dim
        self.cosserat_rod = cosserat_rod
        self.position_field = np.zeros((self.grid_dim, self.num_lag_nodes))
        self.velocity_field = np.zeros_like(self.position_field)
        if grid_dim == 2:
            log = logging.getLogger()
            log.warning(
                "========================================================"
                "\n2D rod forcing grid generated, this assumes the rod is"
                "\nin XY plane! Please initialize your rod and ensuing "
                "\ndynamics in XY plane!"
                "\n========================================================"
            )

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the Cosserat rod"""

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the Cosserat rod"""

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""


class CosseratRodNodalForcingGrid(CosseratRodNoForcingGrid):
    """Class for forcing grid at Cosserat rod nodes"""

    def __init__(self, grid_dim, cosserat_rod: CosseratRod):
        self.num_lag_nodes = cosserat_rod.n_elems + 1
        super().__init__(grid_dim, cosserat_rod)
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


class CosseratRodElementCentricForcingGrid(CosseratRodNoForcingGrid):
    """Class for forcing grid at Cosserat rod element centers"""

    def __init__(self, grid_dim, cosserat_rod: CosseratRod):
        self.num_lag_nodes = cosserat_rod.n_elems
        super().__init__(grid_dim, cosserat_rod)

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

from abc import ABC, abstractmethod
import logging
import numpy as np


class ImmersedBodyForcingGrid(ABC):
    """
    This is the base class for forcing grid in immersed body-flow coupling.

    Notes
    -----
    Every new forcing grid class must be derived
    from ImmersedBodyForcingGrid class.

    """

    # Will be set in derived classes
    num_lag_nodes: int = NotImplementedError

    def __init__(self, grid_dim):
        self.grid_dim = grid_dim
        self.position_field = np.zeros((self.grid_dim, self.num_lag_nodes))
        self.velocity_field = np.zeros_like(self.position_field)
        if grid_dim == 2:
            log = logging.getLogger()
            log.warning(
                "=========================================================="
                "\n2D body forcing grid generated, this assumes the body"
                "\nmoves in XY plane! Please initialize your body such that"
                "\nensuing dynamics are constrained in XY plane!"
                "\n=========================================================="
            )

    @abstractmethod
    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the Cosserat rod"""

    @abstractmethod
    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the Cosserat rod"""

    @abstractmethod
    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""

    @abstractmethod
    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""

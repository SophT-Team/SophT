from abc import abstractmethod
import logging
import numpy as np


class ImmersedBodyForcingGrid:
    """
    This is the base class for forcing grid in immersed body-flow coupling.

    Notes
    -----
    Every new forcing grid class must be derived
    from ImmersedBodyForcingGrid class.

    """

    def __init__(self, grid_dim: int, num_lag_nodes: int) -> None:
        # Will be set in derived classes
        self.grid_dim = grid_dim
        self.num_lag_nodes = num_lag_nodes
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
    def compute_lag_grid_position_field(self) -> None:
        """Computes location of forcing grid for the Cosserat rod"""

    @abstractmethod
    def compute_lag_grid_velocity_field(self) -> None:
        """Computes velocity of forcing grid points for the Cosserat rod"""

    @abstractmethod
    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        lag_grid_forcing_field: np.ndarray,
    ) -> None:
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""

    @abstractmethod
    def get_maximum_lagrangian_grid_spacing(self) -> float:
        """Get the maximum Lagrangian grid spacing"""

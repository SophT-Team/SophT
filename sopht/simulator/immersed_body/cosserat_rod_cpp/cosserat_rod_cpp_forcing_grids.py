from elastica.interaction import node_to_element_velocity
import numpy as np
from sopht.simulator.immersed_body import ImmersedBodyForcingGrid


class CosseratRodCPPElementCentricForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid at Cosserat rod element centers"""

    def __init__(self, grid_dim: int, cosserat_rod_simulator) -> None:
        self.cs = cosserat_rod_simulator
        num_lag_nodes = int(self.cs.n_elems)
        super().__init__(grid_dim, num_lag_nodes)

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def compute_lag_grid_position_field(self) -> None:
        """Computes location of forcing grid for the Cosserat rod"""
        rod, _ = self.cs.communicate()
        rod_position_field = np.asarray(rod.position_collection)

        self.position_field[...] = (
            rod_position_field[: self.grid_dim, 1:]
            + rod_position_field[: self.grid_dim, :-1]
        ) / 2.0

    def compute_lag_grid_velocity_field(self) -> None:
        """Computes velocity of forcing grid points for the Cosserat rod"""
        rod, _ = self.cs.communicate()
        rod_velocity_field = np.asarray(rod.velocity_collection)
        rod_masses = np.asarray(rod.mass)

        self.velocity_field[...] = node_to_element_velocity(
            rod_masses, rod_velocity_field
        )[: self.grid_dim]

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        lag_grid_forcing_field: np.ndarray,
    ) -> None:
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        # negative sign due to Newtons third law
        _, body_forces_cpp = self.cs.communicate()
        body_forces_cpp = np.asarray(body_forces_cpp)
        body_flow_forces[...] = 0.0
        body_flow_forces[: self.grid_dim, 1:] -= 0.5 * lag_grid_forcing_field
        body_flow_forces[: self.grid_dim, :-1] -= 0.5 * lag_grid_forcing_field

        body_forces_cpp[: self.grid_dim, 1:] -= 0.5 * lag_grid_forcing_field
        body_forces_cpp[: self.grid_dim, :-1] -= 0.5 * lag_grid_forcing_field

        # torque from grid forcing (don't modify since set = 0 at initialisation)
        # because no torques acting on element centers

    def get_maximum_lagrangian_grid_spacing(self) -> float:
        """Get the maximum Lagrangian grid spacing"""
        # estimated distance between consecutive elements
        rod, _ = self.cs.communicate()
        return np.amax(np.asarray(rod.lengths))

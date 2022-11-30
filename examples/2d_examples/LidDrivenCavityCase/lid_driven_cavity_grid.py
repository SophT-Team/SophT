import numpy as np
from sopht.simulator.immersed_body import ImmersedBodyForcingGrid
import sopht.utils as spu
from typing import Tuple


class LidDrivenCavityForcingGrid(ImmersedBodyForcingGrid):
    """
    Class for forcing grid of a 2D lid driven cavity in XY plane.

    """

    def __init__(
        self,
        grid_dim: int,
        num_lag_nodes_per_side: int,
        side_length: float,
        lid_velocity: float,
        cavity_com: Tuple[float, float],
    ) -> None:
        self.side_length = side_length
        self.num_lag_nodes_per_side = num_lag_nodes_per_side
        self.cavity_com = cavity_com
        num_forcing_points = 4 * num_lag_nodes_per_side
        super().__init__(grid_dim=grid_dim, num_lag_nodes=num_forcing_points)
        self.compute_initial_lag_grid_position_field()
        self.compute_initial_lag_grid_velocity_field(lid_velocity)

    def compute_initial_lag_grid_position_field(self) -> None:
        """Compute the initial positions of the lid driven cavity"""
        ds = self.side_length / self.num_lag_nodes_per_side
        side_coordinates_range = np.linspace(
            -0.5 * self.side_length + 0.5 * ds,
            0.5 * self.side_length - 0.5 * ds,
            self.num_lag_nodes_per_side,
        )
        x_axis_idx = spu.VectorField.x_axis_idx()
        y_axis_idx = spu.VectorField.y_axis_idx()
        # top boundary
        self.position_field[
            x_axis_idx, : self.num_lag_nodes_per_side
        ] = side_coordinates_range
        self.position_field[y_axis_idx, : self.num_lag_nodes_per_side] = (
            0.5 * self.side_length
        )
        # right boundary
        self.position_field[
            x_axis_idx, self.num_lag_nodes_per_side : 2 * self.num_lag_nodes_per_side
        ] = (0.5 * self.side_length)
        self.position_field[
            y_axis_idx, self.num_lag_nodes_per_side : 2 * self.num_lag_nodes_per_side
        ] = side_coordinates_range
        # bottom boundary
        self.position_field[
            x_axis_idx,
            2 * self.num_lag_nodes_per_side : 3 * self.num_lag_nodes_per_side,
        ] = side_coordinates_range
        self.position_field[
            y_axis_idx,
            2 * self.num_lag_nodes_per_side : 3 * self.num_lag_nodes_per_side,
        ] = (
            -0.5 * self.side_length
        )
        # left boundary
        self.position_field[
            x_axis_idx,
            3 * self.num_lag_nodes_per_side : 4 * self.num_lag_nodes_per_side,
        ] = (
            -0.5 * self.side_length
        )
        self.position_field[
            y_axis_idx,
            3 * self.num_lag_nodes_per_side : 4 * self.num_lag_nodes_per_side,
        ] = side_coordinates_range
        # shift the cavity to the desired COM
        for axis in range(self.grid_dim):
            self.position_field[axis] += self.cavity_com[axis]

    def compute_initial_lag_grid_velocity_field(self, lid_velocity: float) -> None:
        """Compute the initial velocities of the lid driven cavity"""
        self.velocity_field[
            spu.VectorField.x_axis_idx(), : self.num_lag_nodes_per_side
        ] = lid_velocity

    def compute_lag_grid_position_field(self) -> None:
        """Computes positions of the lid driven cavity"""
        # We don't do anything here, since these are computed statically
        # during initialisation
        pass

    def compute_lag_grid_velocity_field(self) -> None:
        """Computes velocities of the lid driven cavity"""
        # We don't do anything here, since these are computed statically
        # during initialisation
        pass

    def get_maximum_lagrangian_grid_spacing(self) -> float:
        """Get the maximum Lagrangian grid spacing"""
        return self.side_length / self.num_lag_nodes_per_side

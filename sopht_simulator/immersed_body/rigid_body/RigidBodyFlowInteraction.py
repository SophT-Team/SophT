from elastica import RigidBodyBase

import numpy as np

from sopht_simulator.immersed_body.rigid_body.rigid_body_forcing_grids import (
    CircularCylinderForcingGrid,
    SquareCylinderForcingGrid,
)

from sopht_simulator.immersed_body import ImmersedBodyFlowInteraction


class RigidBodyFlowInteraction(ImmersedBodyFlowInteraction):
    """Class for Cosserat rod flow interaction."""

    def __init__(
        self,
        num_forcing_points,
        rigid_body: type(RigidBodyBase),
        eul_grid_forcing_field,
        eul_grid_velocity_field,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
        dx,
        grid_dim,
        real_t=np.float64,
        eul_grid_coord_shift=None,
        interp_kernel_width=None,
        enable_eul_grid_forcing_reset=False,
        num_threads=False,
        start_time=0.0,
        forcing_grid_type="2d_circular_cylinder",
    ):
        """Class initialiser."""
        self.body_flow_forces = np.zeros((3, 1))
        self.body_flow_torques = np.zeros((3, 1))
        if forcing_grid_type == "2d_circular_cylinder":
            self.forcing_grid = CircularCylinderForcingGrid(
                grid_dim=grid_dim,
                num_forcing_points=num_forcing_points,
                cylinder=rigid_body,
            )
        elif forcing_grid_type == "2d_square_cylinder":
            self.forcing_grid = SquareCylinderForcingGrid(
                grid_dim=grid_dim,
                num_forcing_points=num_forcing_points,
                cylinder=rigid_body,
            )
        else:
            raise NotImplementedError

        # initialising super class
        super().__init__(
            eul_grid_forcing_field,
            eul_grid_velocity_field,
            virtual_boundary_stiffness_coeff,
            virtual_boundary_damping_coeff,
            dx,
            grid_dim,
            real_t,
            eul_grid_coord_shift,
            interp_kernel_width,
            enable_eul_grid_forcing_reset,
            num_threads,
            start_time,
        )

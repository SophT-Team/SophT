from elastica._linalg import _batch_cross, _batch_matvec
from elastica.rod.cosserat_rod import CosseratRod

import numpy as np

from sopht.numeric.immersed_boundary_ops import VirtualBoundaryForcing


class CosseratRodFlowInteraction(VirtualBoundaryForcing):
    """Class for Cosserat rod flow interaction.
    If grid_dim =2 , ASSUMES ROD MOVES IN XY PLANE!
    """

    def __init__(
        self,
        cosserat_rod: CosseratRod,
        eul_grid_forcing_field,
        eul_grid_velocity_field,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
        dx,
        grid_dim,
        real_t=np.float64,
        eul_grid_coord_shift=None,
        interp_kernel_width=None,
        enable_eul_grid_forcing_reset=True,
        num_threads=False,
        start_time=0.0,
        forcing_grid_type="nodal",
    ):
        """Class initialiser."""
        # these hold references to Eulerian fields
        self.eul_grid_forcing_field = eul_grid_forcing_field.view()
        self.eul_grid_velocity_field = eul_grid_velocity_field.view()
        self.cosserat_rod_flow_forces = np.zeros(
            (3, cosserat_rod.n_elems + 1),
        )
        self.cosserat_rod_flow_torques = np.zeros(
            (3, cosserat_rod.n_elems),
        )

        if forcing_grid_type == "nodal":
            self.forcing_grid = CosseratRodNodalForcingGrid(
                grid_dim=grid_dim, cosserat_rod=cosserat_rod
            )

        # initialising super class
        VirtualBoundaryForcing.__init__(
            self,
            virtual_boundary_stiffness_coeff,
            virtual_boundary_damping_coeff,
            grid_dim,
            dx,
            self.forcing_grid.num_lag_nodes,
            real_t,
            eul_grid_coord_shift,
            interp_kernel_width,
            enable_eul_grid_forcing_reset,
            num_threads,
            start_time,
        )

    def compute_interaction_on_lag_grid(self):
        """Compute interaction forces on the Lagrangian forcing grid."""
        self.forcing_grid.compute_lag_grid_position_field()
        self.forcing_grid.compute_lag_grid_velocity_field()
        self.compute_interaction_force_on_lag_grid(
            eul_grid_velocity_field=self.eul_grid_velocity_field,
            lag_grid_position_field=self.forcing_grid.position_field,
            lag_grid_velocity_field=self.forcing_grid.velocity_field,
        )

    def __call__(self):
        # call the full interaction (eul and lag field force computation)
        self.forcing_grid.compute_lag_grid_position_field()
        self.forcing_grid.compute_lag_grid_velocity_field()
        self.compute_interaction_forcing(
            eul_grid_forcing_field=self.eul_grid_forcing_field,
            eul_grid_velocity_field=self.eul_grid_velocity_field,
            lag_grid_position_field=self.forcing_grid.position_field,
            lag_grid_velocity_field=self.forcing_grid.velocity_field,
        )

    def compute_flow_forces_and_torques(self):
        """Compute flow forces and torques on rod from forces on Lagrangian grid."""
        self.compute_interaction_on_lag_grid()
        self.forcing_grid.transfer_forcing_from_grid_to_rod(
            cosserat_rod_flow_forces=self.cosserat_rod_flow_forces,
            cosserat_rod_flow_torques=self.cosserat_rod_flow_torques,
            lag_grid_forcing_field=self.lag_grid_forcing_field,
        )


class CosseratRodNodalForcingGrid:
    """Class for forcing grid at Cosserat rod nodes"""

    def __init__(self, grid_dim, cosserat_rod: CosseratRod):
        self.grid_dim = grid_dim
        self.cosserat_rod = cosserat_rod
        self.num_lag_nodes = cosserat_rod.n_elems + 1
        self.position_field = np.zeros((self.grid_dim, self.num_lag_nodes))
        self.velocity_field = np.zeros_like(self.position_field)
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

    def transfer_forcing_from_grid_to_rod(
        self,
        cosserat_rod_flow_forces,
        cosserat_rod_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        # negative sign due to Newtons third law
        cosserat_rod_flow_forces[: self.grid_dim] = -lag_grid_forcing_field

        # torque from grid forcing
        self.moment_arm[...] = (
            self.cosserat_rod.position_collection[..., 1:]
            - self.cosserat_rod.position_collection[..., :-1]
        ) / 2.0
        cosserat_rod_flow_torques[...] = _batch_cross(
            self.moment_arm,
            (cosserat_rod_flow_forces[..., 1:] - cosserat_rod_flow_forces[..., :-1])
            / 2.0,
        )
        # end element corrections
        cosserat_rod_flow_torques[..., -1] += (
            np.cross(
                self.moment_arm[..., -1],
                cosserat_rod_flow_forces[..., -1],
            )
            / 2.0
        )
        cosserat_rod_flow_torques[..., 0] -= (
            np.cross(
                self.moment_arm[..., 0],
                cosserat_rod_flow_forces[..., 0],
            )
            / 2.0
        )
        # convert global to local frame
        cosserat_rod_flow_torques[...] = _batch_matvec(
            self.cosserat_rod.director_collection,
            cosserat_rod_flow_torques,
        )

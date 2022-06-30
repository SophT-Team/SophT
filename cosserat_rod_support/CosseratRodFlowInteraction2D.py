import numpy as np

from sopht.numeric.immersed_boundary_ops import VirtualBoundaryForcing


class CosseratRodFlowInteraction2D(VirtualBoundaryForcing):
    """Class for 2D Cosserat rod flow interaction.

    ASSUMES ROD MOVES IN XY PLANE!
    """

    def __init__(
        self,
        cosserat_rod,
        eul_grid_forcing_field,
        eul_grid_velocity_field,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
        dx,
        real_t,
        eul_grid_coord_shift=None,
        interp_kernel_width=None,
        enable_eul_grid_forcing_reset=True,
        num_threads=False,
        start_time=0.0,
        forcing_grid_type="nodal",
    ):
        """Class initialiser."""
        self.cosserat_rod = cosserat_rod
        # these hold references to Eulerian fields
        self.eul_grid_forcing_field = eul_grid_forcing_field.view()
        self.eul_grid_velocity_field = eul_grid_velocity_field.view()
        grid_dim = 2
        if forcing_grid_type == "nodal":
            self.num_lag_nodes = cosserat_rod.n_elems + 1

        # initialising super class
        VirtualBoundaryForcing.__init__(
            self,
            virtual_boundary_stiffness_coeff,
            virtual_boundary_damping_coeff,
            2,
            dx,
            self.num_lag_nodes,
            real_t,
            eul_grid_coord_shift,
            interp_kernel_width,
            enable_eul_grid_forcing_reset,
            num_threads,
            start_time,
        )
        # creating additional internal buffers
        self.lag_grid_position_field = np.zeros_like(
            self.lag_grid_position_mismatch_field
        )
        self.lag_grid_velocity_field = np.zeros_like(self.lag_grid_position_field)
        # 3 here bcoz pyelastica has 3D fields
        self.cosserat_rod_flow_forces = np.zeros(
            (3, cosserat_rod.n_elems + 1), dtype=real_t
        )
        self.cosserat_rod_flow_torques = np.zeros(
            (3, cosserat_rod.n_elems), dtype=real_t
        )

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the Cosserat rod"""
        self.lag_grid_position_field[...] = self.cosserat_rod.position_collection[
            : self.grid_dim
        ]

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the Cosserat rod"""
        self.lag_grid_velocity_field[...] = self.cosserat_rod.velocity_collection[
            : self.grid_dim
        ]

    def compute_interaction_on_lag_grid(self):
        """Compute interaction forces on the Lagrangian forcing grid."""
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()
        self.compute_interaction_force_on_lag_grid(
            eul_grid_velocity_field=self.eul_grid_velocity_field,
            lag_grid_position_field=self.lag_grid_position_field,
            lag_grid_velocity_field=self.lag_grid_velocity_field,
        )

    def __call__(self):
        # call the full interaction (eul and lag field force computation)
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()
        self.compute_interaction_forcing(
            eul_grid_forcing_field=self.eul_grid_forcing_field,
            eul_grid_velocity_field=self.eul_grid_velocity_field,
            lag_grid_position_field=self.lag_grid_position_field,
            lag_grid_velocity_field=self.lag_grid_velocity_field,
        )

    def compute_flow_forces_and_torques(self):
        """Compute flow forces and torques on rod from forces on Lagrangian grid."""
        self.cosserat_rod_flow_forces[: self.grid_dim] = self.lag_grid_forcing_field
        # TODO we need to add torques for nodal forces?

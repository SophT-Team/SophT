import numpy as np

from sopht.numeric.immersed_boundary_ops import VirtualBoundaryForcing

from sopht_simulator.immersed_body import ImmersedBodyForcingGrid


class ImmersedBodyFlowInteraction(VirtualBoundaryForcing):
    """Base class for immersed body flow interaction."""

    # These are meant to be initialised in the derived classes
    body_flow_forces: np.ndarray
    body_flow_torques: np.ndarray
    forcing_grid: type(ImmersedBodyForcingGrid)

    def __init__(
        self,
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
    ):
        """Class initialiser."""
        # these hold references to Eulerian fields
        self.eul_grid_forcing_field = eul_grid_forcing_field.view()
        self.eul_grid_velocity_field = eul_grid_velocity_field.view()

        # initialising super class
        super().__init__(
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
        """Compute flow forces and torques on the body from forces on Lagrangian grid."""
        self.compute_interaction_on_lag_grid()
        self.forcing_grid.transfer_forcing_from_grid_to_body(
            body_flow_forces=self.body_flow_forces,
            body_flow_torques=self.body_flow_torques,
            lag_grid_forcing_field=self.lag_grid_forcing_field,
        )

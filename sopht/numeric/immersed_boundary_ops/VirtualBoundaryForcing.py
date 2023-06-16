"""Virtual boundary forcing for flow-body feedback."""
from numba import njit
from typing import Literal
import numpy as np

from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_set_fixed_val_pyst_kernel_2d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_set_fixed_val_pyst_kernel_3d,
)
from sopht.numeric.immersed_boundary_ops.EulerianLagrangianGridCommunicator2D import (
    EulerianLagrangianGridCommunicator2D,
)
from sopht.numeric.immersed_boundary_ops.EulerianLagrangianGridCommunicator3D import (
    EulerianLagrangianGridCommunicator3D,
)


class VirtualBoundaryForcing:
    """Class for virtual boundary forcing.

    Virtual boundary forcing class for computing feedback between the
    Lagrangian body and Eulerian grid flow, using virtual boundary method
    Refer to Goldstein 1993, JCP for details on the penalty force computation.
    TODO add proper style docs
    """

    def __init__(
        self,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
        grid_dim,
        dx,
        num_lag_nodes,
        real_t,
        eul_grid_coord_shift=None,
        interp_kernel_width=None,
        enable_eul_grid_forcing_reset=True,
        num_threads=False,
        start_time=0.0,
        field_type: Literal["scalar", "vector"] = "vector",
    ):
        """Class initialiser.

        TODO add proper style docs
        Takes in inputs:
        virtual_boundary_stiffness_coeff: stiffness coefficient for computing penalty
        force, set to a high values
        virtual_boundary_damping_coeff: damping coefficient for computing penalty
        force, added for stabilising force
        grid_dim: dimensions of the grid
        dx: Eulerian grid spacing
        eul_grid_coord_shift: shift of the coordinates of the Eulerian grid start from
        0 (usually dx / 2)
        num_lag_nodes: number of Lagrangian grid nodes
        interp_kernel_width: width of interpolation kernel
        real_t: numerical precision
        enable_eul_grid_forcing_reset : flag for enabling option of feedback step
        with resetting of eul_grid_forcing_field
        num_threads: number of threads (only for the resetting function)
        start_time: start time of the forcing
        field_type: This can be a vector like velocity as we use traditionally in p-IBM, or scalar like temperature.
        """
        assert grid_dim == 2 or grid_dim == 3, "Invalid grid dimensions"
        self.grid_dim = grid_dim
        self.virtual_boundary_stiffness_coeff = virtual_boundary_stiffness_coeff
        self.virtual_boundary_damping_coeff = virtual_boundary_damping_coeff
        self.time = start_time
        assert field_type == "scalar" or field_type == "vector", "Invalid field type"
        self.field_type = field_type

        # these are rather invariant hence pushed to fixed kwargs
        if eul_grid_coord_shift is None:
            eul_grid_coord_shift = real_t(dx / 2)
        if interp_kernel_width is None:
            interp_kernel_width = 2

        # creating buffers...
        self.nearest_eul_grid_index_to_lag_grid = np.empty(
            (grid_dim, num_lag_nodes), dtype=int
        )
        eul_grid_support_of_lag_grid_shape = (
            (grid_dim,) + (2 * interp_kernel_width,) * grid_dim + (num_lag_nodes,)
        )
        self.local_eul_grid_support_of_lag_grid = np.empty(
            eul_grid_support_of_lag_grid_shape, dtype=real_t
        )
        interp_weights_shape = (2 * interp_kernel_width,) * grid_dim + (num_lag_nodes,)
        self.interp_weights = np.empty(interp_weights_shape, dtype=real_t)

        if self.field_type == "scalar":
            self.lag_grid_flow_velocity_field = np.zeros((num_lag_nodes), dtype=real_t)
        else:
            self.lag_grid_flow_velocity_field = np.zeros(
                (grid_dim, num_lag_nodes), dtype=real_t
            )
        self.lag_grid_position_mismatch_field = np.zeros_like(
            self.lag_grid_flow_velocity_field
        )
        self.lag_grid_velocity_mismatch_field = np.zeros_like(
            self.lag_grid_position_mismatch_field
        )
        self.lag_grid_forcing_field = np.zeros_like(
            self.lag_grid_velocity_mismatch_field
        )

        self.fixed_vals_eul_grid_forcing = (
            [0] * self.grid_dim if self.field_type == "vector" else 0
        )

        if grid_dim == 2:
            self.eul_lag_grid_communicator = EulerianLagrangianGridCommunicator2D(
                dx=dx,
                eul_grid_coord_shift=eul_grid_coord_shift,
                num_lag_nodes=num_lag_nodes,
                interp_kernel_width=interp_kernel_width,
                real_t=real_t,
                n_components=grid_dim if self.field_type == "vector" else 1,
            )
        elif grid_dim == 3:
            self.eul_lag_grid_communicator = EulerianLagrangianGridCommunicator3D(
                dx=dx,
                eul_grid_coord_shift=eul_grid_coord_shift,
                num_lag_nodes=num_lag_nodes,
                interp_kernel_width=interp_kernel_width,
                real_t=real_t,
                n_components=grid_dim if self.field_type == "vector" else 1,
            )

        if enable_eul_grid_forcing_reset:
            if grid_dim == 2:
                self.set_eul_grid_forcing_field = gen_set_fixed_val_pyst_kernel_2d(
                    real_t=real_t,
                    num_threads=num_threads,
                    field_type=self.field_type,
                    # field_type="vector",
                )

            elif grid_dim == 3:
                self.set_eul_grid_forcing_field = gen_set_fixed_val_pyst_kernel_3d(
                    real_t=real_t,
                    num_threads=num_threads,
                    field_type=self.field_type,
                    # field_type="vector",
                )
            self.compute_interaction_forcing = (
                self.compute_interaction_force_on_eul_and_lag_grid_with_eul_grid_forcing_reset
            )
        else:
            self.compute_interaction_forcing = (
                self.compute_interaction_force_on_eul_and_lag_grid
            )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def compute_lag_grid_velocity_mismatch_field(
        lag_grid_velocity_mismatch_field,
        lag_grid_flow_velocity_field,
        lag_grid_body_velocity_field,
    ):
        """Compute Lagrangian grid velocity mismatch, between the flow and body.

        We can use pystencils for this but seems like it will be O(N) work, and wont be
        the limiter at least for few rods.

        """
        lag_grid_velocity_mismatch_field[...] = (
            lag_grid_flow_velocity_field - lag_grid_body_velocity_field
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def update_lag_grid_position_mismatch_field_via_euler_forward(
        lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field,
        dt,
    ):
        """Update Lagrangian grid position mismatch, between the flow and body.

        We can use pystencils for this but seems like it will be O(N) work, and wont be
        the limiter at least for few rods.

        """
        lag_grid_position_mismatch_field[...] = (
            lag_grid_position_mismatch_field + dt * lag_grid_velocity_mismatch_field
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def compute_lag_grid_forcing_field(
        lag_grid_forcing_field,
        lag_grid_position_mismatch_field,
        lag_grid_velocity_mismatch_field,
        virtual_boundary_stiffness_coeff,
        virtual_boundary_damping_coeff,
    ):
        """Compute penalty force on Lagrangian grid, defined via virtual boundary method.

        Refer to Goldstein 1993, JCP for details on the penalty force computation.
        We can use pystencils for this but seems like it will be O(N) work, and wont be
        the limiter at least for few rods.

        """
        lag_grid_forcing_field[...] = (
            virtual_boundary_stiffness_coeff * lag_grid_position_mismatch_field
            + virtual_boundary_damping_coeff * lag_grid_velocity_mismatch_field
        )

    def compute_interaction_force_on_lag_grid(
        self,
        eul_grid_velocity_field,
        lag_grid_position_field,
        lag_grid_velocity_field,
    ):
        """Virtual boundary: compute interaction force on Lagrangian grid."""
        # 1. Find Eulerian grid local support of the Lagrangian grid
        self.eul_lag_grid_communicator.local_eulerian_grid_support_of_lagrangian_grid_kernel(
            local_eul_grid_support_of_lag_grid=self.local_eul_grid_support_of_lag_grid,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
            lag_positions=lag_grid_position_field,
        )

        # 2. Compute interpolation weights based on local Eulerian grid support
        self.eul_lag_grid_communicator.interpolation_weights_kernel(
            interp_weights=self.interp_weights,
            local_eul_grid_support_of_lag_grid=self.local_eul_grid_support_of_lag_grid,
        )

        # 3. Interpolate Eulerian flow velocity onto the Lagrangian grid
        self.eul_lag_grid_communicator.eulerian_to_lagrangian_grid_interpolation_kernel(
            lag_grid_field=self.lag_grid_flow_velocity_field,
            eul_grid_field=eul_grid_velocity_field,
            interp_weights=self.interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )

        # 4. Compute velocity mismatch between flow and body on Lagrangian grid
        self.compute_lag_grid_velocity_mismatch_field(
            lag_grid_velocity_mismatch_field=self.lag_grid_velocity_mismatch_field,
            lag_grid_flow_velocity_field=self.lag_grid_flow_velocity_field,
            lag_grid_body_velocity_field=lag_grid_velocity_field,
        )

        # 5. Compute penalty force using virtual boundary forcing formulation
        # on Lagrangian grid
        self.compute_lag_grid_forcing_field(
            lag_grid_forcing_field=self.lag_grid_forcing_field,
            lag_grid_position_mismatch_field=self.lag_grid_position_mismatch_field,
            lag_grid_velocity_mismatch_field=self.lag_grid_velocity_mismatch_field,
            virtual_boundary_stiffness_coeff=self.virtual_boundary_stiffness_coeff,
            virtual_boundary_damping_coeff=self.virtual_boundary_damping_coeff,
        )

    def compute_interaction_force_on_eul_and_lag_grid(
        self,
        eul_grid_forcing_field,
        eul_grid_velocity_field,
        lag_grid_position_field,
        lag_grid_velocity_field,
    ):
        """Virtual boundary: compute interaction on Eulerian grid."""
        # 1. Compute penalty force using virtual boundary forcing formulation
        # on Lagrangian grid
        self.compute_interaction_force_on_lag_grid(
            eul_grid_velocity_field,
            lag_grid_position_field,
            lag_grid_velocity_field,
        )
        # 2. Interpolate penalty forcing from Lagrangian onto the Eulerian grid
        self.eul_lag_grid_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
            eul_grid_field=eul_grid_forcing_field,
            lag_grid_field=self.lag_grid_forcing_field,
            interp_weights=self.interp_weights,
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )

    def compute_interaction_force_on_eul_and_lag_grid_with_eul_grid_forcing_reset(
        self,
        eul_grid_forcing_field,
        eul_grid_velocity_field,
        lag_grid_position_field,
        lag_grid_velocity_field,
    ):
        """Virtual boundary: compute interaction on Eulerian grid.

        Resets eul_grid_forcing_field.
        """
        self.set_eul_grid_forcing_field(
            eul_grid_forcing_field, self.fixed_vals_eul_grid_forcing
        )
        self.compute_interaction_force_on_eul_and_lag_grid(
            eul_grid_forcing_field,
            eul_grid_velocity_field,
            lag_grid_position_field,
            lag_grid_velocity_field,
        )

    def time_step(self, dt):
        """Virtual boundary forcing time step, updates grid deviation."""
        self.update_lag_grid_position_mismatch_field_via_euler_forward(
            lag_grid_position_mismatch_field=self.lag_grid_position_mismatch_field,
            lag_grid_velocity_mismatch_field=self.lag_grid_velocity_mismatch_field,
            dt=dt,
        )
        self.time += dt

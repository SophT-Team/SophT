"""Brinkmann boundary forcing for flow-body feedback."""
from numba import njit

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


class BrinkmannBoundaryForcing:
    """Class for Brinkmann boundary forcing.

    Brinkmann boundary forcing class for computing feedback between the
    Lagrangian body and Eulerian grid flow, using Lagrangian variant of
    Brinkmann penalisation.
    TODO add proper style docs
    """

    def __init__(
        self,
        brinkmann_coeff,
        grid_dim,
        dx,
        eul_grid_coord_shift,
        num_lag_nodes,
        interp_kernel_width,
        real_t,
        enable_eul_grid_flux_reset=True,
        num_threads=False,
    ):
        """Class initialiser.

        TODO add proper style docs
        Takes in inputs:
        brinkmann_coeff: Brinkmann penalisation coefficient for computing penalty
        force, set to a high value
        grid_dim: dimensions of the grid
        dx: Eulerian grid spacing
        eul_grid_coord_shift: shift of the coordinates of the Eulerian grid start from
        0 (usually dx / 2)
        num_lag_nodes: number of Lagrangian grid nodes
        interp_kernel_width: width of interpolation kernel
        real_t: numerical precision
        enable_eul_grid_flux_reset : flag for enabling option of feedback step
        with resetting of eul_grid_penalisation_flux
        num_threads: number of threads (only for the resetting function)

        """
        self.dx = dx
        assert grid_dim == 2 or grid_dim == 3, "Invalid grid dimensions"
        self.grid_dim = grid_dim
        self.brinkmann_coeff = brinkmann_coeff

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
        self.lag_grid_flow_velocity_field = np.zeros(
            (grid_dim, num_lag_nodes), dtype=real_t
        )
        self.lag_grid_penalised_velocity_field = np.zeros_like(
            self.lag_grid_flow_velocity_field
        )
        self.lag_grid_penalisation_flux = np.zeros_like(
            self.lag_grid_penalised_velocity_field
        )
        self.lag_grid_penalisation_forcing = np.zeros_like(
            self.lag_grid_penalisation_flux
        )

        if grid_dim == 2:
            self.eul_lag_grid_communicator = EulerianLagrangianGridCommunicator2D(
                dx=dx,
                eul_grid_coord_shift=eul_grid_coord_shift,
                num_lag_nodes=num_lag_nodes,
                interp_kernel_width=interp_kernel_width,
                real_t=real_t,
                n_components=grid_dim,
            )
        elif grid_dim == 3:
            self.eul_lag_grid_communicator = EulerianLagrangianGridCommunicator3D(
                dx=dx,
                eul_grid_coord_shift=eul_grid_coord_shift,
                num_lag_nodes=num_lag_nodes,
                interp_kernel_width=interp_kernel_width,
                real_t=real_t,
                n_components=grid_dim,
            )

        if enable_eul_grid_flux_reset:
            if grid_dim == 2:
                self.set_eul_grid_vector_field = gen_set_fixed_val_pyst_kernel_2d(
                    real_t=real_t,
                    num_threads=num_threads,
                    field_type="vector",
                )

            elif grid_dim == 3:
                self.set_eul_grid_vector_field = gen_set_fixed_val_pyst_kernel_3d(
                    real_t=real_t,
                    num_threads=num_threads,
                    field_type="vector",
                )
            self.compute_interaction_forcing = (
                self.compute_interaction_with_eul_grid_flux_reset
            )
        else:
            self.compute_interaction_forcing = (
                self.compute_interaction_without_eul_grid_flux_reset
            )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def brinkmann_penalise_lag_grid_velocity_field(
        lag_grid_penalised_velocity_field,
        lag_grid_flow_velocity_field,
        lag_grid_body_velocity_field,
        brinkmann_coeff,
        dt,
    ):
        """Brinkmann penalise Lagrangian flow grid velocity.

        We can use pystencils for this but seems like it will be O(N) work, and wont be
        the limiter at least for few rods.

        """
        lag_grid_penalised_velocity_field[...] = (
            lag_grid_flow_velocity_field
            + brinkmann_coeff * dt * lag_grid_body_velocity_field
        ) / (1 + brinkmann_coeff * dt)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def compute_lag_grid_penalisation_flux(
        lag_grid_penalisation_flux,
        lag_grid_penalised_velocity_field,
        lag_grid_flow_velocity_field,
    ):
        """Compute penalisation flux on the Lagrangian grid via Brinkmann penalisation."""
        lag_grid_penalisation_flux[...] = (
            lag_grid_penalised_velocity_field - lag_grid_flow_velocity_field
        )

    def compute_lag_grid_penalisation_forcing(self, dt):
        """Compute penalisation forcing on the Lagrangian grid via Brinkmann penalisation."""
        self.lag_grid_penalisation_forcing[...] = (
            (self.dx**self.grid_dim)  # dx factor indicate integration of delta fund
            * self.lag_grid_penalisation_flux
            / dt
        )

    def compute_interaction_without_eul_grid_flux_reset(
        self,
        eul_grid_penalisation_flux,
        eul_grid_velocity_field,
        lag_grid_position_field,
        lag_grid_velocity_field,
        dt,
    ):
        """Brinkmann boundary feedback step, does not reset eul_grid_penalisation_flux."""
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
        # 4. Brinkmann penalise flow velocity Lagrangian grid
        self.brinkmann_penalise_lag_grid_velocity_field(
            lag_grid_penalised_velocity_field=self.lag_grid_penalised_velocity_field,
            lag_grid_flow_velocity_field=self.lag_grid_flow_velocity_field,
            lag_grid_body_velocity_field=lag_grid_velocity_field,
            brinkmann_coeff=self.brinkmann_coeff,
            dt=dt,
        )
        # 5. Compute penalisation flux using  Brinkmann penalisation formulation
        # on Lagrangian grid
        self.compute_lag_grid_penalisation_flux(
            lag_grid_penalisation_flux=self.lag_grid_penalisation_flux,
            lag_grid_penalised_velocity_field=self.lag_grid_penalised_velocity_field,
            lag_grid_flow_velocity_field=self.lag_grid_flow_velocity_field,
        )
        # 6. Interpolate penalisation flux from Lagrangian onto the Eulerian grid
        self.eul_lag_grid_communicator.lagrangian_to_eulerian_grid_interpolation_kernel(
            eul_grid_field=eul_grid_penalisation_flux,
            lag_grid_field=self.lag_grid_penalisation_flux,
            interp_weights=(
                self.interp_weights
                * (
                    self.dx**self.grid_dim
                )  # dx factor converts delta function to interpolator
            ),
            nearest_eul_grid_index_to_lag_grid=self.nearest_eul_grid_index_to_lag_grid,
        )

        # 7. Compute forcing corresponding to the penalisation flux
        self.compute_lag_grid_penalisation_forcing(dt)

    def compute_interaction_with_eul_grid_flux_reset(
        self,
        eul_grid_penalisation_flux,
        eul_grid_velocity_field,
        lag_grid_position_field,
        lag_grid_velocity_field,
        dt,
    ):
        """Brinkmann boundary feedback step, resets eul_grid_penalisation_flux."""
        self.set_eul_grid_vector_field(
            vector_field=eul_grid_penalisation_flux, fixed_vals=([0] * self.grid_dim)
        )
        self.compute_interaction_without_eul_grid_flux_reset(
            eul_grid_penalisation_flux,
            eul_grid_velocity_field,
            lag_grid_position_field,
            lag_grid_velocity_field,
            dt,
        )

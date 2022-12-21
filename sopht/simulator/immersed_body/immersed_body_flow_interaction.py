import logging
import numpy as np
from sopht.numeric.immersed_boundary_ops import VirtualBoundaryForcing
from .immersed_body_forcing_grid import (
    ImmersedBodyForcingGrid,
)
from typing import Type, Optional


class ImmersedBodyFlowInteraction(VirtualBoundaryForcing):
    """Base class for immersed body flow interaction."""

    def __init__(
        self,
        eul_grid_forcing_field: np.ndarray,
        eul_grid_velocity_field: np.ndarray,
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        forcing_grid_cls: Type[ImmersedBodyForcingGrid],
        virtual_boundary_stiffness_coeff: float,
        virtual_boundary_damping_coeff: float,
        dx: float,
        grid_dim: int,
        real_t: type = np.float64,
        eul_grid_coord_shift: Optional[float] = None,
        interp_kernel_width: Optional[float] = None,
        enable_eul_grid_forcing_reset: bool = False,
        num_threads: int | bool = False,
        start_time: float = 0.0,
        **forcing_grid_kwargs,
    ) -> None:
        """Class initialiser."""
        # These are meant to be specialised/created in the derived classes
        self.body_flow_forces = body_flow_forces
        self.body_flow_torques = body_flow_torques
        self.forcing_grid = forcing_grid_cls(
            grid_dim=grid_dim,
            **forcing_grid_kwargs,
        )
        # these hold references to Eulerian fields
        self.eul_grid_forcing_field = eul_grid_forcing_field.view()
        self.eul_grid_velocity_field = eul_grid_velocity_field.view()
        # this class should only "view" the flow velocity
        self.eul_grid_velocity_field.flags.writeable = False

        # check relative resolutions of the Lagrangian and Eulerian grids
        log = logging.getLogger()
        max_lag_grid_dx = self.forcing_grid.get_maximum_lagrangian_grid_spacing()
        grid_type = type(self.forcing_grid).__name__
        log.warning(
            "==========================================================\n"
            f"For {grid_type}:"
        )
        if (
            max_lag_grid_dx > 2 * dx
        ):  # 2 here since the support of delta function is 2 grid points
            log.warning(
                f"Eulerian grid spacing (dx): {dx}"
                f"\nMax Lagrangian grid spacing: {max_lag_grid_dx} > 2 * dx"
                "\nThe Lagrangian grid of the body is too coarse relative to"
                "\nthe Eulerian grid of the flow, which can lead to unexpected"
                "\nconvergence. Please make the Lagrangian grid finer."
            )
        elif max_lag_grid_dx < 0.5 * dx:  # reverse case of the above condition
            log.warning(
                "==========================================================\n"
                f"Eulerian grid spacing (dx): {dx}"
                f"\nMax Lagrangian grid spacing: {max_lag_grid_dx} < 0.5 * dx"
                "\nThe Lagrangian grid of the body is too fine relative to"
                "\nthe Eulerian grid of the flow, which corresponds to redundant"
                "\nforcing points. Please make the Lagrangian grid coarser."
            )
        else:
            log.warning(
                "Lagrangian grid is resolved almost the same as the Eulerian"
                "\ngrid of the flow."
            )
        log.warning("==========================================================")

        # rescale the virtual boundary coeffs by grid spacings
        # (based on previous penalty immersed boundary method works)
        virtual_boundary_stiffness_coeff *= max_lag_grid_dx ** (grid_dim - 1)
        virtual_boundary_damping_coeff *= max_lag_grid_dx ** (grid_dim - 1)

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

    def compute_interaction_on_lag_grid(self) -> None:
        """Compute interaction forces on the Lagrangian forcing grid."""
        self.forcing_grid.compute_lag_grid_position_field()
        self.forcing_grid.compute_lag_grid_velocity_field()
        self.compute_interaction_force_on_lag_grid(
            eul_grid_velocity_field=self.eul_grid_velocity_field,
            lag_grid_position_field=self.forcing_grid.position_field,
            lag_grid_velocity_field=self.forcing_grid.velocity_field,
        )

    def __call__(self) -> None:
        # call the full interaction (eul and lag field force computation)
        self.forcing_grid.compute_lag_grid_position_field()
        self.forcing_grid.compute_lag_grid_velocity_field()
        self.compute_interaction_forcing(
            eul_grid_forcing_field=self.eul_grid_forcing_field,
            eul_grid_velocity_field=self.eul_grid_velocity_field,
            lag_grid_position_field=self.forcing_grid.position_field,
            lag_grid_velocity_field=self.forcing_grid.velocity_field,
        )

    def compute_flow_forces_and_torques(self) -> None:
        """Compute flow forces and torques on the body from forces on Lagrangian grid."""
        self.compute_interaction_on_lag_grid()
        self.forcing_grid.transfer_forcing_from_grid_to_body(
            body_flow_forces=self.body_flow_forces,
            body_flow_torques=self.body_flow_torques,
            lag_grid_forcing_field=self.lag_grid_forcing_field,
        )

    def get_grid_deviation_error_l2_norm(self) -> float:
        """
        Computes and returns L2 norm of deviation error between flow
        and body grids.

        """
        grid_dev_error_l2_norm = np.linalg.norm(
            self.lag_grid_position_mismatch_field
        ) / np.sqrt(self.forcing_grid.num_lag_nodes)
        return grid_dev_error_l2_norm

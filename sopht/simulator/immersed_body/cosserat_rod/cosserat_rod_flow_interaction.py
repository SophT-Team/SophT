import elastica as ea
import numpy as np
from sopht.simulator.immersed_body import (
    ImmersedBodyForcingGrid,
    ImmersedBodyFlowInteraction,
)
from typing import Type, Optional, Literal


class CosseratRodFlowInteraction(ImmersedBodyFlowInteraction):
    """Class for Cosserat rod flow interaction."""

    def __init__(
        self,
        cosserat_rod: ea.CosseratRod,
        eul_grid_forcing_field: np.ndarray,
        eul_grid_velocity_field: np.ndarray,
        virtual_boundary_stiffness_coeff: float,
        virtual_boundary_damping_coeff: float,
        dx: float,
        grid_dim: int,
        forcing_grid_cls: Type[ImmersedBodyForcingGrid],
        real_t: type = np.float64,
        eul_grid_coord_shift: Optional[float] = None,
        interp_kernel_width: Optional[float] = None,
        enable_eul_grid_forcing_reset: bool = False,
        num_threads: int | bool = False,
        start_time: float = 0.0,
        field_type: Literal["scalar", "vector"] = "vector",
        **forcing_grid_kwargs,
    ) -> None:
        """Class initialiser."""
        body_flow_forces = np.zeros(
            (3, cosserat_rod.n_elems + 1),
        )
        body_flow_torques = np.zeros(
            (3, cosserat_rod.n_elems),
        )
        forcing_grid_kwargs["cosserat_rod"] = cosserat_rod

        # initialising super class
        super().__init__(
            eul_grid_forcing_field,
            eul_grid_velocity_field,
            body_flow_forces,
            body_flow_torques,
            forcing_grid_cls,
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
            field_type,
            **forcing_grid_kwargs,
        )

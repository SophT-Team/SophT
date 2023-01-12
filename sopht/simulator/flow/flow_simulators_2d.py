import numpy as np
from typing import Literal
from .navier_stokes_flow_simulators import UnboundedNavierStokesFlowSimulator2D


def UnboundedFlowSimulator2D(
    grid_size: tuple[int, int],
    x_range: float,
    kinematic_viscosity: float,
    CFL: float = 0.1,
    flow_type: Literal["navier_stokes", "navier_stokes_with_forcing"] = "navier_stokes",
    real_t: type = np.float32,
    num_threads: int = 1,
    time: float = 0.0,
    **kwargs,
) -> UnboundedNavierStokesFlowSimulator2D:
    """Function forwarding 2D unbounded Navier Stokes flow simulator.

    :param grid_size: Grid size of simulator
    :param x_range: Range of X coordinate of the grid
    :param kinematic_viscosity: kinematic viscosity of the fluid
    :param CFL: Courant Freidrich Lewy number (advection timestep)
    :param flow_type: Nature of the simulator, can be "passive_scalar" (default value),
    "navier_stokes" or "navier_stokes_with_forcing"
    :param real_t: precision of the solver
    :param num_threads: number of threads
    :param time: simulator time at initialisation

    Notes
    -----
    This function exists for backward compatability.
    Currently, only supports Euler forward timesteps :(
    """
    match flow_type:
        case "navier_stokes":
            with_forcing = False
        case "navier_stokes_with_forcing":
            with_forcing = True
        case _:
            raise ValueError("Invalid flow type given")

    flow_simulator = UnboundedNavierStokesFlowSimulator2D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=kinematic_viscosity,
        cfl=CFL,
        real_t=real_t,
        num_threads=num_threads,
        time=time,
        with_forcing=with_forcing,
        **kwargs,
    )
    return flow_simulator

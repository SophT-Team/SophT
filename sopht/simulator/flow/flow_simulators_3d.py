import numpy as np
from typing import Literal
from .navier_stokes_flow_simulators import UnboundedNavierStokesFlowSimulator3D


def UnboundedFlowSimulator3D(
    grid_size: tuple[int, int, int],
    x_range: float,
    kinematic_viscosity: float,
    CFL: float = 0.1,
    flow_type: Literal["navier_stokes", "navier_stokes_with_forcing"] = "navier_stokes",
    real_t: type = np.float32,
    num_threads: int = 1,
    filter_vorticity: bool = False,
    poisson_solver_type: Literal[
        "greens_function_convolution", "fast_diagonalisation"
    ] = "greens_function_convolution",
    time: float = 0.0,
    **kwargs,
) -> UnboundedNavierStokesFlowSimulator3D:
    """Function forwarding 3D unbounded Navier Stokes flow simulator.

    :param grid_size: Grid size of simulator
    :param x_range: Range of X coordinate of the grid
    :param kinematic_viscosity: kinematic viscosity of the fluid
    :param CFL: Courant Freidrich Lewy number (advection timestep)
    :param flow_type: Nature of the simulator, can be "passive_scalar" (default value),
    "navier_stokes" or "navier_stokes_with_forcing"
    :param real_t: precision of the solver
    :param num_threads: number of threads
    :param filter_vorticity: flag to determine if vorticity should be filtered or not,
    needed for stability sometimes
    :param poisson_solver_type: Type of the poisson solver algorithm, can be
    "greens_function_convolution" or "fast_diagonalisation"
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

    flow_simulator = UnboundedNavierStokesFlowSimulator3D(
        grid_size=grid_size,
        x_range=x_range,
        kinematic_viscosity=kinematic_viscosity,
        cfl=CFL,
        real_t=real_t,
        num_threads=num_threads,
        time=time,
        with_forcing=with_forcing,
        filter_vorticity=filter_vorticity,
        poisson_solver_type=poisson_solver_type,
        **kwargs,
    )
    return flow_simulator

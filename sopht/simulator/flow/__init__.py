"""Flow module for the SophT Simulator."""

from .flow_simulators import FlowSimulator
from .flow_simulators_2d import create_unbounded_flow_simulator_2d
from .flow_simulators_3d import create_unbounded_flow_simulator_3d
from .navier_stokes_flow_simulators import (
    UnboundedNavierStokesFlowSimulator2D,
    UnboundedNavierStokesFlowSimulator3D,
)
from .passive_transport_flow_simulators import PassiveTransportFlowSimulator

__all__ = [
    "FlowSimulator",
    "PassiveTransportFlowSimulator",
    "UnboundedNavierStokesFlowSimulator2D",
    "UnboundedNavierStokesFlowSimulator3D",
    "create_unbounded_flow_simulator_2d",
    "create_unbounded_flow_simulator_3d",
]

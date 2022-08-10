__all__ = ["FlowForces"]
from sopht_simulator.immersed_body import ImmersedBodyFlowInteraction

from elastica import CosseratRod, NoForces, RigidBodyBase

from typing import Union


class FlowForces(NoForces):
    def __init__(self, body_flow_interactor: type(ImmersedBodyFlowInteraction)):
        super(NoForces, self).__init__()
        self.body_flow_interactor = body_flow_interactor

    def apply_forces(
        self, system: Union[type(CosseratRod), type(RigidBodyBase)], time=0.0
    ):
        self.body_flow_interactor.compute_flow_forces_and_torques()
        system.external_forces += self.body_flow_interactor.body_flow_forces
        system.external_torques += self.body_flow_interactor.body_flow_torques

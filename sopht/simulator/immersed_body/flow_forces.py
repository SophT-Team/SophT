from sopht.simulator.immersed_body import (
    CosseratRodFlowInteraction,
    RigidBodyFlowInteraction,
)
from elastica import CosseratRod, NoForces, RigidBodyBase


class FlowForces(NoForces):
    def __init__(
        self,
        body_flow_interactor: CosseratRodFlowInteraction | RigidBodyFlowInteraction,
    ) -> None:
        super(NoForces, self).__init__()
        self.body_flow_interactor = body_flow_interactor

    def apply_forces(
        self, system: CosseratRod | RigidBodyBase, time: float = 0.0
    ) -> None:
        self.body_flow_interactor.compute_flow_forces_and_torques()
        system.external_forces += self.body_flow_interactor.body_flow_forces
        system.external_torques += self.body_flow_interactor.body_flow_torques

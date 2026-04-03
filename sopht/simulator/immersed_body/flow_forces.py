from elastica import CosseratRod, NoForces, RigidBodyBase
from typing_extensions import override

from sopht.simulator.immersed_body.cosserat_rod.cosserat_rod_flow_interaction import (
    CosseratRodFlowInteraction,
)
from sopht.simulator.immersed_body.rigid_body.rigid_body_flow_interaction import (
    RigidBodyFlowInteraction,
)


class FlowForces(NoForces):
    @override
    def __init__(
        self,
        body_flow_interactor: CosseratRodFlowInteraction | RigidBodyFlowInteraction,
    ) -> None:
        super().__init__()
        self.body_flow_interactor = body_flow_interactor

    @override
    def apply_forces(self, system: CosseratRod | RigidBodyBase, time: float = 0.0) -> None:
        self.body_flow_interactor.compute_flow_forces_and_torques()
        system.external_forces += self.body_flow_interactor.body_flow_forces
        system.external_torques += self.body_flow_interactor.body_flow_torques

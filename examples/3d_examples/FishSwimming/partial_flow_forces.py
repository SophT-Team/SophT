from sopht.simulator.immersed_body import (
    CosseratRodFlowInteraction,
    RigidBodyFlowInteraction,
)
from elastica import CosseratRod, NoForces, RigidBodyBase
import numpy as np


class PartialFlowForces(NoForces):
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
        net_force = np.sum(self.body_flow_interactor.body_flow_forces, axis=1)
        acceleration = net_force / system.mass.sum()
        system.external_forces += system.mass * acceleration.reshape(-1, 1)

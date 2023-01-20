from sopht.simulator.immersed_body import (
    CosseratRodFlowInteraction,
    RigidBodyFlowInteraction,
)
from elastica import CosseratRod, NoForces, RigidBodyBase
import numpy as np


class FishFlowForces(NoForces):
    def __init__(
        self,
        body_flow_interactor: CosseratRodFlowInteraction | RigidBodyFlowInteraction,
        time_step,
    ) -> None:
        super(NoForces, self).__init__()
        self.body_flow_interactor = body_flow_interactor
        self.time_step = time_step

    def apply_forces(
        self, system: CosseratRod | RigidBodyBase, time: float = 0.0
    ) -> None:
        self.body_flow_interactor.compute_flow_forces_and_torques()
        # system.external_forces += self.body_flow_interactor.body_flow_forces
        # system.external_torques += self.body_flow_interactor.body_flow_torques

        net_force = np.sum(self.body_flow_interactor.body_flow_forces, axis=1)
        acceleration = net_force / system.mass.sum()
        velocity = (
            system.compute_velocity_center_of_mass() + acceleration * self.time_step
        )

        delta_position = velocity * self.time_step

        system.position_collection[0] += delta_position[0]
        system.position_collection[1] += delta_position[1]
        system.position_collection[2] += delta_position[2]

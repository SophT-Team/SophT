from sopht_simulator.cosserat_rod_support.CosseratRodFlowInteraction import (
    CosseratRodFlowInteraction,
)

from elastica.external_forces import NoForces
from elastica.typing import RodType


class FlowForces(NoForces):
    def __init__(self, cosserat_rod_flow_interactor: CosseratRodFlowInteraction):
        super(NoForces, self).__init__()
        self.cosserat_rod_flow_interactor = cosserat_rod_flow_interactor

    def apply_forces(self, system: RodType, time=0.0):
        self.cosserat_rod_flow_interactor.compute_flow_forces_and_torques()
        system.external_forces += (
            self.cosserat_rod_flow_interactor.cosserat_rod_flow_forces
        )
        system.external_torques += (
            self.cosserat_rod_flow_interactor.cosserat_rod_flow_torques
        )

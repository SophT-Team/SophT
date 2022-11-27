import numpy as np
import sopht.simulator as sps


class MockBodyFlowInteractor:
    def __init__(self):
        self.body_flow_forces = 0.0
        self.body_flow_torques = 0.0

    def compute_flow_forces_and_torques(self):
        self.body_flow_forces = 1.0
        self.body_flow_torques = 2.0


class MockRod:
    def __init__(self):
        self.external_forces = 3.0
        self.external_torques = 4.0


def test_flow_forces():
    body_flow_interactor = MockBodyFlowInteractor()
    flow_forcing = sps.FlowForces(body_flow_interactor=body_flow_interactor)
    assert body_flow_interactor is flow_forcing.body_flow_interactor

    rod = MockRod()
    flow_forcing.apply_forces(system=rod)
    # check mock systems for values below
    correct_external_forces = 1.0 + 3.0
    correct_external_torques = 2.0 + 4.0
    np.testing.assert_allclose(correct_external_forces, rod.external_forces)
    np.testing.assert_allclose(correct_external_torques, rod.external_torques)

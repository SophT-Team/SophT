import numpy as np
import elastica as ea
from typing import Type
from elastica._linalg import _batch_matvec, _batch_matrix_transpose, _batch_matmul


class FishTorques(ea.NoForces):
    """ """

    def __init__(
        self,
        virtual_rod: Type[ea.RigidBodyBase],
        ramp_up_time: float,
    ):
        """ """
        super(FishTorques, self).__init__()
        self.virtual_rod = virtual_rod
        self.ramp_up_time = ramp_up_time

    def apply_torques(self, rod: ea.CosseratRod, time: float = 0.0):

        if time <= self.ramp_up_time:
            factor = (1 + np.sin(np.pi * time / self.ramp_up_time - np.pi / 2)) / 2
        else:
            factor = 1.0

        # rod.external_torques[:] -= factor * _batch_matvec(
        #     rod.director_collection,
        #     _batch_matvec(
        #         _batch_matrix_transpose(rod.director_collection),
        #         self.virtual_rod.internal_torques[:],
        #     ),
        # )

        # Q_QT = _batch_matmul(
        #     rod.director_collection,
        #     _batch_matrix_transpose(self.virtual_rod.director_collection),
        # )
        #
        # rod.external_torques[:] -= factor * _batch_matvec(
        #     Q_QT, self.virtual_rod.internal_torques[:]
        # )

        rod.external_torques[:] -= factor * self.virtual_rod.internal_torques[:]
        rod.external_forces[:] -= factor * self.virtual_rod.internal_forces[:]

        # rod.rest_kappa[:] = factor * self.virtual_rod.kappa[:]
        # rod.rest_sigma[:] = factor * self.virtual_rod.sigma[:]

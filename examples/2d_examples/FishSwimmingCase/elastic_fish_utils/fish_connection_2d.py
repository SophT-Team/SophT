import elastica as ea
from elastica._linalg import _batch_matvec, _batch_matrix_transpose, _batch_matmul
import sopht.utils as spu


class FishConnection(ea.FreeJoint):
    """
    The FishConnection class connects a simulated fish to a virtual imposed fish
    using a spring force and torque.
    """

    def __init__(self, k: float) -> None:
        super(FishConnection, self).__init__(k, nu=0)
        self.k = k

    def apply_forces(
        self,
        rod_one: ea.CosseratRod,
        index_one: int,
        rod_two: ea.CosseratRod,
        index_two: int,
    ) -> None:

        y_axis_idx = spu.VectorField.y_axis_idx()
        fish_position = rod_one.position_collection.view()
        # TODO: spring defined between y positions figure out a better way.
        target_distance = (
            fish_position[y_axis_idx, :] - rod_two.position_collection[y_axis_idx, :]
        )
        spring_force = self.k * target_distance
        rod_one.external_forces[y_axis_idx, :] -= spring_force

    def apply_torques(
        self,
        rod_one: ea.CosseratRod,
        index_one: int,
        rod_two: ea.CosseratRod,
        index_two: int,
    ) -> None:

        Q_QT = _batch_matmul(
            rod_one.director_collection,
            _batch_matrix_transpose(rod_two.director_collection),
        )

        rod_one.external_torques[:] -= _batch_matvec(Q_QT, rod_two.internal_torques)

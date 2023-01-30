import numpy as np
import elastica as ea
from elastica._linalg import _batch_matvec, _batch_matrix_transpose, _batch_matmul
from elastica.interaction import elements_to_nodes_inplace


class FishConnection(ea.FreeJoint):
    def __init__(self, k):

        super(FishConnection, self).__init__(k, nu=0)
        self.k = k

    def apply_forces(self, rod_one, index_one, rod_two, index_two):

        fish_position = rod_one.position_collection.copy()

        # fish_position[0,:] -= fish_position[0,0]

        # head_correction = fish_position[0,0] - rod_two.position_collection[0,0]
        #
        # fish_position -= fish_position[:,0].reshape(3,1)
        #
        # virtual_position = rod_two.position_collection.copy()
        # virtual_position -= virtual_position[:,0].reshape(3,1)
        #
        # target_distance = fish_position[:] - virtual_position[:]
        #
        # np.round_(target_distance, 12, target_distance)
        #
        # self.spring_force = self.k * target_distance
        #
        # rod_one.external_forces[:] -= self.spring_force

        # TODO: spring defined between y positions figure out a better way.

        target_distance = fish_position[1, :] - rod_two.position_collection[1, :]

        self.spring_force = self.k * target_distance

        rod_one.external_forces[1, :] -= self.spring_force

        # fish_element_pos = 0.5 * (rod_one.position_collection[:,1:] + rod_one.position_collection[:,:-1])
        # virtual_element_pos = 0.5 * (rod_two.position_collection[:,1:] + rod_two.position_collection[:,:-1])
        #
        # fish_position_in_material_frame = _batch_matvec(rod_one.director_collection,  fish_element_pos)
        # virtual_rod_position_in_material_frame = _batch_matvec(rod_two.director_collection, virtual_element_pos)
        #
        # target_distance = fish_position_in_material_frame - virtual_rod_position_in_material_frame
        #
        # self.spring_force = _batch_matvec(_batch_matrix_transpose(rod_one.director_collection), self.k * target_distance)
        #
        # elements_to_nodes_inplace(-self.spring_force, rod_one.external_forces)

    def apply_torques(self, rod_one, index_one, rod_two, index_two):

        Q_QT = _batch_matmul(
            rod_one.director_collection,
            _batch_matrix_transpose(rod_two.director_collection),
        )

        rod_one.external_torques[:] -= _batch_matvec(Q_QT, rod_two.internal_torques[:])

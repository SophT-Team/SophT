import elastica as ea
import numpy as np
from typing import Optional, Type, Union
from numba import njit
from elastica._linalg import _batch_norm, _batch_cross


class CarlingFishBC(ea.ConstraintBase):
    """ """

    def __init__(self, period, wave_number, phase_shift, **kwargs):
        """ """
        super().__init__(**kwargs)
        self.period = period
        self.angular_frequency = 2 * np.pi / period
        self.wave_number = wave_number
        self.phase_shift = phase_shift
        rod = kwargs["_system"]

        n_elems = rod.n_elems
        self.n_nodes = n_elems + 1
        self.rest_lengths = rod.rest_lengths
        self.s = np.hstack((0, np.cumsum(self.rest_lengths)))
        self.base_length = self.rest_lengths.sum()
        self.positions = np.zeros((3, self.n_nodes))
        self.directors = np.zeros((3, 3, n_elems))
        self.directors[0, 1, :] = 1.0

    def constrain_values(
        self, rod: Union[Type[ea.RodBase], Type[ea.RigidBodyBase]], time: float
    ) -> None:

        self.constrain_fish_positions(
            self.n_nodes,
            time,
            self.s,
            self.base_length,
            self.rest_lengths,
            self.angular_frequency,
            self.wave_number,
            self.phase_shift,
            self.positions,
            rod.position_collection,
            self.directors,
            rod.director_collection,
        )

    @staticmethod
    @njit(cache=True)
    def constrain_fish_positions(
        n_nodes,
        time,
        s,
        base_length,
        rest_lengths,
        angular_frequency,
        wave_number,
        phase_shift,
        positions,
        position_collection,
        directors,
        director_collection,
    ):

        # Carling`s formula Eqn 45 in Gazzola JCP paper
        positions[1, :] = (
            0.125
            * base_length
            * (0.03125 + s)
            / 1.03125
            * np.sin(wave_number * s - angular_frequency * time + phase_shift)
        )

        # Compute delta y between nodes
        delta_y = positions[1, 1:] - positions[1, :-1]

        # Compute delta x between nodes using the base length
        delta_x = np.sqrt(rest_lengths**2 - delta_y**2)

        # Compute x positions
        for i in range(1, n_nodes):
            positions[0, i] = positions[0, i - 1] + delta_x[i - 1]

        position_collection[1, :] = positions[1, :]
        position_collection[0, :] = positions[0, :]
        position_collection[2, :] = positions[2, :]

        # Compute tangents
        tangents = positions[:, 1:] - positions[:, :-1]
        tangents /= _batch_norm(tangents)

        # Update directors
        directors[1, :, :] = _batch_cross(tangents, directors[0, :, :])
        directors[2, :, :] = tangents[:]

        director_collection[:] = directors[:]

    def constrain_rates(
        self, rod: Union[Type[ea.RodBase], Type[ea.RigidBodyBase]], time: float
    ) -> None:

        rod.omega_collection *= 0
        rod.velocity_collection *= 0

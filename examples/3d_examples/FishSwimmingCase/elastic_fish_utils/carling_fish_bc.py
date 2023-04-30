import elastica as ea
import numpy as np
from numba import njit
from elastica._linalg import _batch_norm, _batch_cross


class CarlingFishBC(ea.ConstraintBase):
    """
    This class implements the imposed boundary conditions corresponding to the swimming
    motion of a fish presented first by Carling et al., 1998.
    """

    def __init__(
        self,
        period: float,
        wave_number: float,
        phase_shift: float,
        ramp_up_time: float,
        fish_rod: ea.CosseratRod,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.period = period
        self.angular_frequency = 2 * np.pi / period
        self.wave_number = wave_number
        self.phase_shift = phase_shift
        self.ramp_up_time = ramp_up_time
        rod = kwargs["_system"]

        n_elems = rod.n_elems
        self.n_nodes = n_elems + 1
        self.rest_lengths = rod.rest_lengths
        self.s = np.hstack((0, np.cumsum(self.rest_lengths)))
        self.base_length = self.rest_lengths.sum()
        rod_dim = 3
        self.positions = np.zeros((rod_dim, self.n_nodes))
        self.directors = np.zeros((rod_dim, rod_dim, n_elems))
        self.directors[0, 2, :] = 1.0  # fixing normal of each rod element

        self.start_position = rod.position_collection[..., 0].reshape(3, 1).copy()
        self.fish_rod = fish_rod

    def constrain_values(self, rod: ea.CosseratRod, time: float) -> None:

        fish_head_y_position = self.fish_rod.compute_position_center_of_mass()[1]

        self.constrain_fish_positions(
            self.start_position,
            self.ramp_up_time,
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
            fish_head_y_position,
            rod.mass,
        )

    @staticmethod
    @njit(cache=True)
    def constrain_fish_positions(
        start_position: np.ndarray,
        ramp_up_time: float,
        n_nodes: int,
        time: float,
        s: np.ndarray,
        base_length: float,
        rest_lengths: np.ndarray,
        angular_frequency: float,
        wave_number: float,
        phase_shift: float,
        positions: np.ndarray,
        position_collection: np.ndarray,
        directors: np.ndarray,
        director_collection: np.ndarray,
        fish_head_y_position: float,
        mass: np.ndarray,
    ) -> None:

        # Carling`s formula Eqn 45 in Gazzola JCP paper
        if time <= ramp_up_time:
            factor = (1 + np.sin(np.pi * time / ramp_up_time - np.pi / 2)) / 2
        else:
            factor = 1.0

        x_axis_idx = 0
        y_axis_idx = 1
        positions[y_axis_idx, :] = (
            0.125
            * factor
            * base_length
            * (0.03125 + s)
            / 1.03125
            * np.sin(wave_number * s - angular_frequency * time + phase_shift)
        )

        # Compute delta y between nodes
        delta_y = positions[y_axis_idx, 1:] - positions[y_axis_idx, :-1]

        # Compute delta x between nodes using the base length
        delta_x = np.sqrt(rest_lengths**2 - delta_y**2)

        # Compute x positions
        for i in range(1, n_nodes):
            positions[x_axis_idx, i] = positions[x_axis_idx, i - 1] + delta_x[i - 1]

        position_collection[:] = positions[:]
        position_collection += start_position
        y_com = (position_collection[y_axis_idx, :] * mass).sum() / mass.sum()
        offset = fish_head_y_position - y_com
        position_collection[y_axis_idx, :] += offset

        # Compute tangents
        tangents = positions[:, 1:] - positions[:, :-1]
        tangents /= _batch_norm(tangents)

        # Update directors
        directors[1, :, :] = _batch_cross(tangents, directors[0, :, :])
        directors[2, :, :] = tangents[:]

        director_collection[:] = directors[:]

    def constrain_rates(self, rod: ea.CosseratRod, time: float) -> None:
        pass

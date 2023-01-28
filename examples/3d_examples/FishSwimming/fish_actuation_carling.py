import numpy as np
from elastica._linalg import _batch_norm, _batch_cross, _batch_dot
from elastica.rod.cosserat_rod import _compute_bending_twist_strains
from numba import njit
from scipy.interpolate import CubicSpline
import elastica as ea


class FishCurvatureCarling(ea.NoForces):
    """
    This class updates the rest curvature of the rod based on a cubic spine and a traveling wave function.

        Attributes
        ----------
        angular_frequency: float
            Angular frequency of traveling wave.
        wave_number: float
            Wave number of traveling wave.
        phase_shift: float
            Phase shift of traveling wave.
        ramp_up_time: float
            Applied muscle torques are ramped up until ramp up time.
        my_spline: numpy.ndarray
            1D (blocksize) array containing data with 'float' type. Generated spline.

    """

    def __init__(
        self,
        period,
        wave_number,
        phase_shift,
        rest_lengths,
        ramp_up_time,
    ):
        """

        Parameters
        ----------
        rest_lengths: float
            Rest length of the rod-like object.
        coefficients: nump.ndarray
            2D array containing data with 'float' type.
            Cubic spline coefficients.
        period: float
            Period of traveling wave.
        wave_number: float
            Wave number of traveling wave.
        phase_shift: float
            Phase shift of traveling wave.
        ramp_up_time: float
            Applied muscle torques are ramped up until ramp up time.

        """
        super(FishCurvatureCarling, self).__init__()

        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = wave_number
        self.phase_shift = phase_shift

        assert ramp_up_time > 0.0
        self.ramp_up_time = ramp_up_time

        # s is the non-dimensional position of inner nodes.
        self.n_elems = rest_lengths.shape[0]
        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1
        self.rest_lengths = rest_lengths
        self.base_length = np.sum(rest_lengths)
        self.rest_voronoi_lengths = 0.5 * (rest_lengths[1:] + rest_lengths[:-1])
        self.s = np.hstack((0, np.cumsum(rest_lengths)))
        self.s /= self.s[-1]

        self.positions = np.zeros((3, self.n_nodes))
        self.directors = np.zeros((3, 3, self.n_elems))
        self.directors[0, 2, :] = 1.0
        self.kappa = np.zeros((3, self.n_voronoi))

    def apply_torques(self, rod: ea.CosseratRod, time: float = 0.0):

        self.compute_curvature(
            self.n_nodes,
            time,
            self.s,
            self.base_length,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.angular_frequency,
            self.wave_number,
            self.phase_shift,
            self.ramp_up_time,
            self.positions,
            self.directors,
            self.kappa,
            rod.rest_kappa,
        )

    @staticmethod
    @njit(cache=True)
    def compute_curvature(
        n_nodes,
        time,
        s,
        base_length,
        rest_lengths,
        rest_voronoi_lengths,
        angular_frequency,
        wave_number,
        phase_shift,
        ramp_up_time,
        positions,
        directors,
        kappa,
        rest_kappa,
    ):
        # Ramp up the muscle torque
        if time <= ramp_up_time:
            factor = (1 + np.sin(np.pi * time / ramp_up_time - np.pi / 2)) / 2
        else:
            factor = 1.0

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

        # Compute tangents
        tangents = positions[:, 1:] - positions[:, :-1]
        tangents /= _batch_norm(tangents)

        # Update directors
        directors[1, :, :] = _batch_cross(tangents, directors[0, :, :])
        directors[2, :, :] = tangents[:]

        # Compute curvatures
        _compute_bending_twist_strains(directors, rest_voronoi_lengths, kappa)

        rest_kappa[1, :] = factor * kappa[0]

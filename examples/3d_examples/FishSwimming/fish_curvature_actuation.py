import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
import elastica as ea
from elastica.typing import SystemType, RodType


class FishCurvature(ea.NoForces):
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
        coefficients,
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
        super(FishCurvature, self).__init__()

        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = wave_number
        self.phase_shift = phase_shift

        assert ramp_up_time > 0.0
        self.ramp_up_time = ramp_up_time

        # s is the non-dimensional position of inner nodes.
        self.s = np.cumsum(rest_lengths)
        self.s /= self.s[-1]
        self.s = self.s[:-1]

        my_spline = CubicSpline(
            coefficients[0, :], coefficients[1, :], bc_type="natural"
        )

        self.my_spline = my_spline(self.s)

    def apply_torques(self, rod: RodType, time: float = 0.0):
        self.compute_curvature(
            time,
            self.my_spline,
            self.s,
            self.angular_frequency,
            self.wave_number,
            self.phase_shift,
            self.ramp_up_time,
            rod.rest_kappa,
        )

    @staticmethod
    @njit(cache=True)
    def compute_curvature(
        time,
        my_spline,
        s,
        angular_frequency,
        wave_number,
        phase_shift,
        ramp_up_time,
        rest_kappa,
    ):
        # Ramp up the muscle torque
        # factor = min(1.0, time / ramp_up_time)

        if time <= ramp_up_time:
            factor = (1 + np.sin(np.pi * time / ramp_up_time - np.pi / 2)) / 2
        else:
            factor = 1.0

        # From the node 1 to node nelem-1
        # Magnitude of the torque. Am = K(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
        curvature = (
            factor
            * my_spline
            * np.sin(angular_frequency * time - wave_number * s + phase_shift)
        )
        # Update rest curvature for actuation
        rest_kappa[1, :] = curvature[:]

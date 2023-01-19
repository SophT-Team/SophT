import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
import elastica as ea
from elastica.typing import SystemType, RodType


class FishCurvature(ea.NoForces):
    """
    This class applies muscle torques along the body. The applied muscle torques are
    treated as applied external forces. This class can apply muscle torques as a
    traveling wave with a beta spline or only as a traveling wave. For implementation
    details refer to Gazzola et. al. RSoS. (2018).

        Attributes
        ----------
        direction: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Muscle torque direction.
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
        base_length: float
            Rest length of the rod-like object.
        b_coeff: nump.ndarray
            1D array containing data with 'float' type.
            Beta coefficients for beta-spline.
        period: float
            Period of traveling wave.
        wave_number: float
            Wave number of traveling wave.
        phase_shift: float
            Phase shift of traveling wave.
        direction: numpy.ndarray
           1D (dim) array containing data with 'float' type. Muscle torque direction.
        ramp_up_time: float
            Applied muscle torques are ramped up until ramp up time.
        with_spline: boolean
            Option to use beta-spline.

        """
        super(FishCurvature, self).__init__()

        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = wave_number
        self.phase_shift = phase_shift

        assert ramp_up_time > 0.0
        self.ramp_up_time = ramp_up_time

        # s is the position of nodes on the rod, we go from node=1 to node=nelem-1, because there is no
        # torques applied by first and last node on elements. Reason is that we cannot apply torque in an
        # infinitesimal segment at the beginning and end of rod, because there is no additional element
        # (at element=-1 or element=n_elem+1) to provide internal torques to cancel out an external
        # torque. This coupled with the requirement that the sum of all muscle torques has
        # to be zero results in this condition.
        self.s = np.cumsum(rest_lengths)
        self.s /= self.s[-1]
        self.s = self.s[:-1]

        my_spline = CubicSpline(
            coefficients[0, :], coefficients[1, :], bc_type="natural"
        )

        self.my_spline = my_spline(self.s)

    def apply_torques(self, rod: RodType, time: float = 0.0):
        self.compute_muscle_torques(
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
    def compute_muscle_torques(
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
        factor = min(1.0, time / ramp_up_time)
        # From the node 1 to node nelem-1
        # Magnitude of the torque. Am = beta(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
        # There is an inconsistency with paper and Elastica cpp implementation. In paper sign in
        # front of wave number is positive, in Elastica cpp it is negative.
        curvature = (
            factor
            * my_spline
            * np.sin(angular_frequency * time - wave_number * s + phase_shift)
        )

        rest_kappa[1, :] = curvature[:]

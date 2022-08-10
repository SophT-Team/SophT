__doc__ = """
Oblique muscle implementation.
"""

from typing import Union

import numpy as np
from numba import njit

from comm.actuations.muscles.muscle import MuscleForce


class ObliqueMuscle(MuscleForce):
    """ObliqueMuscle."""

    def __init__(
        self,
        muscle_init_angle: float,
        ratio_muscle_position: np.ndarray,
        rotation_number: float,
        rest_muscle_area: np.ndarray,
        max_muscle_stress: Union[float, np.ndarray],
        **kwargs,
    ):
        """__init__.

        Parameters
        ----------
        muscle_init_angle : float
        ratio_muscle_position : np.ndarray
            shape: (3, n_element)
        rotation_number : float
        rest_muscle_area : np.ndarray
            shape: (n_element)
        max_muscle_stress : Union[float, np.ndarray]
            shape: (n_element)
        """
        n_elem = rest_muscle_area.shape[0]
        s = np.linspace(0, 1, n_elem + 1)
        s_muscle_position = (s[:-1] + s[1:]) / 2
        self.N = rotation_number  # TODO: Where is this used? If not outsideo of ObliqueMuscle, we shouldn't need self. Also we need better name than N.
        # fmt: off
        super().__init__(
            ratio_muscle_position=ratio_muscle_position * np.array([ \
                    np.cos(muscle_init_angle + 2 * np.pi * self.N * s_muscle_position),
                    np.sin(muscle_init_angle + 2 * np.pi * self.N * s_muscle_position),
                    np.zeros(n_elem)]),
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=max_muscle_stress,
            type_name="OM",
            **kwargs
        )
        # fmt: on

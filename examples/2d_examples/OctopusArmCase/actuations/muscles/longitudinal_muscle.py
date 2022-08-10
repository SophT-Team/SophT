__doc__ = """
Longitudinal muscle model definition.
"""

from typing import Union

import numpy as np
from numba import njit

from actuations.muscles.muscle import MuscleForce


class LongitudinalMuscle(MuscleForce):
    """LongitudinalMuscle"""

    def __init__(
        self,
        muscle_init_angle: float,
        ratio_muscle_position: np.ndarray,
        rest_muscle_area: np.ndarray,
        max_muscle_stress: Union[float, np.ndarray],
        **kwargs,
    ):
        """
        Initialize longitudinal muscle model.

        Parameters
        ----------
        muscle_init_angle : float
        ratio_muscle_position : np.ndarray
            shape: (3, n_element)
        rest_muscle_area : np.ndarray
            shape: (n_element)
        max_muscle_stress : Union[float, np.ndarray]
            shape: (n_element)
        """

        # FIXME: I'm not sure keeping ratio_muscle_position name after the operation is good idea.
        ratio_muscle_position = ratio_muscle_position * np.array(
            [[np.cos(muscle_init_angle)], [np.sin(muscle_init_angle)], [0]]
        )
        super().__init__(
            ratio_muscle_position=ratio_muscle_position,
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=max_muscle_stress,
            type_name="LM",
            **kwargs,
        )

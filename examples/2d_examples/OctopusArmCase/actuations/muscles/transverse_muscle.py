__doc__ = """
Transverse muscle implementation
"""

from typing import Union

import numpy as np
from numba import njit

from actuations.muscles.muscle import MuscleForce


class TransverseMuscle(MuscleForce):
    """TransverseMuscle."""

    def __init__(
        self,
        rest_muscle_area: np.ndarray,
        max_muscle_stress: Union[float, np.ndarray],
        **kwargs,
    ):
        """__init__.

        Parameters
        ----------
        rest_muscle_area : np.ndarray
            shape: (n_element)
        max_muscle_stress : Union[float, np.ndarray]
            shape: (n_element)
        """
        n_elem = rest_muscle_area.shape[0]
        super().__init__(
            ratio_muscle_position=np.zeros((3, n_elem)),
            rest_muscle_area=rest_muscle_area,
            max_muscle_stress=-max_muscle_stress,
            type_name="TM",
            **kwargs,
        )

    @staticmethod
    @njit(cache=True)
    def calculate_muscle_length(muscle_length: np.ndarray, muscle_strain: np.ndarray):
        """calculate_muscle_length.

        Parameters
        ----------
        muscle_length : np.ndarray
            shape: (n_element)
        muscle_strain : np.ndarray
            shape: (3, n_element)
        """
        blocksize = muscle_length.shape[0]
        for i in range(blocksize):
            # fmt: off
            muscle_length[i] = (
                1 / (muscle_strain[0, i] ** 2 +  \
                     muscle_strain[1, i] ** 2 +  \
                     muscle_strain[2, i] ** 2) ** 0.25 \
            )
            # fmt: on

import numpy as np
import sopht.utils as spu
from typing import Type, Union


def compute_lamb_oseen_vorticity(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    x_cm: float,
    y_cm: float,
    nu: float,
    gamma: float,
    t: float,
    real_t: Type,
) -> Union[float, np.ndarray]:
    """
    Compute Lamb-Oseen vorticity based on:
    x, y: Cartesian coordinates
    x_com: Vortex center x coordinate
    y_com: Vortex center y coordinate
    nu: kinematic viscosity
    gamma: Circulation
    t: Time
    real_t: precision
    For formula refer to
    https://en.wikipedia.org/wiki/Lamb%E2%80%93Oseen_vortex
    """
    return (
        gamma
        / (4 * real_t(np.pi) * nu * t)
        * np.exp(-((x - x_cm) ** 2 + (y - y_cm) ** 2) / (4 * nu * t))
    ).astype(real_t)


def compute_lamb_oseen_velocity(
    x: np.ndarray,
    y: np.ndarray,
    x_cm: float,
    y_cm: float,
    nu: float,
    gamma: float,
    t: float,
    real_t: Type,
) -> np.ndarray:

    """
    Compute Lamb-Oseen velocity based on:
    x, y: Cartesian coordinates
    x_com: Vortex center x coordinate
    y_com: Vortex center y coordinate
    nu: kinematic viscosity
    gamma: Circulation
    t: Time
    real_t: precision
    For formula refer to
    https://en.wikipedia.org/wiki/Lamb%E2%80%93Oseen_vortex
    """
    r = np.sqrt((x - x_cm) ** 2 + (y - y_cm) ** 2).astype(real_t)
    velocity_theta = (
        gamma / real_t(2 * np.pi * r) * (1 - np.exp(-(r**2) / (4 * nu * t)))
    ).astype(real_t)

    cos_theta = (x - x_cm) / r
    sin_theta = (y - y_cm) / r

    velocity_field = np.zeros((2, *x.shape), dtype=real_t)

    velocity_field[spu.VectorField.x_axis_idx()] = velocity_theta * (-sin_theta)
    velocity_field[spu.VectorField.y_axis_idx()] = velocity_theta * cos_theta

    return velocity_field

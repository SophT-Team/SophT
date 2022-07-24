import numpy as np


def compute_lamb_oseen_vorticity(x, y, x_cm, y_cm, nu, gamma, t, real_t):
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


def compute_lamb_oseen_velocity(x, y, x_cm, y_cm, nu, gamma, t, real_t):
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

    velocity_field[0] = velocity_theta * (-sin_theta)
    velocity_field[1] = velocity_theta * cos_theta

    return velocity_field

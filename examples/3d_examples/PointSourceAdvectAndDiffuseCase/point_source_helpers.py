import numpy as np


def compute_diffused_point_source_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    x_grid_cm: float,
    y_grid_cm: float,
    z_grid_cm: float,
    nu: float,
    point_mag: float,
    t: float,
    real_dtype: type,
) -> np.ndarray:
    """
    Compute diffused point source based on:
    x_grid, y_grid, z_grid: Cartesian coordinates
    x_grid_com: Source center X coordinate
    y_grid_com: Source center y_grid coordinate
    z_grid_com: Source center z_grid coordinate
    nu: kinematic viscosity
    point_mag: magnitude of forcing
    t: time
    real_dtype: precision
    For formula refer to
    https://www.damtp.cam.ac.uk/user/dbs26/1BMethods/GreensPDE.pdf
    """
    return (
        point_mag
        / (4 * real_dtype(np.pi) * nu * t) ** (3 / 2)
        * np.exp(
            -(
                (x_grid - x_grid_cm) ** 2
                + (y_grid - y_grid_cm) ** 2
                + (z_grid - z_grid_cm) ** 2
            )
            / (4 * nu * t)
        )
    ).astype(real_dtype)

import numpy as np

from sopht.utils.precision import get_test_tol


def compute_advection_diffusion_timestep(
    velocity_field,
    CFL,
    nu,
    dx,
    precision="single",
    dt_prefac=1,
):
    """Compute stable timestep based on advection and diffusion limits."""
    # This may need a numba or pystencil version
    velocity_mag_field = np.sqrt(velocity_field[0] ** 2 + velocity_field[1] ** 2)
    dt = min(
        CFL * dx / (np.amax(velocity_mag_field) + get_test_tol(precision)),
        dx**2 / 4 / nu,
    )
    return dt * dt_prefac

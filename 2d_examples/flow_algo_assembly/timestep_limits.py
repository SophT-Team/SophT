import numpy as np


def compute_advection_diffusion_timestep(velocity_field, CFL, nu, dx):
    """Compute stable timestep based on advection and diffusion limits."""
    # This may need a numba or pystencil version
    velocity_mag_field = np.sqrt(velocity_field[0] ** 2 + velocity_field[1] ** 2)
    dt = min(CFL * dx / np.amax(velocity_mag_field), dx**2 / 4 / nu)
    return dt

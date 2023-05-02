import logging
import numpy as np
from elastica.utils import MaxDimension, Tolerance
import elastica as ea


def create_fish_geometry(rest_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    Compute the width and height of a fish geometry given an array of rest lengths.

    Parameters
    ----------
    rest_lengths : np.ndarray
        Array of rest lengths for each element in the fish.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing the width and height arrays, respectively.

    Notes
    -----
    This function uses a piecewise function to compute the width of the fish at
    each node along its length. The function is determined by the position along
    the length of the fish, with different formulas for different regions of the length.
    The height of the fish is computed using a formula that depends on the distance from
    a specific point along the length of the fish.

    """

    n_elements = rest_lengths.shape[0]
    s_node = np.zeros(n_elements + 1)
    s_node[1:] = np.cumsum(rest_lengths)
    s_node /= s_node[-1]
    s = 0.5 * (s_node[1:] + s_node[:-1])

    # Compute the width of the fish along its length
    base_length = rest_lengths.sum()
    s_b = 0.04 * base_length
    s_t = 0.95 * base_length
    w_h = 0.04 * base_length
    w_t = 0.01 * base_length

    width = np.zeros(n_elements)
    for i in range(n_elements):
        if s[i] >= 0 and s[i] <= s_b:
            width[i] = np.sqrt(2 * w_h * s[i] - s[i] ** 2)
        elif s_b <= s[i] and s[i] <= s_t:
            width[i] = w_h - (w_h - w_t) * ((s[i] - s_b) / (s_t - s_b)) ** 2

        elif s_t <= s[i] and s[i] <= base_length:
            width[i] = w_t * (base_length - s[i]) / (base_length - s_t)

    # Compute the height of the fish
    a = 0.51 * base_length
    b = 0.08 * base_length
    height = b * np.sqrt(1 - ((s - a) / a) ** 2)

    return width, height


def update_rod_for_fish_geometry(
    density: float, youngs_modulus: float, rod: ea.CosseratRod
) -> None:
    """
    Update the Cosserat rod object with properties for fish-like geometry.

    Parameters
    ----------
    density : float
        The density of the material.
    youngs_modulus : float
        The Young's modulus of the material.
    rod : ea.CosseratRod
        The Cosserat rod object to update.

    Returns
    -------
    None

    Notes
    -----
    This function computes mass, mass moment of inertia, radius, shear matrix,
    bend matrix, and mass of elements of a Cosserat rod object based on a fish-like geometry.

    The Cosserat rod object is updated in-place.

    Raises
    ------
    AssertionError
        If the rank of mass moment of inertia matrix is not equal to MaxDimension.value().

    Warnings
    --------
    If the mass moment of inertia matrix is smaller than the tolerance value, a warning is
    logged.
    """

    width, height = create_fish_geometry(rod.rest_lengths)

    n_elements = rod.n_elems
    rest_lengths = rod.rest_lengths.copy()

    # Compute mass, mass moment of inertia, radius
    A0 = np.pi * width * height
    volume = A0 * rest_lengths

    # Second moment of inertia of an ellipse
    I0_1 = np.pi / 4 * width**3 * height
    I0_2 = np.pi / 4 * width * height**3
    I0_3 = np.pi / 4 * width * height * (width**2 + height**2)
    I0 = np.array([I0_1, I0_2, I0_3]).transpose()
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )

    mass_second_moment_of_inertia_temp = np.einsum(
        "ij,i->ij", I0, density * rest_lengths
    )

    for i in range(n_elements):
        np.fill_diagonal(
            mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[i, :],
        )
    # sanity check of mass second moment of inertia
    if (mass_second_moment_of_inertia < Tolerance.atol()).all():
        log = logging.getLogger()
        log.warning(
            "Mass moment of inertia matrix smaller than tolerance."
            "Please check provided radius, density and length."
        )

    # Inverse of second moment of inertia
    inv_mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements)
    )
    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert (
            np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i])
            == MaxDimension.value()
        )
        inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            mass_second_moment_of_inertia[..., i]
        )

    rod.inv_mass_second_moment_of_inertia[:] = inv_mass_second_moment_of_inertia[:]

    # Shear/Stretch matrix
    shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.5))

    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    shear_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            shear_matrix[..., i],
            [
                alpha_c * shear_modulus * A0[i],
                alpha_c * shear_modulus * A0[i],
                youngs_modulus * A0[i],
            ],
        )

    rod.shear_matrix[:] = shear_matrix[:]

    # Bend/Twist matrix
    bend_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )
    for i in range(n_elements):
        np.fill_diagonal(
            bend_matrix[..., i],
            [
                youngs_modulus * I0_1[i],
                youngs_modulus * I0_2[i],
                shear_modulus * I0_3[i],
            ],
        )
    for i in range(0, MaxDimension.value()):
        assert np.all(
            bend_matrix[i, i, :] > Tolerance.atol()
        ), " Bend matrix has to be greater than 0."

    # Compute bend matrix in Voronoi Domain
    bend_matrix = (
        bend_matrix[..., 1:] * rest_lengths[1:]
        + bend_matrix[..., :-1] * rest_lengths[0:-1]
    ) / (rest_lengths[1:] + rest_lengths[:-1])

    rod.bend_matrix[:] = bend_matrix[:]

    # Compute mass of elements
    mass = np.zeros(n_elements + 1)
    mass[:-1] += 0.5 * density * volume
    mass[1:] += 0.5 * density * volume

    rod.mass[:] = mass[:]

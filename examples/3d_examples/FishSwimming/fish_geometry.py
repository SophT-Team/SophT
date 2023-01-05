import numpy as np
from scipy.interpolate import interp1d
from elastica._linalg import _batch_norm, _batch_cross
from elastica._rotations import _inv_rotate
from elastica.utils import MaxDimension, Tolerance


def create_fish_geometry(rod):
    """
    TODO docs
    Parameters
    ----------
    rod

    Returns
    -------

    """

    n_elements = rod.n_elems
    rest_lengths = rod.rest_lengths.copy()

    element_positions = 0.5 * (
        rod.position_collection[:, 1:] + rod.position_collection[:, :-1]
    )
    s = _batch_norm(element_positions)

    # Compute the width of the fish along its length
    base_length = rest_lengths.sum()
    s_b = 0.04 * base_length
    s_t = 0.95 * base_length
    w_h = 0.04 * base_length
    w_t = 0.01 * base_length

    width = np.zeros((n_elements))
    for i in range(n_elements):
        if s[i] >= 0 and s[i] <= s_b:
            width[i] = np.sqrt(2 * w_h * s[i] - s[i] ** 2)
        elif s_b <= s[i] and s[i] <= s_t:
            width[i] = w_h - (w_h - w_t) * (s[i] - s_b) / (s_t - s_b)

        elif s_t <= s[i] and s[i] <= base_length:
            width[i] = w_t * (base_length - s[i]) / (base_length - s_t)

    # Compute the height of the fish
    a = 0.51 * base_length
    b = 0.08 * base_length
    height = b * np.sqrt(1 - ((s - a) / a) ** 2)

    return width, height


def update_rod_for_fish_geometry(density, youngs_modulus, rod):
    """
    TODO: docs

    Parameters
    ----------
    density
    youngs_modulus
    rod

    Returns
    -------

    """

    width, height = create_fish_geometry(rod)

    n_elements = rod.n_elems
    rest_lengths = rod.rest_lengths.copy()

    # Compute mass, mass moment of inertia, radius
    A0 = 4 * np.pi * width * height
    volume = A0 * rest_lengths

    # Second moment of inertia of an ellipse
    I0_1 = np.pi / 4 * width * height**3
    I0_2 = np.pi / 4 * width**3 * height
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
        message = "Mass moment of inertia matrix smaller than tolerance, please check provided radius, density and length."
        log.warning(message)

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

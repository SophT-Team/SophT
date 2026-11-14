"""Kernels for penalising field boundary in 2D."""

from collections.abc import Callable

import numpy as np


def gen_penalise_field_boundary_pyst_kernel_2d(
    width: int,
    dx: float,
    x_grid_field: np.ndarray,
    y_grid_field: np.ndarray,
    real_t: type,
    num_threads: bool | int = False,  # noqa: ARG001
    grid_size: tuple[int, int] | None = None,  # noqa: ARG001
) -> Callable:
    # TODO: expand docs
    """2D penalise field boundary kernel generator."""
    if not isinstance(width, int) or width < 0:
        msg = "Invalid zone width, must be a non-negative integer"
        raise ValueError(msg)

    if width == 0:
        # bypass option to prevent penalisation, done this way since by
        # default to avoid artifacts one must use penalisation...
        return lambda field: None  # noqa: ARG005

    # Change 0.1.0:
    #
    # All pystencils JIT is replaced with generic NumPy functions.
    # Penalizing the boundaries is a relatively complex operation in terms
    # of homogeneity and branching. Compiling multiple pystencils functions
    # and invoking them one by one is clumsy and pays the overhead of OpemMP
    # thread launches multiple times. Savings gained from compiled code do
    # not justify these complications.
    #
    # The API is kept for compatibility for now, but should be cleaned up in
    # the future.
    # TODO: clean up API signature

    x_grid_field_start = x_grid_field[0, 0]
    y_grid_field_start = y_grid_field[0, 0]
    x_grid_field_end = x_grid_field[0, -1]
    y_grid_field_end = y_grid_field[-1, 0]

    sine_prefactor = (np.pi / 2) / (width * dx)
    x_front_weight = np.sin(sine_prefactor * (x_grid_field[:, :width] - x_grid_field_start)).astype(
        real_t
    )
    y_front_weight = np.sin(sine_prefactor * (y_grid_field[:width, :] - y_grid_field_start)).astype(
        real_t
    )
    x_back_weight = np.sin(sine_prefactor * (x_grid_field_end - x_grid_field[:, -width:])).astype(
        real_t
    )
    y_back_weight = np.sin(sine_prefactor * (y_grid_field_end - y_grid_field[-width:, :])).astype(
        real_t
    )

    def penalise_field_boundary_pyst_kernel_2d(field: np.ndarray) -> None:
        """2D penalise field boundary kernel.

        Penalises field on the boundaries in a sine wave fashion
        in the given width in X and Y direction
        field: field to be penalised
        """

        field[:, :width] = field[:, width - 1, np.newaxis] * x_front_weight
        field[:, -width:] = field[:, -width, np.newaxis] * x_back_weight
        field[:width, :] = field[width - 1, :] * y_front_weight
        field[-width:, :] = field[-width, :] * y_back_weight

    return penalise_field_boundary_pyst_kernel_2d

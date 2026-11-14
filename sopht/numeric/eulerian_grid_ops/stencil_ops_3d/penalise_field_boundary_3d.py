"""Kernels for penalising field boundary in 3D."""

from collections.abc import Callable
from typing import Literal

import numpy as np

import sopht.utils as spu


def gen_penalise_field_boundary_pyst_kernel_3d(
    width: int,
    dx: float,
    x_grid_field: np.ndarray,
    y_grid_field: np.ndarray,
    z_grid_field: np.ndarray,
    real_t: type,
    num_threads: bool | int = False,  # noqa: ARG001
    grid_size: tuple[int, int, int] | None = None,  # noqa: ARG001
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO: expand docs
    """3D penalise field boundary kernel generator."""
    if not isinstance(width, int) or width < 0:
        msg = "Invalid width for boundary zone, must be a non-negative integer"
        raise ValueError(msg)

    if width == 0:
        # bypass option to prevent penalisation, done this way since by
        # default to avoid artifacts one must use penalisation...
        match field_type:
            case "scalar":
                return lambda field: None  # noqa: ARG005
            case "vector":
                return lambda vector_field: None  # noqa: ARG005
            case _:
                msg = "Invalid field type"
                raise ValueError(msg)

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

    x_grid_field_start = x_grid_field[0, 0, 0]
    y_grid_field_start = y_grid_field[0, 0, 0]
    z_grid_field_start = z_grid_field[0, 0, 0]
    x_grid_field_end = x_grid_field[0, 0, -1]
    y_grid_field_end = y_grid_field[0, -1, 0]
    z_grid_field_end = z_grid_field[-1, 0, 0]

    sine_prefactor = (np.pi / 2) / (width * dx)
    x_front_weight = np.sin(
        sine_prefactor * (x_grid_field[:, :, :width] - x_grid_field_start)
    ).astype(real_t)
    y_front_weight = np.sin(
        sine_prefactor * (y_grid_field[:, :width, :] - y_grid_field_start)
    ).astype(real_t)
    z_front_weight = np.sin(
        sine_prefactor * (z_grid_field[:width, :, :] - z_grid_field_start)
    ).astype(real_t)
    x_back_weight = np.sin(
        sine_prefactor * (x_grid_field_end - x_grid_field[:, :, -width:])
    ).astype(real_t)
    y_back_weight = np.sin(
        sine_prefactor * (y_grid_field_end - y_grid_field[:, -width:, :])
    ).astype(real_t)
    z_back_weight = np.sin(
        sine_prefactor * (z_grid_field_end - z_grid_field[-width:, :, :])
    ).astype(real_t)

    def penalise_field_boundary_pyst_kernel_3d(field: np.ndarray) -> None:
        """3D penalise field boundary kernel.

        Penalises field on the boundaries in a sine wave fashion
        in the given width in X, Y and Z direction
        field: field to be penalised
        """
        field[:, :, :width] = field[:, :, width - 1, np.newaxis] * x_front_weight
        field[:, :, -width:] = field[:, :, -width, np.newaxis] * x_back_weight
        field[:, :width, :] = field[:, width - 1, np.newaxis, :] * y_front_weight
        field[:, -width:, :] = field[:, -width, np.newaxis, :] * y_back_weight
        field[:width, :, :] = field[width - 1, :, :] * z_front_weight
        field[-width:, :, :] = field[-width, :, :] * z_back_weight

    match field_type:
        case "scalar":
            return penalise_field_boundary_pyst_kernel_3d
        case "vector":
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()
            z_axis_idx = spu.VectorField.z_axis_idx()

            def penalise_vector_field_boundary_pyst_kernel_3d(
                vector_field: np.ndarray,
            ) -> None:
                """3D penalise vector field boundary kernel.

                Penalises vector field on the boundaries in a sine wave
                fashion in the given width in X, Y and Z direction
                vector_field: vector field to be penalised
                """
                penalise_field_boundary_pyst_kernel_3d(
                    field=vector_field[x_axis_idx],
                )
                penalise_field_boundary_pyst_kernel_3d(
                    field=vector_field[y_axis_idx],
                )
                penalise_field_boundary_pyst_kernel_3d(
                    field=vector_field[z_axis_idx],
                )

            return penalise_vector_field_boundary_pyst_kernel_3d
        case _:
            msg = "Invalid field type"
            raise ValueError(msg)

"""Kernels for Brinkmann penalisation in 2D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.utils as spu
from typing import Callable, Literal


def gen_brinkmann_penalise_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    """Brinkmann penalisation 2D kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "2D"
    )

    @ps.kernel
    def _brinkmann_penalise_stencil_2d():
        penalised_field, field, char_field, penalty_field = ps.fields(
            f"penalised_field, field, char_field, penalty_field : {pyst_dtype}[{grid_info}]"
        )
        penalty_factor = sp.symbols("penalty_factor")
        penalised_field[0, 0] @= (
            field[0, 0] + penalty_factor * char_field[0, 0] * penalty_field[0, 0]
        ) / (1 + penalty_factor * char_field[0, 0])

    brinkmann_penalise_pyst_kernel_2d = ps.create_kernel(
        _brinkmann_penalise_stencil_2d, config=kernel_config
    ).compile()
    match field_type:
        case "scalar":
            return brinkmann_penalise_pyst_kernel_2d
        case "vector":
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()

            def brinkmann_penalise_vector_field_pyst_kernel_2d(
                penalised_vector_field: np.ndarray,
                penalty_factor: float,
                char_field: np.ndarray,
                penalty_vector_field: np.ndarray,
                vector_field: np.ndarray,
            ) -> None:
                """Brinkmann penalises a vector field in 2D."""
                brinkmann_penalise_pyst_kernel_2d(
                    penalised_field=penalised_vector_field[x_axis_idx],
                    penalty_factor=penalty_factor,
                    char_field=char_field,
                    penalty_field=penalty_vector_field[x_axis_idx],
                    field=vector_field[x_axis_idx],
                )
                brinkmann_penalise_pyst_kernel_2d(
                    penalised_field=penalised_vector_field[y_axis_idx],
                    penalty_factor=penalty_factor,
                    char_field=char_field,
                    penalty_field=penalty_vector_field[y_axis_idx],
                    field=vector_field[y_axis_idx],
                )

            return brinkmann_penalise_vector_field_pyst_kernel_2d

        case _:
            raise ValueError("Invalid field type")


def gen_brinkmann_penalise_vs_fixed_val_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    """Brinkmann penalisation against fixed val, 2D kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "2D"
    )

    @ps.kernel
    def _brinkmann_penalise_vs_fixed_val_stencil_2d():
        penalised_field, field, char_field = ps.fields(
            f"penalised_field, field, char_field : {pyst_dtype}[{grid_info}]"
        )
        penalty_factor, penalty_val = sp.symbols("penalty_factor, penalty_val")
        penalised_field[0, 0] @= (
            field[0, 0] + penalty_factor * char_field[0, 0] * penalty_val
        ) / (1 + penalty_factor * char_field[0, 0])

    brinkmann_penalise_vs_fixed_val_pyst_kernel_2d = ps.create_kernel(
        _brinkmann_penalise_vs_fixed_val_stencil_2d, config=kernel_config
    ).compile()
    match field_type:
        case "scalar":
            return brinkmann_penalise_vs_fixed_val_pyst_kernel_2d
        case "vector":
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()

            def brinkmann_penalise_vector_field_vs_fixed_val_pyst_kernel_2d(
                penalised_vector_field: np.ndarray,
                penalty_factor: float,
                char_field: np.ndarray,
                penalty_val: tuple[int, int],
                vector_field: np.ndarray,
            ) -> None:
                """Brinkmann penalises a vector field against fixed values in 2D."""
                brinkmann_penalise_vs_fixed_val_pyst_kernel_2d(
                    penalised_field=penalised_vector_field[x_axis_idx],
                    penalty_factor=penalty_factor,
                    char_field=char_field,
                    penalty_val=penalty_val[x_axis_idx],
                    field=vector_field[x_axis_idx],
                )
                brinkmann_penalise_vs_fixed_val_pyst_kernel_2d(
                    penalised_field=penalised_vector_field[y_axis_idx],
                    penalty_factor=penalty_factor,
                    char_field=char_field,
                    penalty_val=penalty_val[y_axis_idx],
                    field=vector_field[y_axis_idx],
                )

            return brinkmann_penalise_vector_field_vs_fixed_val_pyst_kernel_2d
        case _:
            raise ValueError("Invalid field type")

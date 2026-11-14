"""Kernels for Brinkmann penalisation in 2D."""

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import pystencils as ps
import sympy as sp

import sopht.utils as spu


def gen_brinkmann_penalise_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int] | None = None,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    """Brinkmann penalisation 2D kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = spu.get_grid_info(grid_size, 2, "scalar")

    def penalised_field_expr(
        field: ps.Field, penalty_factor: sp.Expr, char_field: ps.Field, penalty_field: ps.Field
    ) -> sp.Expr:
        return (field.center + penalty_factor * char_field.center * penalty_field.center) / (
            1 + penalty_factor * char_field.center
        )

    match field_type:
        case "scalar":
            penalised_field, field, char_field, penalty_field = ps.fields(
                f"penalised_field, field, char_field, penalty_field : {pyst_dtype}[{grid_info}]"
            )
            penalty_factor = sp.symbols("penalty_factor")
            eqn = ps.Assignment(
                penalised_field.center,
                penalised_field_expr(field, penalty_factor, char_field, penalty_field),
            )
            return ps.create_kernel(eqn, config=kernel_config).compile()

        case "vector":
            penalised_field_x, field_x, penalty_field_x = ps.fields(
                f"penalised_field_x, field_x, penalty_field_x : {pyst_dtype}[{grid_info}]"
            )
            penalised_field_y, field_y, penalty_field_y = ps.fields(
                f"penalised_field_y, field_y, penalty_field_y : {pyst_dtype}[{grid_info}]"
            )
            char_field = cast("ps.Field", ps.fields(f"char_field : {pyst_dtype}[{grid_info}]"))
            penalty_factor = sp.symbols("penalty_factor")
            eqn = [
                ps.Assignment(
                    penalised_field_x.center,
                    penalised_field_expr(field_x, penalty_factor, char_field, penalty_field_x),
                ),
                ps.Assignment(
                    penalised_field_y.center,
                    penalised_field_expr(field_y, penalty_factor, char_field, penalty_field_y),
                ),
            ]
            _vector_kernel = ps.create_kernel(eqn, config=kernel_config).compile()
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
                _vector_kernel(
                    penalised_field_x=penalised_vector_field[x_axis_idx],
                    penalised_field_y=penalised_vector_field[y_axis_idx],
                    penalty_field_x=penalty_vector_field[x_axis_idx],
                    penalty_field_y=penalty_vector_field[y_axis_idx],
                    field_x=vector_field[x_axis_idx],
                    field_y=vector_field[y_axis_idx],
                    penalty_factor=penalty_factor,
                    char_field=char_field,
                )

            return brinkmann_penalise_vector_field_pyst_kernel_2d

        case _:
            msg = "Invalid field type"
            raise ValueError(msg)


def gen_brinkmann_penalise_vs_fixed_val_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int] | None = None,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    """Brinkmann penalisation against fixed val, 2D kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = spu.get_grid_info(grid_size, 2, "scalar")

    def penalised_field_expr(
        field: ps.Field, penalty_factor: sp.Expr, char_field: ps.Field, penalty_val: sp.Expr
    ) -> sp.Expr:
        return (field.center + penalty_factor * penalty_val * char_field.center) / (
            1 + penalty_factor * char_field.center
        )

    match field_type:
        case "scalar":
            penalised_field, field, char_field = ps.fields(
                f"penalised_field, field, char_field : {pyst_dtype}[{grid_info}]"
            )
            penalty_factor, penalty_val = sp.symbols("penalty_factor, penalty_val")
            eqn = ps.Assignment(
                penalised_field.center,
                penalised_field_expr(field, penalty_factor, char_field, penalty_val),
            )
            return ps.create_kernel(eqn, config=kernel_config).compile()

        case "vector":
            penalised_field_x, penalised_field_y, field_x, field_y, char_field = ps.fields(
                f"penalised_field_x, penalised_field_y, field_x, field_y, char_field "
                f": {pyst_dtype}[{grid_info}]"
            )
            penalty_factor, penalty_val_x, penalty_val_y = sp.symbols(
                "penalty_factor, penalty_val_x, penalty_val_y"
            )
            eqn = [
                ps.Assignment(
                    penalised_field_x.center,
                    penalised_field_expr(field_x, penalty_factor, char_field, penalty_val_x),
                ),
                ps.Assignment(
                    penalised_field_y.center,
                    penalised_field_expr(field_y, penalty_factor, char_field, penalty_val_y),
                ),
            ]
            _vector_kernel = ps.create_kernel(eqn, config=kernel_config).compile()

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
                _vector_kernel(
                    penalised_field_x=penalised_vector_field[x_axis_idx],
                    penalised_field_y=penalised_vector_field[y_axis_idx],
                    penalty_val_x=penalty_val[x_axis_idx],
                    penalty_val_y=penalty_val[y_axis_idx],
                    field_x=vector_field[x_axis_idx],
                    field_y=vector_field[y_axis_idx],
                    penalty_factor=penalty_factor,
                    char_field=char_field,
                )

            return brinkmann_penalise_vector_field_vs_fixed_val_pyst_kernel_2d

        case _:
            msg = "Invalid field type"
            raise ValueError(msg)

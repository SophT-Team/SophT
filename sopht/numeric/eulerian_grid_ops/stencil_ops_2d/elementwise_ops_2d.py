"""Kernels for elementwise operations in 2D."""

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import pystencils as ps
import sympy as sp

import sopht.utils as spu


def gen_elementwise_sum_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int] | None = None,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO: expand docs
    """2D elementwise sum kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    grid_info = spu.get_grid_info(grid_size, 2, field_type)

    sum_field, field_1, field_2 = ps.fields(
        f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
    )
    eqn = ps.Assignment(sum_field.center, field_1.center + field_2.center)

    return ps.create_kernel(eqn, config=kernel_config).compile()


def gen_set_fixed_val_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int] | None = None,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO: expand docs
    """2D set field to fixed value kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = spu.get_grid_info(grid_size, 2, "scalar")

    match field_type:
        case "scalar":
            field = cast("ps.Field", ps.fields(f"field : {pyst_dtype}[{grid_info}]"))
            fixed_val = sp.symbols("fixed_val")
            eqn = ps.Assignment(field.center, fixed_val)
            return ps.create_kernel(eqn, config=kernel_config).compile()

        case "vector":
            field_x, field_y = ps.fields(f"field_x, field_y : {pyst_dtype}[{grid_info}]")
            fixed_val_x, fixed_val_y = sp.symbols("fixed_val_x, fixed_val_y")
            eqn = [
                ps.Assignment(field_x.center, fixed_val_x),
                ps.Assignment(field_y.center, fixed_val_y),
            ]
            vector_kernel = ps.create_kernel(eqn, config=kernel_config).compile()

            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()

            def vector_field_set_fixed_val_pyst_kernel_2d(
                vector_field: np.ndarray,
                fixed_vals: tuple[float, float],
            ) -> None:
                """Set vector field to fixed value in 2D"""
                vector_kernel(
                    field_x=vector_field[x_axis_idx],
                    field_y=vector_field[y_axis_idx],
                    fixed_val_x=fixed_vals[x_axis_idx],
                    fixed_val_y=fixed_vals[y_axis_idx],
                )

            return vector_field_set_fixed_val_pyst_kernel_2d

        case _:
            msg = "Invalid field type"
            raise ValueError(msg)


def gen_elementwise_copy_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int] | None = None,
) -> Callable:
    # TODO: expand docs
    """2D elementwise copy one field to another kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = spu.get_grid_info(grid_size, 2, "scalar")

    field, rhs_field = ps.fields(f"field, rhs_field : {pyst_dtype}[{grid_info}]")
    eqn = ps.Assignment(field.center, rhs_field.center)
    return ps.create_kernel(eqn, config=kernel_config).compile()


def gen_elementwise_complex_product_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int] | None = None,
) -> Callable:
    # TODO: expand docs
    """2D elementwise complex number product kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = spu.get_grid_info(grid_size, 2, "scalar")

    product_field_real, field_1_real, field_2_real = ps.fields(
        f"product_field_real, field_1_real, field_2_real : {pyst_dtype}[{grid_info}]"
    )
    product_field_imag, field_1_imag, field_2_imag = ps.fields(
        f"product_field_imag, field_1_imag, field_2_imag : {pyst_dtype}[{grid_info}]"
    )
    eqn = [
        ps.Assignment(
            product_field_real.center,
            field_1_real.center * field_2_real.center - field_1_imag.center * field_2_imag.center,
        ),
        ps.Assignment(
            product_field_imag.center,
            field_1_real.center * field_2_imag.center + field_1_imag.center * field_2_real.center,
        ),
    ]

    _elementwise_complex_product_pyst_kernel_2d = ps.create_kernel(
        eqn, config=kernel_config
    ).compile()

    def elementwise_complex_product_pyst_kernel_2d(
        product_field: np.ndarray, field_1: np.ndarray, field_2: np.ndarray
    ) -> None:
        """2D Elementwise complex number product."""
        # complex numbers had compilation issues in pystencils :/
        _elementwise_complex_product_pyst_kernel_2d(
            product_field_real=product_field.real,
            product_field_imag=product_field.imag,
            field_1_real=field_1.real,
            field_1_imag=field_1.imag,
            field_2_real=field_2.real,
            field_2_imag=field_2.imag,
        )

    return elementwise_complex_product_pyst_kernel_2d


def gen_set_fixed_val_at_boundaries_pyst_kernel_2d(
    real_t: type,
    width: int,
    num_threads: bool | int = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO: expand docs
    """2D set field to fixed value at boundaries kernel generator."""
    if not isinstance(width, int) or width <= 0:
        msg = "Invalid width for boundary zone, must be a positive integer"
        raise ValueError(msg)

    set_fixed_val_kernel_2d = gen_set_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        num_threads=num_threads,
        field_type=field_type,
    )
    match field_type:
        case "scalar":

            def set_fixed_val_at_boundaries_pyst_kernel_2d(
                field: np.ndarray, fixed_val: float
            ) -> None:
                """Set field to fixed value at boundaries.

                Set a 2D scalar field to a fixed value at boundaries in zone: width,
                used for clearing boundary noise.
                """
                set_fixed_val_kernel_2d(field=field[:width, :], fixed_val=fixed_val)
                set_fixed_val_kernel_2d(field=field[-width:, :], fixed_val=fixed_val)
                set_fixed_val_kernel_2d(field=field[:, :width], fixed_val=fixed_val)
                set_fixed_val_kernel_2d(field=field[:, -width:], fixed_val=fixed_val)

            return set_fixed_val_at_boundaries_pyst_kernel_2d

        case "vector":

            def vector_field_set_fixed_val_at_boundaries_pyst_kernel_2d(
                vector_field: np.ndarray, fixed_vals: tuple[int, int]
            ) -> None:
                """Set field to fixed value at boundaries.

                Set a 2D vector field to a fixed value at boundaries in zone: width,
                used for clearing boundary noise.
                """
                set_fixed_val_kernel_2d(
                    vector_field=vector_field[:, :width, :], fixed_vals=fixed_vals
                )
                set_fixed_val_kernel_2d(
                    vector_field=vector_field[:, -width:, :], fixed_vals=fixed_vals
                )
                set_fixed_val_kernel_2d(
                    vector_field=vector_field[:, :, :width], fixed_vals=fixed_vals
                )
                set_fixed_val_kernel_2d(
                    vector_field=vector_field[:, :, -width:], fixed_vals=fixed_vals
                )

            return vector_field_set_fixed_val_at_boundaries_pyst_kernel_2d

        case _:
            msg = "Invalid field type"
            raise ValueError(msg)


def gen_add_fixed_val_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int] | None = None,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO: expand docs
    """2D add a fixed value to a field kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = spu.get_grid_info(grid_size, 2, "scalar")

    match field_type:
        case "scalar":
            sum_field, field = ps.fields(f"sum_field, field : {pyst_dtype}[{grid_info}]")
            fixed_val = sp.symbols("fixed_val")
            eqn = ps.Assignment(sum_field.center, field.center + fixed_val)
            return ps.create_kernel(eqn, config=kernel_config).compile()

        case "vector":
            sum_field_x, sum_field_y = ps.fields(
                f"sum_field_x, sum_field_y : {pyst_dtype}[{grid_info}]"
            )
            field_x, field_y = ps.fields(f"field_x, field_y : {pyst_dtype}[{grid_info}]")
            fixed_val_x, fixed_val_y = sp.symbols("fixed_val_x, fixed_val_y")
            eqn = [
                ps.Assignment(sum_field_x.center, field_x.center + fixed_val_x),
                ps.Assignment(sum_field_y.center, field_y.center + fixed_val_y),
            ]
            vector_kernel = ps.create_kernel(eqn, config=kernel_config).compile()
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()

            def vector_field_add_fixed_val_pyst_kernel_2d(
                sum_field: np.ndarray,
                vector_field: np.ndarray,
                fixed_vals: tuple[int, int],
            ) -> None:
                """Add fixed value to vector field in 2D"""
                vector_kernel(
                    sum_field_x=sum_field[x_axis_idx],
                    sum_field_y=sum_field[y_axis_idx],
                    field_x=vector_field[x_axis_idx],
                    field_y=vector_field[y_axis_idx],
                    fixed_val_x=fixed_vals[x_axis_idx],
                    fixed_val_y=fixed_vals[y_axis_idx],
                )

            return vector_field_add_fixed_val_pyst_kernel_2d

        case _:
            msg = "Invalid field type"
            raise ValueError(msg)


def gen_elementwise_saxpby_pyst_kernel_2d(
    real_t: type,
    num_threads: bool | int = False,
    grid_size: tuple[int, int] | None = None,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO: expand docs
    """2D elementwise saxpby (s = a * x + b * y) kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    grid_info = spu.get_grid_info(grid_size, 2, field_type)

    sum_field, field_1, field_2 = ps.fields(
        f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
    )
    field_1_prefac, field_2_prefac = sp.symbols("field_1_prefac, field_2_prefac")
    eqn = ps.Assignment(
        sum_field.center, field_1_prefac * field_1.center + field_2_prefac * field_2.center
    )
    return ps.create_kernel(eqn, config=kernel_config).compile()

"""Kernels for elementwise operations in 3D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.utils as spu
from typing import Callable, Literal


def gen_elementwise_sum_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO expand docs
    """3D elementwise sum kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)

    match field_type:
        case "scalar":
            grid_info = (
                f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
                if type(fixed_grid_size) is tuple[int, ...]
                else "3D"
            )

            @ps.kernel
            def _elementwise_sum_stencil_3d():
                sum_field, field_1, field_2 = ps.fields(
                    f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
                )
                sum_field[0, 0, 0] @= field_1[0, 0, 0] + field_2[0, 0, 0]

        case "vector":
            grid_info = (
                f"3, {fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
                if type(fixed_grid_size) is tuple[int, ...]
                else "4D"
            )

            @ps.kernel
            def _elementwise_sum_stencil_3d():  # noqa F811
                sum_field, field_1, field_2 = ps.fields(
                    f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
                )
                sum_field[0, 0, 0, 0] @= field_1[0, 0, 0, 0] + field_2[0, 0, 0, 0]

        case _:
            raise ValueError("Invalid field type")

    elementwise_sum_pyst_kernel_3d = ps.create_kernel(
        _elementwise_sum_stencil_3d, config=kernel_config
    ).compile()
    return elementwise_sum_pyst_kernel_3d


def gen_set_fixed_val_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO expand docs
    """3D set field to fixed value kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _set_fixed_val_stencil_3d():
        field = ps.fields(f"field : {pyst_dtype}[{grid_info}]")
        fixed_val = sp.symbols("fixed_val")
        field[0, 0, 0] @= fixed_val

    set_fixed_val_pyst_kernel_3d = ps.create_kernel(
        _set_fixed_val_stencil_3d, config=kernel_config
    ).compile()
    match field_type:
        case "scalar":
            return set_fixed_val_pyst_kernel_3d
        case "vector":
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()
            z_axis_idx = spu.VectorField.z_axis_idx()

            def vector_field_set_fixed_val_pyst_kernel_3d(
                vector_field: np.ndarray,
                fixed_vals: tuple[float, float, float],
            ) -> None:
                """Set vector field to fixed value in 3D.

                Sets spatially constant values for a 3D vector field,
                assumes shape of fields (3, n, n, n).
                """
                set_fixed_val_pyst_kernel_3d(
                    field=vector_field[x_axis_idx],
                    fixed_val=fixed_vals[x_axis_idx],
                )
                set_fixed_val_pyst_kernel_3d(
                    field=vector_field[y_axis_idx],
                    fixed_val=fixed_vals[y_axis_idx],
                )
                set_fixed_val_pyst_kernel_3d(
                    field=vector_field[z_axis_idx],
                    fixed_val=fixed_vals[z_axis_idx],
                )

            return vector_field_set_fixed_val_pyst_kernel_3d
        case _:
            raise ValueError("Invalid field type")


def gen_elementwise_copy_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
) -> Callable:

    # TODO expand docs
    """3D elementwise copy one field to another kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _elementwise_copy_stencil_3d():
        field, rhs_field = ps.fields(f"field, rhs_field : {pyst_dtype}[{grid_info}]")
        field[0, 0, 0] @= rhs_field[0, 0, 0]

    elementwise_copy_pyst_kernel_3d = ps.create_kernel(
        _elementwise_copy_stencil_3d, config=kernel_config
    ).compile()
    return elementwise_copy_pyst_kernel_3d


def gen_elementwise_complex_product_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
) -> Callable:
    # TODO expand docs
    """3D elementwise complex number product kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _elementwise_complex_product_stencil_3d():
        product_field_real, field_1_real, field_2_real = ps.fields(
            f"product_field_real, field_1_real, field_2_real : {pyst_dtype}[{grid_info}]"
        )
        product_field_imag, field_1_imag, field_2_imag = ps.fields(
            f"product_field_imag, field_1_imag, field_2_imag : {pyst_dtype}[{grid_info}]"
        )
        product_field_real[0, 0, 0] @= (
            field_1_real[0, 0, 0] * field_2_real[0, 0, 0]
            - field_1_imag[0, 0, 0] * field_2_imag[0, 0, 0]
        )
        product_field_imag[0, 0, 0] @= (
            field_1_real[0, 0, 0] * field_2_imag[0, 0, 0]
            + field_1_imag[0, 0, 0] * field_2_real[0, 0, 0]
        )

    _elementwise_complex_product_pyst_kernel_3d = ps.create_kernel(
        _elementwise_complex_product_stencil_3d, config=kernel_config
    ).compile()

    def elementwise_complex_product_pyst_kernel_3d(
        product_field: np.ndarray, field_1: np.ndarray, field_2: np.ndarray
    ) -> None:
        """3D Elementwise complex number product."""
        # complex numbers had compilation issues in pystencils :/
        _elementwise_complex_product_pyst_kernel_3d(
            product_field_real=product_field.real,
            product_field_imag=product_field.imag,
            field_1_real=field_1.real,
            field_1_imag=field_1.imag,
            field_2_real=field_2.real,
            field_2_imag=field_2.imag,
        )

    return elementwise_complex_product_pyst_kernel_3d


def gen_set_fixed_val_at_boundaries_pyst_kernel_3d(
    real_t: type,
    width: int,
    num_threads: bool | int = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:

    # TODO expand docs
    """3D set field to fixed value at boundaries kernel generator."""
    assert width > 0 and isinstance(width, int), "invalid zone width"
    set_fixed_val_kernel_3d = gen_set_fixed_val_pyst_kernel_3d(
        real_t=real_t,
        num_threads=num_threads,
        field_type=field_type,
    )

    match field_type:
        case "scalar":

            def set_fixed_val_at_boundaries_pyst_kernel_3d(
                field: np.ndarray, fixed_val: float
            ) -> None:
                """Set field to fixed value at boundaries.

                Set a 3D scalar field to a fixed value at boundaries in zone: width,
                used for clearing boundary noise.
                """
                set_fixed_val_kernel_3d(field=field[:width, :, :], fixed_val=fixed_val)
                set_fixed_val_kernel_3d(field=field[-width:, :, :], fixed_val=fixed_val)
                set_fixed_val_kernel_3d(field=field[:, :width, :], fixed_val=fixed_val)
                set_fixed_val_kernel_3d(field=field[:, -width:, :], fixed_val=fixed_val)
                set_fixed_val_kernel_3d(field=field[:, :, :width], fixed_val=fixed_val)
                set_fixed_val_kernel_3d(field=field[:, :, -width:], fixed_val=fixed_val)

            return set_fixed_val_at_boundaries_pyst_kernel_3d
        case "vector":

            def vector_field_set_fixed_val_at_boundaries_pyst_kernel_3d(
                vector_field: np.ndarray, fixed_vals: tuple[float, float, float]
            ) -> None:
                """Set field to fixed value at boundaries.

                Set a 3D vector field to a fixed value at boundaries in zone: width,
                used for clearing boundary noise.
                """
                set_fixed_val_kernel_3d(
                    vector_field=vector_field[:, :width, :, :], fixed_vals=fixed_vals
                )
                set_fixed_val_kernel_3d(
                    vector_field=vector_field[:, -width:, :, :], fixed_vals=fixed_vals
                )
                set_fixed_val_kernel_3d(
                    vector_field=vector_field[:, :, :width, :], fixed_vals=fixed_vals
                )
                set_fixed_val_kernel_3d(
                    vector_field=vector_field[:, :, -width:, :], fixed_vals=fixed_vals
                )
                set_fixed_val_kernel_3d(
                    vector_field=vector_field[:, :, :, :width], fixed_vals=fixed_vals
                )
                set_fixed_val_kernel_3d(
                    vector_field=vector_field[:, :, :, -width:], fixed_vals=fixed_vals
                )

            return vector_field_set_fixed_val_at_boundaries_pyst_kernel_3d
        case _:
            raise ValueError("Invalid field type")


def gen_add_fixed_val_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:

    # TODO expand docs
    """3D add a fixed value to a field kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _add_fixed_val_stencil_3d():
        sum_field, field = ps.fields(f"sum_field, field : {pyst_dtype}[{grid_info}]")
        fixed_val = sp.symbols("fixed_val")
        sum_field[0, 0, 0] @= field[0, 0, 0] + fixed_val

    add_fixed_val_pyst_kernel_3d = ps.create_kernel(
        _add_fixed_val_stencil_3d, config=kernel_config
    ).compile()
    match field_type:
        case "scalar":
            return add_fixed_val_pyst_kernel_3d
        case "vector":
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()
            z_axis_idx = spu.VectorField.z_axis_idx()

            def vector_field_add_fixed_val_pyst_kernel_3d(
                sum_field: np.ndarray,
                vector_field: np.ndarray,
                fixed_vals: tuple[int, int, int],
            ) -> None:
                """Add fixed value to vector field in 3D.

                Adds spatially constant values to a 3D vector field,
                assumes shape of fields (3, n, n).
                """
                add_fixed_val_pyst_kernel_3d(
                    sum_field=sum_field[x_axis_idx],
                    field=vector_field[x_axis_idx],
                    fixed_val=fixed_vals[x_axis_idx],
                )
                add_fixed_val_pyst_kernel_3d(
                    sum_field=sum_field[y_axis_idx],
                    field=vector_field[y_axis_idx],
                    fixed_val=fixed_vals[y_axis_idx],
                )
                add_fixed_val_pyst_kernel_3d(
                    sum_field=sum_field[z_axis_idx],
                    field=vector_field[z_axis_idx],
                    fixed_val=fixed_vals[z_axis_idx],
                )

            return vector_field_add_fixed_val_pyst_kernel_3d
        case _:
            raise ValueError("Invalid field type")


def gen_elementwise_saxpby_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO expand docs
    """3D elementwise saxpby (s = a * x + b * y) kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)

    match field_type:
        case "scalar":
            grid_info = (
                f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
                if type(fixed_grid_size) is tuple[int, ...]
                else "3D"
            )

            @ps.kernel
            def _elementwise_saxpby_stencil_3d():
                sum_field, field_1, field_2 = ps.fields(
                    f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
                )
                field_1_prefac, field_2_prefac = sp.symbols(
                    "field_1_prefac, field_2_prefac"
                )
                sum_field[0, 0, 0] @= (
                    field_1_prefac * field_1[0, 0, 0]
                    + field_2_prefac * field_2[0, 0, 0]
                )

        case "vector":
            grid_info = (
                f"3, {fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
                if type(fixed_grid_size) is tuple[int, ...]
                else "4D"
            )

            @ps.kernel
            def _elementwise_saxpby_stencil_3d():  # noqa F811
                sum_field, field_1, field_2 = ps.fields(
                    f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
                )
                field_1_prefac, field_2_prefac = sp.symbols(
                    "field_1_prefac, field_2_prefac"
                )
                sum_field[0, 0, 0, 0] @= (
                    field_1_prefac * field_1[0, 0, 0, 0]
                    + field_2_prefac * field_2[0, 0, 0, 0]
                )

        case _:
            raise ValueError("Invalid field type")

    elementwise_saxpby_pyst_kernel_3d = ps.create_kernel(
        _elementwise_saxpby_stencil_3d, config=kernel_config
    ).compile()
    return elementwise_saxpby_pyst_kernel_3d


def gen_elementwise_cross_product_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
) -> Callable:
    # TODO expand docs
    """3D elementwise cross product kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _elementwise_cross_product_single_axis_stencil_3d():
        result_field_i, field_1_j, field_1_k, field_2_j, field_2_k = ps.fields(
            f"result_field_i, field_1_j, field_1_k, field_2_j, field_2_k : {pyst_dtype}[{grid_info}]"
        )
        result_field_i[0, 0, 0] @= (
            field_1_j[0, 0, 0] * field_2_k[0, 0, 0]
            - field_2_j[0, 0, 0] * field_1_k[0, 0, 0]
        )

    elementwise_cross_product_single_axis_kernel_3d = ps.create_kernel(
        _elementwise_cross_product_single_axis_stencil_3d, config=kernel_config
    ).compile()
    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()

    def elementwise_cross_product_pyst_kernel_3d(
        result_field: np.ndarray, field_1: np.ndarray, field_2: np.ndarray
    ) -> None:
        """Elementwise cross product of 2 3D vector fields"""
        # Assumes fields are (3, n, n, n)
        elementwise_cross_product_single_axis_kernel_3d(
            result_field_i=result_field[x_axis_idx],
            field_1_j=field_1[y_axis_idx],
            field_1_k=field_1[z_axis_idx],
            field_2_j=field_2[y_axis_idx],
            field_2_k=field_2[z_axis_idx],
        )
        elementwise_cross_product_single_axis_kernel_3d(
            result_field_i=result_field[y_axis_idx],
            field_1_j=field_1[z_axis_idx],
            field_1_k=field_1[x_axis_idx],
            field_2_j=field_2[z_axis_idx],
            field_2_k=field_2[x_axis_idx],
        )
        elementwise_cross_product_single_axis_kernel_3d(
            result_field_i=result_field[z_axis_idx],
            field_1_j=field_1[x_axis_idx],
            field_1_k=field_1[y_axis_idx],
            field_2_j=field_2[x_axis_idx],
            field_2_k=field_2[y_axis_idx],
        )

    return elementwise_cross_product_pyst_kernel_3d

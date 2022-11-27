"""Kernels for elementwise operations in 2D."""
import pystencils as ps

import sympy as sp

from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_elementwise_sum_pyst_kernel_2d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    # TODO expand docs
    """2D elementwise sum kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)

    if field_type == "scalar":
        grid_info = (
            f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
        )

        @ps.kernel
        def _elementwise_sum_stencil_2d():
            sum_field, field_1, field_2 = ps.fields(
                f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
            )
            sum_field[0, 0] @= field_1[0, 0] + field_2[0, 0]

    elif field_type == "vector":
        grid_info = (
            f"2, {fixed_grid_size[0]}, {fixed_grid_size[1]}"
            if fixed_grid_size
            else "3D"
        )

        @ps.kernel
        def _elementwise_sum_stencil_2d():
            sum_field, field_1, field_2 = ps.fields(
                f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
            )
            sum_field[0, 0, 0] @= field_1[0, 0, 0] + field_2[0, 0, 0]

    elementwise_sum_pyst_kernel_2d = ps.create_kernel(
        _elementwise_sum_stencil_2d, config=kernel_config
    ).compile()
    return elementwise_sum_pyst_kernel_2d


def gen_set_fixed_val_pyst_kernel_2d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    # TODO expand docs
    """2D set field to fixed value kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
    )

    @ps.kernel
    def _set_fixed_val_stencil_2d():
        field = ps.fields(f"field : {pyst_dtype}[{grid_info}]")
        fixed_val = sp.symbols("fixed_val")
        field[0, 0] @= fixed_val

    set_fixed_val_pyst_kernel_2d = ps.create_kernel(
        _set_fixed_val_stencil_2d, config=kernel_config
    ).compile()
    if field_type == "scalar":
        return set_fixed_val_pyst_kernel_2d
    elif field_type == "vector":

        def vector_field_set_fixed_val_pyst_kernel_2d(
            vector_field,
            fixed_vals,
        ):
            """Set vector field to fixed value in 2D.

            Sets spatially constant values for a 2D vector field,
            assumes shape of fields (2, n, n).
            """
            set_fixed_val_pyst_kernel_2d(
                field=vector_field[0],
                fixed_val=fixed_vals[0],
            )
            set_fixed_val_pyst_kernel_2d(
                field=vector_field[1],
                fixed_val=fixed_vals[1],
            )

        return vector_field_set_fixed_val_pyst_kernel_2d


def gen_elementwise_copy_pyst_kernel_2d(
    real_t, num_threads=False, fixed_grid_size=False
):
    # TODO expand docs
    """2D elementwise copy one field to another kernel generator."""
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
    )

    @ps.kernel
    def _elementwise_copy_stencil_2d():
        field, rhs_field = ps.fields(f"field, rhs_field : {pyst_dtype}[{grid_info}]")
        field[0, 0] @= rhs_field[0, 0]

    elementwise_copy_pyst_kernel_2d = ps.create_kernel(
        _elementwise_copy_stencil_2d, config=kernel_config
    ).compile()
    return elementwise_copy_pyst_kernel_2d


def gen_elementwise_complex_product_pyst_kernel_2d(
    real_t, num_threads=False, fixed_grid_size=False
):
    # TODO expand docs
    """2D elementwise complex number product kernel generator."""
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
    )

    @ps.kernel
    def _elementwise_complex_product_stencil_2d():
        product_field_real, field_1_real, field_2_real = ps.fields(
            f"product_field_real, field_1_real, field_2_real : {pyst_dtype}[{grid_info}]"
        )
        product_field_imag, field_1_imag, field_2_imag = ps.fields(
            f"product_field_imag, field_1_imag, field_2_imag : {pyst_dtype}[{grid_info}]"
        )
        product_field_real[0, 0] @= (
            field_1_real[0, 0] * field_2_real[0, 0]
            - field_1_imag[0, 0] * field_2_imag[0, 0]
        )
        product_field_imag[0, 0] @= (
            field_1_real[0, 0] * field_2_imag[0, 0]
            + field_1_imag[0, 0] * field_2_real[0, 0]
        )

    _elementwise_complex_product_pyst_kernel_2d = ps.create_kernel(
        _elementwise_complex_product_stencil_2d, config=kernel_config
    ).compile()

    def elementwise_complex_product_pyst_kernel_2d(product_field, field_1, field_2):
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
    real_t,
    width,
    num_threads=False,
    field_type="scalar",
):
    # TODO expand docs
    """2D set field to fixed value at boundaries kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    assert width > 0 and isinstance(width, int), "invalid zone width"
    set_fixed_val_kernel_2d = gen_set_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        num_threads=num_threads,
        field_type=field_type,
    )

    if field_type == "scalar":

        def set_fixed_val_at_boundaries_pyst_kernel_2d(field, fixed_val):
            """Set field to fixed value at boundaries.

            Set a 2D scalar field to a fixed value at boundaries in zone: width,
            used for clearing boundary noise.
            """
            set_fixed_val_kernel_2d(field=field[:width, :], fixed_val=fixed_val)
            set_fixed_val_kernel_2d(field=field[-width:, :], fixed_val=fixed_val)
            set_fixed_val_kernel_2d(field=field[:, :width], fixed_val=fixed_val)
            set_fixed_val_kernel_2d(field=field[:, -width:], fixed_val=fixed_val)

        return set_fixed_val_at_boundaries_pyst_kernel_2d

    elif field_type == "vector":

        def vector_field_set_fixed_val_at_boundaries_pyst_kernel_2d(
            vector_field, fixed_vals
        ):
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


def gen_add_fixed_val_pyst_kernel_2d(
    real_t, num_threads=False, fixed_grid_size=False, field_type="scalar"
):
    # TODO expand docs
    """2D add a fixed value to a field kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
    )

    @ps.kernel
    def _add_fixed_val_stencil_2d():
        sum_field, field = ps.fields(f"sum_field, field : {pyst_dtype}[{grid_info}]")
        fixed_val = sp.symbols("fixed_val")
        sum_field[0, 0] @= field[0, 0] + fixed_val

    add_fixed_val_pyst_kernel_2d = ps.create_kernel(
        _add_fixed_val_stencil_2d, config=kernel_config
    ).compile()
    if field_type == "scalar":
        return add_fixed_val_pyst_kernel_2d
    elif field_type == "vector":

        def vector_field_add_fixed_val_pyst_kernel_2d(
            sum_field, vector_field, fixed_vals
        ):
            """Add fixed value to vector field in 2D.

            Adds spatially constant values to a 2D vector field,
            assumes shape of fields (2, n, n).
            """
            add_fixed_val_pyst_kernel_2d(
                sum_field=sum_field[0], field=vector_field[0], fixed_val=fixed_vals[0]
            )
            add_fixed_val_pyst_kernel_2d(
                sum_field=sum_field[1], field=vector_field[1], fixed_val=fixed_vals[1]
            )

        return vector_field_add_fixed_val_pyst_kernel_2d


def gen_elementwise_saxpby_pyst_kernel_2d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    # TODO expand docs
    """2D elementwise saxpby (s = a * x + b * y) kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)

    if field_type == "scalar":
        grid_info = (
            f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
        )

        @ps.kernel
        def _elementwise_saxpby_stencil_2d():
            sum_field, field_1, field_2 = ps.fields(
                f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
            )
            field_1_prefac, field_2_prefac = sp.symbols(
                "field_1_prefac, field_2_prefac"
            )
            sum_field[0, 0] @= (
                field_1_prefac * field_1[0, 0] + field_2_prefac * field_2[0, 0]
            )

    elif field_type == "vector":
        grid_info = (
            f"2, {fixed_grid_size[0]}, {fixed_grid_size[1]}"
            if fixed_grid_size
            else "3D"
        )

        @ps.kernel
        def _elementwise_saxpby_stencil_2d():
            sum_field, field_1, field_2 = ps.fields(
                f"sum_field, field_1, field_2 : {pyst_dtype}[{grid_info}]"
            )
            field_1_prefac, field_2_prefac = sp.symbols(
                "field_1_prefac, field_2_prefac"
            )
            sum_field[0, 0, 0] @= (
                field_1_prefac * field_1[0, 0, 0] + field_2_prefac * field_2[0, 0, 0]
            )

    elementwise_saxpby_pyst_kernel_2d = ps.create_kernel(
        _elementwise_saxpby_stencil_2d, config=kernel_config
    ).compile()
    return elementwise_saxpby_pyst_kernel_2d

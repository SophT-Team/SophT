"""Kernels for Brinkmann penalisation in 2D."""
import pystencils as ps

import sympy as sp

from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config


def gen_brinkmann_penalise_pyst_kernel_2d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    """Brinkmann penalisation 2D kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
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
    if field_type == "scalar":
        return brinkmann_penalise_pyst_kernel_2d
    elif field_type == "vector":

        def brinkmann_penalise_vector_field_pyst_kernel_2d(
            penalised_vector_field,
            penalty_factor,
            char_field,
            penalty_vector_field,
            vector_field,
        ):
            """Brinkmann penalises a vector field in 2D."""
            brinkmann_penalise_pyst_kernel_2d(
                penalised_field=penalised_vector_field[0],
                penalty_factor=penalty_factor,
                char_field=char_field,
                penalty_field=penalty_vector_field[0],
                field=vector_field[0],
            )
            brinkmann_penalise_pyst_kernel_2d(
                penalised_field=penalised_vector_field[1],
                penalty_factor=penalty_factor,
                char_field=char_field,
                penalty_field=penalty_vector_field[1],
                field=vector_field[1],
            )

        return brinkmann_penalise_vector_field_pyst_kernel_2d


def gen_brinkmann_penalise_vs_fixed_val_pyst_kernel_2d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    """Brinkmann penalisation against fixed val, 2D kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}" if fixed_grid_size else "2D"
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
    if field_type == "scalar":
        return brinkmann_penalise_vs_fixed_val_pyst_kernel_2d
    elif field_type == "vector":

        def brinkmann_penalise_vector_field_vs_fixed_val_pyst_kernel_2d(
            penalised_vector_field,
            penalty_factor,
            char_field,
            penalty_val,
            vector_field,
        ):
            """Brinkmann penalises a vector field against fixed values in 2D."""
            brinkmann_penalise_vs_fixed_val_pyst_kernel_2d(
                penalised_field=penalised_vector_field[0],
                penalty_factor=penalty_factor,
                char_field=char_field,
                penalty_val=penalty_val[0],
                field=vector_field[0],
            )
            brinkmann_penalise_vs_fixed_val_pyst_kernel_2d(
                penalised_field=penalised_vector_field[1],
                penalty_factor=penalty_factor,
                char_field=char_field,
                penalty_val=penalty_val[1],
                field=vector_field[1],
            )

        return brinkmann_penalise_vector_field_vs_fixed_val_pyst_kernel_2d

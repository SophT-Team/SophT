"""Kernels applying laplacian filter on 3d grid for scalar and vector fields"""
import numpy as np
import pystencils as ps
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d import (
    gen_elementwise_copy_pyst_kernel_3d,
    gen_elementwise_saxpby_pyst_kernel_3d,
    gen_set_fixed_val_at_boundaries_pyst_kernel_3d,
)
from sopht.utils.pyst_kernel_config import get_pyst_dtype, get_pyst_kernel_config
from typing import Union, Tuple, Type


def gen_laplacian_filter_kernel_3d(
    filter_order: int,
    filter_flux_buffer: np.ndarray,
    field_buffer: np.ndarray,
    real_t: Type,
    num_threads: Union[bool, int] = False,
    fixed_grid_size: Union[Tuple, bool] = False,
    field_type: str = "scalar",
    filter_type: str = "multiplicative",
    filter_flux_buffer_boundary_width: int = 1,
):
    """
    Laplacian filter kernel generator. Based on the field type
    filter kernels for both scalar and vectorial field can be constructed.
    One dimensional laplacian filter applied on the field in 3D.

    Notes
    -----
    For details regarding the numerics behind the filtering, refer to [1]_, [2]_.
    .. [1] Jeanmart, H., & Winckelmans, G. (2007). Investigation of eddy-viscosity
       models modified using discrete filters: a simplified “regularized variational
       multiscale model” and an “enhanced field model”. Physics of fluids, 19(5), 055110.
    .. [2] Lorieul, G. (2018). Development and validation of a 2D Vortex Particle-Mesh
       method for incompressible multiphase flows (Doctoral dissertation,
       Université Catholique de Louvain).
    """

    assert filter_order >= 0 and isinstance(filter_order, int), "Invalid filter order"
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    supported_filter_types = ["multiplicative", "convolution"]
    if filter_type not in supported_filter_types:
        raise ValueError("Invalid filter type")
    pyst_dtype = get_pyst_dtype(real_t)
    kernel_config = get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple
        else "3D"
    )

    @ps.kernel
    def _laplacian_filter_3d_x():
        filter_flux, field = ps.fields(
            f"filter_flux, field : {pyst_dtype}[{grid_info}]"
        )
        filter_flux[0, 0, 0] @= 0.25 * (
            -field[0, 0, 1] - field[0, 0, -1] + 2 * field[0, 0, 0]
        )

    laplacian_filter_3d_x = ps.create_kernel(
        _laplacian_filter_3d_x, config=kernel_config
    ).compile()

    @ps.kernel
    def _laplacian_filter_3d_y():
        filter_flux, field = ps.fields(
            f"filter_flux, field : {pyst_dtype}[{grid_info}]"
        )
        filter_flux[0, 0, 0] @= 0.25 * (
            -field[0, 1, 0] - field[0, -1, 0] + 2 * field[0, 0, 0]
        )

    laplacian_filter_3d_y = ps.create_kernel(
        _laplacian_filter_3d_y, config=kernel_config
    ).compile()

    @ps.kernel
    def _laplacian_filter_3d_z():
        filter_flux, field = ps.fields(
            f"filter_flux, field : {pyst_dtype}[{grid_info}]"
        )
        filter_flux[0, 0, 0] @= 0.25 * (
            -field[1, 0, 0] - field[-1, 0, 0] + 2 * field[0, 0, 0]
        )

    laplacian_filter_3d_z = ps.create_kernel(
        _laplacian_filter_3d_z, config=kernel_config
    ).compile()

    elementwise_copy_3d = gen_elementwise_copy_pyst_kernel_3d(
        real_t=real_t, num_threads=num_threads, fixed_grid_size=fixed_grid_size
    )
    elementwise_saxpby_3d = gen_elementwise_saxpby_pyst_kernel_3d(
        real_t=real_t, num_threads=num_threads, fixed_grid_size=fixed_grid_size
    )
    # to set boundary zone = 0
    boundary_width = filter_flux_buffer_boundary_width
    set_fixed_val_at_boundaries_3d = gen_set_fixed_val_at_boundaries_pyst_kernel_3d(
        real_t=real_t,
        width=boundary_width,
        # complexity of this operation is O(N^2), hence setting serial version
        num_threads=False,
        field_type="scalar",
    )

    def scalar_field_multiplicative_filter_kernel_3d(scalar_field):
        """
        Applies multiplicative Laplacian filter on any scalar field.
        """
        set_fixed_val_at_boundaries_3d(field=filter_flux_buffer, fixed_val=0)
        elementwise_copy_3d(field=field_buffer, rhs_field=scalar_field)
        for _ in range(filter_order):
            # Laplacian filter in x direction
            laplacian_filter_3d_x(filter_flux=filter_flux_buffer, field=field_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)
            # Laplacian filter in y direction
            laplacian_filter_3d_y(filter_flux=filter_flux_buffer, field=field_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)
            # Laplacian filter in z direction
            laplacian_filter_3d_z(filter_flux=filter_flux_buffer, field=field_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)

        elementwise_saxpby_3d(
            sum_field=scalar_field,
            field_1=scalar_field,
            field_1_prefac=1.0,
            field_2=filter_flux_buffer,
            field_2_prefac=-1.0,
        )

    def scalar_field_convolution_filter_kernel_3d(scalar_field):
        """
        Applies convolution Laplacian filter on any scalar field.
        """
        set_fixed_val_at_boundaries_3d(field=filter_flux_buffer, fixed_val=0)

        # Laplacian filter in x direction
        elementwise_copy_3d(field=field_buffer, rhs_field=scalar_field)
        for _ in range(filter_order):
            laplacian_filter_3d_x(filter_flux=filter_flux_buffer, field=field_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)
        elementwise_saxpby_3d(
            sum_field=scalar_field,
            field_1=scalar_field,
            field_1_prefac=1.0,
            field_2=filter_flux_buffer,
            field_2_prefac=-1.0,
        )

        # Laplacian filter in y direction
        elementwise_copy_3d(field=field_buffer, rhs_field=scalar_field)
        for _ in range(filter_order):
            laplacian_filter_3d_y(filter_flux=filter_flux_buffer, field=field_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)
        elementwise_saxpby_3d(
            sum_field=scalar_field,
            field_1=scalar_field,
            field_1_prefac=1.0,
            field_2=filter_flux_buffer,
            field_2_prefac=-1.0,
        )

        # Laplacian filter in z direction
        elementwise_copy_3d(field=field_buffer, rhs_field=scalar_field)
        for _ in range(filter_order):
            laplacian_filter_3d_z(filter_flux=filter_flux_buffer, field=field_buffer)
            elementwise_copy_3d(field=field_buffer, rhs_field=filter_flux_buffer)
        elementwise_saxpby_3d(
            sum_field=scalar_field,
            field_1=scalar_field,
            field_1_prefac=1.0,
            field_2=filter_flux_buffer,
            field_2_prefac=-1.0,
        )

    scalar_field_filter_kernel_3d = None
    if filter_type == "multiplicative":
        scalar_field_filter_kernel_3d = scalar_field_multiplicative_filter_kernel_3d
    elif filter_type == "convolution":
        scalar_field_filter_kernel_3d = scalar_field_convolution_filter_kernel_3d
    # Depending on the field type return the relevant filter implementation
    if field_type == "scalar":
        return scalar_field_filter_kernel_3d
    elif field_type == "vector":

        def vector_filed_filter_kernel_3d(vector_field):
            """
            Applies laplacian filter on any vector field.
            """
            scalar_field_filter_kernel_3d(scalar_field=vector_field[0])
            scalar_field_filter_kernel_3d(scalar_field=vector_field[1])
            scalar_field_filter_kernel_3d(scalar_field=vector_field[2])

        return vector_filed_filter_kernel_3d

"""Kernels for updating passive field based on ibm forcing in 3D."""
import numpy as np
import pystencils as ps
import sympy as sp
import sopht.utils as spu
from typing import Callable


def gen_update_passive_field_from_forcing_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
) -> Callable:
    # TODO expand docs
    """Update passive field based on feedback forcing in 3D kernel generator."""
    pyst_dtype = spu.get_pyst_dtype(real_t)
    kernel_config = spu.get_pyst_kernel_config(real_t, num_threads)
    # we can add dtype checks later
    grid_info = (
        f"{fixed_grid_size[0]}, {fixed_grid_size[1]}, {fixed_grid_size[2]}"
        if type(fixed_grid_size) is tuple[int, ...]
        else "3D"
    )

    @ps.kernel
    def _update_passive_field_from_forcing_stencil_3d():
        passive_field, forcing_field = ps.fields(
            f"passive_field, forcing_field " f": {pyst_dtype}[{grid_info}]"
        )
        prefactor = sp.symbols("prefactor")
        passive_field[0, 0, 0] @= (
            passive_field[0, 0, 0] + forcing_field[0, 0, 0] * prefactor
        )

    _update_passive_field_from_forcing_pyst_kernel_3d = ps.create_kernel(
        _update_passive_field_from_forcing_stencil_3d, config=kernel_config
    ).compile()

    def update_passive_field_from_forcing_pyst_kernel_3d(
        passive_field: np.ndarray,
        forcing_field: np.ndarray,
        prefactor: float,
    ) -> None:
        """Kernel for updating passive field based on forcing.

        Updates passive_field based on forcing_field
        passive_field += prefactor * forcing_field
        prefactor: grid spacing factored out, along with any other multiplier
        Assumes forcing_field is (n, n, n).
        """
        _update_passive_field_from_forcing_pyst_kernel_3d(
            passive_field=passive_field,
            forcing_field=forcing_field,
            prefactor=prefactor,
        )

    return update_passive_field_from_forcing_pyst_kernel_3d

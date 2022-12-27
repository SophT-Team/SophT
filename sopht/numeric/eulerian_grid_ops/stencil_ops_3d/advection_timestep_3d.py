"""Kernels for performing advection timestep in 3D."""
import numpy as np
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu
from typing import Callable, Literal


def gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
    real_t: type,
    num_threads: bool | int = False,
    fixed_grid_size: tuple[int, int, int] | bool = False,
    field_type: Literal["scalar", "vector"] = "scalar",
) -> Callable:
    # TODO expand docs
    """3D Advection (ENO3 stencil) Euler forward timestep generator."""
    elementwise_sum_pyst_kernel_3d = spne.gen_elementwise_sum_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    set_fixed_val_pyst_kernel_3d = spne.gen_set_fixed_val_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    advection_flux_conservative_eno3_pyst_kernel_3d = (
        spne.gen_advection_flux_conservative_eno3_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=fixed_grid_size,
            num_threads=num_threads,
        )
    )

    def advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
        field: np.ndarray,
        advection_flux: np.ndarray,
        velocity: np.ndarray,
        dt_by_dx: float,
    ) -> None:
        """3D Advection (ENO3 stencil) Euler forward timestep (scalar field).

        Performs an inplace advection timestep via ENO3 in 3D using Euler forward,
        for a 3D scalar field (n, n, n).
        """
        set_fixed_val_pyst_kernel_3d(field=advection_flux, fixed_val=0)
        advection_flux_conservative_eno3_pyst_kernel_3d(
            advection_flux=advection_flux,
            field=field,
            velocity=velocity,
            inv_dx=-dt_by_dx,
        )
        elementwise_sum_pyst_kernel_3d(
            sum_field=field, field_1=field, field_2=advection_flux
        )

    match field_type:
        case "scalar":
            return advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d
        case "vector":
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()
            z_axis_idx = spu.VectorField.z_axis_idx()

            def vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                vector_field: np.ndarray,
                advection_flux: np.ndarray,
                velocity: np.ndarray,
                dt_by_dx: float,
            ) -> None:
                """3D Advection (ENO3 stencil) Euler forward timestep (vector field).

                Performs an inplace advection timestep via ENO3 in 3D using Euler forward,
                for a 3D vector field (3, n, n, n).
                """
                advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                    field=vector_field[x_axis_idx],
                    advection_flux=advection_flux,
                    velocity=velocity,
                    dt_by_dx=dt_by_dx,
                )
                advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                    field=vector_field[y_axis_idx],
                    advection_flux=advection_flux,
                    velocity=velocity,
                    dt_by_dx=dt_by_dx,
                )
                advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                    field=vector_field[z_axis_idx],
                    advection_flux=advection_flux,
                    velocity=velocity,
                    dt_by_dx=dt_by_dx,
                )

            return vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d
        case _:
            raise ValueError("Invalid field type")

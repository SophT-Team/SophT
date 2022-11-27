"""Kernels for performing advection timestep in 3D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.advection_flux_3d import (
    gen_advection_flux_conservative_eno3_pyst_kernel_3d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_elementwise_sum_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
)


def gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    # TODO expand docs
    """3D Advection (ENO3 stencil) Euler forward timestep generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    elementwise_sum_pyst_kernel_3d = gen_elementwise_sum_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    set_fixed_val_pyst_kernel_3d = gen_set_fixed_val_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    advection_flux_conservative_eno3_pyst_kernel_3d = (
        gen_advection_flux_conservative_eno3_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=fixed_grid_size,
            num_threads=num_threads,
        )
    )

    def advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
        field, advection_flux, velocity, dt_by_dx
    ):
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

    if field_type == "scalar":
        return advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d
    elif field_type == "vector":

        def vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
            vector_field, advection_flux, velocity, dt_by_dx
        ):
            """3D Advection (ENO3 stencil) Euler forward timestep (vector field).

            Performs an inplace advection timestep via ENO3 in 3D using Euler forward,
            for a 3D vector field (3, n, n, n).
            """
            advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                field=vector_field[0],
                advection_flux=advection_flux,
                velocity=velocity,
                dt_by_dx=dt_by_dx,
            )
            advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                field=vector_field[1],
                advection_flux=advection_flux,
                velocity=velocity,
                dt_by_dx=dt_by_dx,
            )
            advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                field=vector_field[2],
                advection_flux=advection_flux,
                velocity=velocity,
                dt_by_dx=dt_by_dx,
            )

        return vector_field_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d

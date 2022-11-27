"""Kernels for performing diffusion timestep in 3D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.diffusion_flux_3d import (
    gen_diffusion_flux_pyst_kernel_3d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_elementwise_sum_pyst_kernel_3d,
)


def gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
    real_t,
    num_threads=False,
    fixed_grid_size=False,
    field_type="scalar",
):
    # TODO expand docs
    """3D Diffusion euler forward timestep kernel generator."""
    assert field_type == "scalar" or field_type == "vector", "Invalid field type"
    elementwise_sum_pyst_kernel_3d = gen_elementwise_sum_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )
    diffusion_flux_kernel_3d = gen_diffusion_flux_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
    )

    def diffusion_timestep_euler_forward_pyst_kernel_3d(
        field, diffusion_flux, nu_dt_by_dx2
    ):
        """3D Diffusion Euler forward timestep (scalar field).

        Performs an inplace diffusion timestep in 3D using Euler forward,
        for a 3D field (n, n, n).
        """
        diffusion_flux_kernel_3d(
            diffusion_flux=diffusion_flux,
            field=field,
            prefactor=nu_dt_by_dx2,
        )
        elementwise_sum_pyst_kernel_3d(
            sum_field=field, field_1=field, field_2=diffusion_flux
        )

    if field_type == "scalar":
        return diffusion_timestep_euler_forward_pyst_kernel_3d
    elif field_type == "vector":

        def vector_field_diffusion_timestep_euler_forward_pyst_kernel_3d(
            vector_field, diffusion_flux, nu_dt_by_dx2
        ):
            """3D Diffusion Euler forward timestep (vector field).

            Performs an inplace diffusion timestep in 3D using Euler forward,
            for a 3D vector field (3, n, n, n).
            """
            diffusion_timestep_euler_forward_pyst_kernel_3d(
                field=vector_field[0],
                diffusion_flux=diffusion_flux,
                nu_dt_by_dx2=nu_dt_by_dx2,
            )
            diffusion_timestep_euler_forward_pyst_kernel_3d(
                field=vector_field[1],
                diffusion_flux=diffusion_flux,
                nu_dt_by_dx2=nu_dt_by_dx2,
            )
            diffusion_timestep_euler_forward_pyst_kernel_3d(
                field=vector_field[2],
                diffusion_flux=diffusion_flux,
                nu_dt_by_dx2=nu_dt_by_dx2,
            )

        return vector_field_diffusion_timestep_euler_forward_pyst_kernel_3d

"""Kernels for performing vorticity stretching timestep in 3D."""
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_elementwise_sum_pyst_kernel_3d,
    gen_elementwise_saxpby_pyst_kernel_3d,
)
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.vorticity_stretching_flux_3d import (
    gen_vorticity_stretching_flux_pyst_kernel_3d,
)


def gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d(
    real_t, num_threads=False, fixed_grid_size=False
):
    # TODO expand docs
    """3D Vorticity stretching euler forward timestep kernel generator."""
    elementwise_sum_pyst_kernel_3d = gen_elementwise_sum_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
        field_type="vector",
    )
    vorticity_stretching_flux_pyst_kernel_3d = (
        gen_vorticity_stretching_flux_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=fixed_grid_size,
            num_threads=num_threads,
        )
    )

    def vorticity_stretching_timestep_euler_forward_pyst_kernel_3d(
        vorticity_field, velocity_field, vorticity_stretching_flux_field, dt_by_2_dx
    ):
        """3D Vorticity stretching Euler forward timestep kernel.

        Performs the vorticity stretching timestep in 3D using Euler forward,
        for a 3D vorticity_field (3, n, n, n), velocity_field (3, n, n, n) and
        updates result in vorticity_field (3, n, n, n)
        """
        vorticity_stretching_flux_pyst_kernel_3d(
            vorticity_stretching_flux_field=vorticity_stretching_flux_field,
            vorticity_field=vorticity_field,
            velocity_field=velocity_field,
            prefactor=dt_by_2_dx,
        )

        elementwise_sum_pyst_kernel_3d(
            sum_field=vorticity_field,
            field_1=vorticity_field,
            field_2=vorticity_stretching_flux_field,
        )

    return vorticity_stretching_timestep_euler_forward_pyst_kernel_3d


def gen_vorticity_stretching_timestep_ssprk3_pyst_kernel_3d(
    real_t, midstep_buffer_vector_field, num_threads=False, fixed_grid_size=False
):
    # TODO expand docs
    """3D Vorticity stretching SSP-RK3 timestep kernel generator."""
    elementwise_sum_pyst_kernel_3d = gen_elementwise_sum_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
        field_type="vector",
    )
    elementwise_saxpby_pyst_kernel_3d = gen_elementwise_saxpby_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=fixed_grid_size,
        num_threads=num_threads,
        field_type="vector",
    )
    vorticity_stretching_flux_pyst_kernel_3d = (
        gen_vorticity_stretching_flux_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=fixed_grid_size,
            num_threads=num_threads,
        )
    )

    def vorticity_stretching_timestep_ssprk3_pyst_kernel_3d(
        vorticity_field, velocity_field, vorticity_stretching_flux_field, dt_by_2_dx
    ):
        """3D Vorticity stretching SSP-RK3 timestep kernel.

        Performs the vorticity stretching timestep in 3D using SSP RK3,
        for a 3D vorticity_field (3, n, n, n), velocity_field (3, n, n, n) and
        updates result in vorticity_field (3, n, n, n)

        Notes
        -----

        References:
        https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html#ssp-rk3
        https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        """

        # first step
        vorticity_stretching_flux_pyst_kernel_3d(
            vorticity_stretching_flux_field=vorticity_stretching_flux_field,
            vorticity_field=vorticity_field,
            velocity_field=velocity_field,
            prefactor=dt_by_2_dx,
        )
        post_step_1_vorticity_field = midstep_buffer_vector_field.view()
        elementwise_sum_pyst_kernel_3d(
            sum_field=post_step_1_vorticity_field,
            field_1=vorticity_field,
            field_2=vorticity_stretching_flux_field,
        )

        # second step
        vorticity_stretching_flux_pyst_kernel_3d(
            vorticity_stretching_flux_field=vorticity_stretching_flux_field,
            vorticity_field=post_step_1_vorticity_field,
            velocity_field=velocity_field,
            prefactor=dt_by_2_dx,
        )
        elementwise_sum_pyst_kernel_3d(
            sum_field=post_step_1_vorticity_field,
            field_1=post_step_1_vorticity_field,
            field_2=vorticity_stretching_flux_field,
        )
        post_step_2_vorticity_field = midstep_buffer_vector_field.view()
        elementwise_saxpby_pyst_kernel_3d(
            sum_field=post_step_2_vorticity_field,
            field_1=vorticity_field,
            field_2=post_step_1_vorticity_field,
            field_1_prefac=0.75,
            field_2_prefac=0.25,
        )

        # third step
        vorticity_stretching_flux_pyst_kernel_3d(
            vorticity_stretching_flux_field=vorticity_stretching_flux_field,
            vorticity_field=post_step_2_vorticity_field,
            velocity_field=velocity_field,
            prefactor=(dt_by_2_dx * 0.5),
        )
        elementwise_sum_pyst_kernel_3d(
            sum_field=post_step_2_vorticity_field,
            field_1=post_step_2_vorticity_field,
            field_2=vorticity_stretching_flux_field,
        )
        elementwise_saxpby_pyst_kernel_3d(
            sum_field=vorticity_field,
            field_1=vorticity_field,
            field_2=post_step_2_vorticity_field,
            field_1_prefac=(1.0 / 3.0),
            field_2_prefac=(2.0 / 3.0),
        )

    return vorticity_stretching_timestep_ssprk3_pyst_kernel_3d

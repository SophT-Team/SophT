import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d,
    gen_vorticity_stretching_timestep_ssprk3_pyst_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol
from tests.test_numeric.test_eulerian_grid_ops.test_stencil_ops_3d.test_vorticity_stretching_flux_3d import (
    vorticity_stretching_flux_reference,
)


def vorticity_stretching_timestep_euler_forward_reference(
    vorticity_field, velocity_field, dt_by_2_dx
):
    vorticity_stretching_flux_field = vorticity_stretching_flux_reference(
        vorticity_field, velocity_field, prefactor=dt_by_2_dx
    )
    return vorticity_field + vorticity_stretching_flux_field


def vorticity_stretching_timestep_ssprk3_reference(
    vorticity_field, velocity_field, dt_by_2_dx
):
    # first step
    vorticity_stretching_flux_field = vorticity_stretching_flux_reference(
        vorticity_field, velocity_field, prefactor=dt_by_2_dx
    )
    post_step_1_vorticity_field = np.zeros_like(vorticity_field)
    post_step_1_vorticity_field[...] = vorticity_field + vorticity_stretching_flux_field

    # second step
    vorticity_stretching_flux_field[...] = vorticity_stretching_flux_reference(
        post_step_1_vorticity_field, velocity_field, prefactor=dt_by_2_dx
    )
    post_step_1_vorticity_field[...] = (
        post_step_1_vorticity_field + vorticity_stretching_flux_field
    )
    post_step_2_vorticity_field = post_step_1_vorticity_field.view()
    post_step_2_vorticity_field[...] = (
        0.75 * vorticity_field + 0.25 * post_step_1_vorticity_field
    )

    # third step
    vorticity_stretching_flux_field[...] = vorticity_stretching_flux_reference(
        post_step_2_vorticity_field, velocity_field, prefactor=(0.5 * dt_by_2_dx)
    )
    post_step_2_vorticity_field[...] = (
        post_step_2_vorticity_field + vorticity_stretching_flux_field
    )
    new_vorticity_field = np.zeros_like(vorticity_field)
    new_vorticity_field[...] = (1.0 / 3.0) * vorticity_field + (
        2.0 / 3.0
    ) * post_step_2_vorticity_field
    return new_vorticity_field


class VorticityStretchingTimestepSolution:
    def __init__(self, n_samples, time_stepper="euler_forward", precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_vorticity_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.ref_velocity_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.dt_by_2_dx = real_t(0.1)
        if time_stepper == "euler_forward":
            self.ref_new_vorticity_field = (
                vorticity_stretching_timestep_euler_forward_reference(
                    self.ref_vorticity_field, self.ref_velocity_field, self.dt_by_2_dx
                )
            )
        elif time_stepper == "ssprk3":
            self.ref_new_vorticity_field = (
                vorticity_stretching_timestep_ssprk3_reference(
                    self.ref_vorticity_field, self.ref_velocity_field, self.dt_by_2_dx
                )
            )

    def check_equals(self, new_vorticity_field):
        np.testing.assert_allclose(
            self.ref_new_vorticity_field,
            new_vorticity_field,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vort_stretching_timestep_euler_forward_3d(n_values, precision):
    real_t = get_real_t(precision)
    solution = VorticityStretchingTimestepSolution(
        n_values, time_stepper="euler_forward", precision=precision
    )
    vorticity_field = solution.ref_vorticity_field.copy()
    vorticity_stretching_flux_field = np.ones_like(vorticity_field)
    vorticity_stretching_timestep_euler_forward_pyst_kernel_3d = (
        gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
        )
    )
    vorticity_stretching_timestep_euler_forward_pyst_kernel_3d(
        vorticity_field=vorticity_field,
        velocity_field=solution.ref_velocity_field,
        vorticity_stretching_flux_field=vorticity_stretching_flux_field,
        dt_by_2_dx=solution.dt_by_2_dx,
    )
    solution.check_equals(vorticity_field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vort_stretching_timestep_ssprk3_3d(n_values, precision):
    real_t = get_real_t(precision)
    solution = VorticityStretchingTimestepSolution(
        n_values, time_stepper="ssprk3", precision=precision
    )
    vorticity_field = solution.ref_vorticity_field.copy()
    vorticity_stretching_flux_field = np.ones_like(vorticity_field)
    vorticity_stretching_timestep_ssprk3_pyst_kernel_3d = (
        gen_vorticity_stretching_timestep_ssprk3_pyst_kernel_3d(
            real_t=real_t,
            midstep_buffer_vector_field=np.zeros_like(vorticity_field),
            fixed_grid_size=(n_values, n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
        )
    )
    vorticity_stretching_timestep_ssprk3_pyst_kernel_3d(
        vorticity_field=vorticity_field,
        velocity_field=solution.ref_velocity_field,
        vorticity_stretching_flux_field=vorticity_stretching_flux_field,
        dt_by_2_dx=solution.dt_by_2_dx,
    )
    solution.check_equals(vorticity_field)

import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol

from test_advection_flux_3d import advection_flux_conservative_eno3_reference


def advection_timestep_conservative_eno3_euler_forward_reference(
    field,
    velocity_x,
    velocity_y,
    velocity_z,
    inv_dx,
    dt,
    real_t,
):
    advection_flux = advection_flux_conservative_eno3_reference(
        field=field,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
        inv_dx=inv_dx,
        real_t=real_t,
    )
    new_field = field - dt * advection_flux
    return new_field


class AdvectionTimestepEulerForwardSolution:
    def __init__(self, n_samples, flux_type="conservative_eno3", precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = np.random.randn(n_samples, n_samples, n_samples).astype(real_t)
        self.ref_vector_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.ref_velocity = np.random.rand(3, n_samples, n_samples, n_samples).astype(
            real_t
        )
        self.inv_dx = real_t(0.2)
        self.dt = real_t(0.1)
        self.flux_type = flux_type
        self.ref_new_vector_field = np.zeros_like(self.ref_vector_field)
        if self.flux_type == "conservative_eno3":
            self.kernel_width = 2
            self.ref_new_field = (
                advection_timestep_conservative_eno3_euler_forward_reference(
                    field=self.ref_field,
                    velocity_x=self.ref_velocity[0],
                    velocity_y=self.ref_velocity[1],
                    velocity_z=self.ref_velocity[2],
                    inv_dx=self.inv_dx,
                    dt=self.dt,
                    real_t=real_t,
                )
            )
            for i in range(3):
                self.ref_new_vector_field[
                    i
                ] = advection_timestep_conservative_eno3_euler_forward_reference(
                    field=self.ref_vector_field[i],
                    velocity_x=self.ref_velocity[0],
                    velocity_y=self.ref_velocity[1],
                    velocity_z=self.ref_velocity[2],
                    inv_dx=self.inv_dx,
                    dt=self.dt,
                    real_t=real_t,
                )

    def check_equals(self, new_field):
        np.testing.assert_allclose(
            self.ref_new_field,
            new_field,
            atol=self.test_tol,
        )

    def check_vector_field_equals(self, new_vector_field):
        np.testing.assert_allclose(
            self.ref_new_vector_field,
            new_vector_field,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_advection_eno3_euler_forward(n_values, precision):
    real_t = get_real_t(precision)
    solution = AdvectionTimestepEulerForwardSolution(
        n_values,
        flux_type="conservative_eno3",
        precision=precision,
    )
    advection_flux = np.zeros_like(solution.ref_field)
    dt_by_dx = real_t(solution.dt * solution.inv_dx)
    advection_timestep_euler_forward_conservative_eno3_kernel_3d = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
            field_type="scalar",
        )
    )
    advection_timestep_euler_forward_conservative_eno3_kernel_3d(
        field=solution.ref_field,
        advection_flux=advection_flux,
        velocity=solution.ref_velocity,
        dt_by_dx=dt_by_dx,
    )
    solution.check_equals(solution.ref_field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vector_field_advection_timestep_eno3_3d(n_values, precision):
    real_t = get_real_t(precision)
    solution = AdvectionTimestepEulerForwardSolution(
        n_values,
        flux_type="conservative_eno3",
        precision=precision,
    )
    vector_field = solution.ref_vector_field.copy()
    advection_flux = np.zeros((n_values, n_values, n_values), dtype=real_t)
    dt_by_dx = real_t(solution.dt * solution.inv_dx)
    vector_field_advection_timestep_euler_forward_pyst_kernel_3d = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
            field_type="vector",
        )
    )
    vector_field_advection_timestep_euler_forward_pyst_kernel_3d(
        vector_field=vector_field,
        advection_flux=advection_flux,
        velocity=solution.ref_velocity,
        dt_by_dx=dt_by_dx,
    )
    solution.check_vector_field_equals(vector_field)

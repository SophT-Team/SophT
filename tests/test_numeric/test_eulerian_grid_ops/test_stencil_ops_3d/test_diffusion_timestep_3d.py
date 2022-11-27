import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_diffusion_timestep_euler_forward_pyst_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def diffusion_timestep_euler_forward_reference(field, nu_dt_by_dx2, real_t):
    diffusion_flux = np.zeros_like(field)
    diffusion_flux[1:-1, 1:-1, 1:-1] = nu_dt_by_dx2 * (
        field[1:-1, 1:-1, 2:]
        + field[1:-1, 1:-1, :-2]
        + field[1:-1, 2:, 1:-1]
        + field[1:-1, :-2, 1:-1]
        + field[2:, 1:-1, 1:-1]
        + field[:-2, 1:-1, 1:-1]
        - real_t(6) * field[1:-1, 1:-1, 1:-1]
    )
    new_field = field + diffusion_flux
    return new_field


class DiffusionTimestepEulerForwardSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = np.random.randn(n_samples, n_samples, n_samples).astype(real_t)
        self.nu_dt_by_dx2 = real_t(0.1)
        self.ref_new_field = diffusion_timestep_euler_forward_reference(
            self.ref_field,
            self.nu_dt_by_dx2,
            real_t,
        )

        self.ref_vector_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.ref_new_vector_field = np.zeros_like(self.ref_vector_field)
        self.ref_new_vector_field[0] = diffusion_timestep_euler_forward_reference(
            self.ref_vector_field[0],
            self.nu_dt_by_dx2,
            real_t,
        )
        self.ref_new_vector_field[1] = diffusion_timestep_euler_forward_reference(
            self.ref_vector_field[1],
            self.nu_dt_by_dx2,
            real_t,
        )
        self.ref_new_vector_field[2] = diffusion_timestep_euler_forward_reference(
            self.ref_vector_field[2],
            self.nu_dt_by_dx2,
            real_t,
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
def test_diffusion_timestep_euler_forward_3d(n_values, precision):
    real_t = get_real_t(precision)
    solution = DiffusionTimestepEulerForwardSolution(n_values, precision)
    field = solution.ref_field.copy()
    diffusion_flux = np.ones_like(field)
    diffusion_timestep_euler_forward_pyst_kernel = (
        gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
            field_type="scalar",
        )
    )
    diffusion_timestep_euler_forward_pyst_kernel(
        field=field,
        diffusion_flux=diffusion_flux,
        nu_dt_by_dx2=solution.nu_dt_by_dx2,
    )
    solution.check_equals(field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vector_field_diffusion_timestep_euler_forward_3d(n_values, precision):
    real_t = get_real_t(precision)
    solution = DiffusionTimestepEulerForwardSolution(n_values, precision)
    vector_field = solution.ref_vector_field.copy()
    diffusion_flux = np.ones((n_values, n_values, n_values), dtype=real_t)
    vector_field_diffusion_timestep_euler_forward_pyst_kernel = (
        gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
            field_type="vector",
        )
    )
    vector_field_diffusion_timestep_euler_forward_pyst_kernel(
        vector_field=vector_field,
        diffusion_flux=diffusion_flux,
        nu_dt_by_dx2=solution.nu_dt_by_dx2,
    )
    solution.check_vector_field_equals(vector_field)

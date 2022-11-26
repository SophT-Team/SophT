import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_diffusion_timestep_euler_forward_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t, get_test_tol

from test_diffusion_flux_2d import diffusion_flux_reference


def diffusion_timestep_euler_forward_reference(field, nu_dt_by_dx2, real_t):
    diffusion_flux = diffusion_flux_reference(
        field, prefactor=nu_dt_by_dx2, real_t=real_t
    )
    new_field = field + diffusion_flux
    return new_field


class DiffusionTimestepSolution:
    def __init__(self, n_samples, timestepper="euler_forward", precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = np.random.randn(n_samples, n_samples).astype(real_t)
        self.nu_dt_by_dx2 = real_t(0.1)
        if timestepper == "euler_forward":
            self.ref_new_field = diffusion_timestep_euler_forward_reference(
                self.ref_field,
                self.nu_dt_by_dx2,
                real_t=real_t,
            )

    def check_equals(self, new_field):
        np.testing.assert_allclose(
            self.ref_new_field,
            new_field,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_diffusion_timestep_euler_forward_2d(n_values, precision):
    real_t = get_real_t(precision)
    solution = DiffusionTimestepSolution(n_values, precision=precision)
    field = solution.ref_field.copy()
    diffusion_flux = np.ones_like(field)
    diffusion_timestep_euler_forward_pyst_kernel = (
        gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
        )
    )
    diffusion_timestep_euler_forward_pyst_kernel(
        field=field,
        diffusion_flux=diffusion_flux,
        nu_dt_by_dx2=solution.nu_dt_by_dx2,
    )
    solution.check_equals(field)

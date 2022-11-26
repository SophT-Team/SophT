import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_diffusion_flux_pyst_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def diffuse_reference(field, prefactor, real_t):
    diffusion_flux = np.zeros_like(field)
    diffusion_flux[1:-1, 1:-1, 1:-1] = prefactor * (
        field[1:-1, 1:-1, 2:]
        + field[1:-1, 1:-1, :-2]
        + field[1:-1, 2:, 1:-1]
        + field[1:-1, :-2, 1:-1]
        + field[2:, 1:-1, 1:-1]
        + field[:-2, 1:-1, 1:-1]
        - real_t(6) * field[1:-1, 1:-1, 1:-1]
    )
    return diffusion_flux


class DiffusionFluxSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = np.random.randn(n_samples, n_samples, n_samples).astype(real_t)
        self.prefactor = real_t(0.1)
        self.ref_diffusion_flux = diffuse_reference(
            self.ref_field, self.prefactor, real_t
        )

        self.ref_vector_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.ref_vector_field_diffusion_flux = np.zeros_like(self.ref_vector_field)
        self.ref_vector_field_diffusion_flux[0] = diffuse_reference(
            self.ref_vector_field[0],
            self.prefactor,
            real_t,
        )
        self.ref_vector_field_diffusion_flux[1] = diffuse_reference(
            self.ref_vector_field[1],
            self.prefactor,
            real_t,
        )
        self.ref_vector_field_diffusion_flux[2] = diffuse_reference(
            self.ref_vector_field[2],
            self.prefactor,
            real_t,
        )

    @property
    def ref_rhs(self):
        return self.ref_diffusion_flux

    def get(self):
        return (self.ref_field, self.prefactor)

    def check_equals(self, diffusion_flux):
        np.testing.assert_allclose(
            self.ref_diffusion_flux,
            diffusion_flux,
            atol=self.test_tol,
        )

    def check_vector_field_equals(self, vector_field_diffusion_flux):
        np.testing.assert_allclose(
            self.ref_vector_field_diffusion_flux,
            vector_field_diffusion_flux,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("reset_ghost_zone", [True, False])
def test_diffusion_flux_3d(n_values, precision, reset_ghost_zone):
    real_t = get_real_t(precision)
    solution = DiffusionFluxSolution(n_values, precision)
    diffusion_flux = (
        np.ones_like(solution.ref_diffusion_flux)
        if reset_ghost_zone
        else np.zeros_like(solution.ref_diffusion_flux)
    )
    diffusion_flux_pyst_openmp_kernel_3d = gen_diffusion_flux_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="scalar",
        reset_ghost_zone=reset_ghost_zone,
    )
    diffusion_flux_pyst_openmp_kernel_3d(
        diffusion_flux=diffusion_flux,
        field=solution.ref_field,
        prefactor=solution.prefactor,
    )
    solution.check_equals(diffusion_flux)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("reset_ghost_zone", [True, False])
def test_vector_field_diffusion_flux_3d(n_values, precision, reset_ghost_zone):
    real_t = get_real_t(precision)
    solution = DiffusionFluxSolution(n_values, precision)
    vector_field_diffusion_flux = (
        np.ones_like(solution.ref_vector_field_diffusion_flux)
        if reset_ghost_zone
        else np.zeros_like(solution.ref_vector_field_diffusion_flux)
    )
    vector_field_diffusion_flux_pyst_kernel_3d = gen_diffusion_flux_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="vector",
        reset_ghost_zone=reset_ghost_zone,
    )
    vector_field_diffusion_flux_pyst_kernel_3d(
        vector_field_diffusion_flux=vector_field_diffusion_flux,
        vector_field=solution.ref_vector_field,
        prefactor=solution.prefactor,
    )
    solution.check_vector_field_equals(vector_field_diffusion_flux)

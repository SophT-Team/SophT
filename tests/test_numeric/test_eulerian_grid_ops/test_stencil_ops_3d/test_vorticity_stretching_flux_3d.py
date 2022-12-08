import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_vorticity_stretching_flux_pyst_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def vorticity_stretching_flux_reference(vorticity_field, velocity_field, prefactor):
    vorticity_stretching_flux_field = np.zeros_like(vorticity_field)
    vorticity_stretching_flux_field[0, 1:-1, 1:-1, 1:-1] = prefactor * (
        vorticity_field[2, 1:-1, 1:-1, 1:-1]
        * (velocity_field[0, 2:, 1:-1, 1:-1] - velocity_field[0, :-2, 1:-1, 1:-1])
        + vorticity_field[1, 1:-1, 1:-1, 1:-1]
        * (velocity_field[0, 1:-1, 2:, 1:-1] - velocity_field[0, 1:-1, :-2, 1:-1])
        + vorticity_field[0, 1:-1, 1:-1, 1:-1]
        * (velocity_field[0, 1:-1, 1:-1, 2:] - velocity_field[0, 1:-1, 1:-1, :-2])
    )
    vorticity_stretching_flux_field[1, 1:-1, 1:-1, 1:-1] = prefactor * (
        vorticity_field[2, 1:-1, 1:-1, 1:-1]
        * (velocity_field[1, 2:, 1:-1, 1:-1] - velocity_field[1, :-2, 1:-1, 1:-1])
        + vorticity_field[1, 1:-1, 1:-1, 1:-1]
        * (velocity_field[1, 1:-1, 2:, 1:-1] - velocity_field[1, 1:-1, :-2, 1:-1])
        + vorticity_field[0, 1:-1, 1:-1, 1:-1]
        * (velocity_field[1, 1:-1, 1:-1, 2:] - velocity_field[1, 1:-1, 1:-1, :-2])
    )
    vorticity_stretching_flux_field[2, 1:-1, 1:-1, 1:-1] = prefactor * (
        vorticity_field[2, 1:-1, 1:-1, 1:-1]
        * (velocity_field[2, 2:, 1:-1, 1:-1] - velocity_field[2, :-2, 1:-1, 1:-1])
        + vorticity_field[1, 1:-1, 1:-1, 1:-1]
        * (velocity_field[2, 1:-1, 2:, 1:-1] - velocity_field[2, 1:-1, :-2, 1:-1])
        + vorticity_field[0, 1:-1, 1:-1, 1:-1]
        * (velocity_field[2, 1:-1, 1:-1, 2:] - velocity_field[2, 1:-1, 1:-1, :-2])
    )
    return vorticity_stretching_flux_field


class VorticityStretchingFluxSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_vorticity_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.ref_velocity_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.prefactor = real_t(0.1)
        self.ref_vorticity_stretching_flux_field = vorticity_stretching_flux_reference(
            self.ref_vorticity_field, self.ref_velocity_field, self.prefactor
        )

    def check_equals(self, vorticity_stretching_flux_field):
        np.testing.assert_allclose(
            self.ref_vorticity_stretching_flux_field,
            vorticity_stretching_flux_field,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("reset_ghost_zone", [True, False])
def test_vort_stretching_flux_3d(n_values, precision, reset_ghost_zone):
    real_t = get_real_t(precision)
    solution = VorticityStretchingFluxSolution(n_values, precision)
    vorticity_stretching_flux_field = (
        np.ones_like(solution.ref_vorticity_stretching_flux_field)
        if reset_ghost_zone
        else np.zeros_like(solution.ref_vorticity_stretching_flux_field)
    )
    vorticity_stretching_flux_kernel_3d = gen_vorticity_stretching_flux_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        reset_ghost_zone=reset_ghost_zone,
    )
    vorticity_stretching_flux_kernel_3d(
        vorticity_stretching_flux_field=vorticity_stretching_flux_field,
        vorticity_field=solution.ref_vorticity_field,
        velocity_field=solution.ref_velocity_field,
        prefactor=solution.prefactor,
    )
    solution.check_equals(vorticity_stretching_flux_field)

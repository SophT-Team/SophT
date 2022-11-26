import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_outplane_field_curl_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def outplane_curl_reference(field, prefactor, real_t):
    curl = np.zeros((2, field.shape[0], field.shape[1]), dtype=real_t)
    # curl_x = d (field) / dy
    curl[0, 1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) * prefactor
    # curl_y = -d (field) / dx
    curl[1, 1:-1, 1:-1] = (field[1:-1, :-2] - field[1:-1, 2:]) * prefactor
    return curl


class OutplaneCurlSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = np.random.randn(n_samples, n_samples).astype(real_t)
        self.prefactor = real_t(0.1)
        self.ref_curl = outplane_curl_reference(
            self.ref_field, self.prefactor, real_t=real_t
        )

    def check_equals(self, curl):
        np.testing.assert_allclose(self.ref_curl, curl, atol=self.test_tol)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("reset_ghost_zone", [True, False])
def test_outplane_field_curl(n_values, precision, reset_ghost_zone):
    real_t = get_real_t(precision)
    solution = OutplaneCurlSolution(n_values, precision)
    curl = (
        np.ones_like(solution.ref_curl)
        if reset_ghost_zone
        else np.zeros_like(solution.ref_curl)
    )
    outplane_field_curl_pyst_kernel = gen_outplane_field_curl_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        reset_ghost_zone=reset_ghost_zone,
    )
    outplane_field_curl_pyst_kernel(
        curl=curl,
        field=solution.ref_field,
        prefactor=solution.prefactor,
    )
    solution.check_equals(curl)

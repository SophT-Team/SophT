import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_inplane_field_curl_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def inplane_curl_reference(field_x, field_y, prefactor):
    curl = np.zeros_like(field_x)
    curl[1:-1, 1:-1] = (
        field_y[1:-1, 2:] - field_y[1:-1, :-2] - field_x[2:, 1:-1] + field_x[:-2, 1:-1]
    ) * prefactor

    return curl


class InplaneCurlSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field_x = np.random.randn(n_samples, n_samples).astype(real_t)
        self.ref_field_y = np.random.randn(n_samples, n_samples).astype(real_t)
        self.ref_field = np.zeros((2, n_samples, n_samples)).astype(real_t)
        self.ref_field[0] = self.ref_field_x
        self.ref_field[1] = self.ref_field_y
        # pystencil vector field needs to be (n, n, 2)
        self.transposed_ref_field = np.transpose(self.ref_field, axes=(1, 2, 0))
        self.prefactor = real_t(0.1)
        self.ref_curl = inplane_curl_reference(
            self.ref_field_x, self.ref_field_y, self.prefactor
        )

    def check_equals(self, curl):
        np.testing.assert_allclose(self.ref_curl, curl, atol=self.test_tol)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_inplane_field_curl(n_values, precision):
    real_t = get_real_t(precision)
    solution = InplaneCurlSolution(n_values, precision)
    curl = np.zeros_like(solution.ref_curl)
    inplane_field_curl_kernel = gen_inplane_field_curl_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
    )
    inplane_field_curl_kernel(
        curl=curl,
        field=solution.ref_field,
        prefactor=solution.prefactor,
    )
    solution.check_equals(curl)

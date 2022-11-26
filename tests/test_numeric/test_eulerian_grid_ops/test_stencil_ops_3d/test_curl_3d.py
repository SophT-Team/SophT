import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_curl_pyst_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def curl_reference(field_x, field_y, field_z, prefactor):
    curl_x = np.zeros_like(field_x)
    curl_y = np.zeros_like(field_y)
    curl_z = np.zeros_like(field_z)
    # curl_x = df_z / dy - df_y / dz
    curl_x[1:-1, 1:-1, 1:-1] = (
        field_z[1:-1, 2:, 1:-1]
        - field_z[1:-1, :-2, 1:-1]
        - field_y[2:, 1:-1, 1:-1]
        + field_y[:-2, 1:-1, 1:-1]
    ) * prefactor
    # curl_y = df_x / dz - df_z / dx
    curl_y[1:-1, 1:-1, 1:-1] = (
        field_x[2:, 1:-1, 1:-1]
        - field_x[:-2, 1:-1, 1:-1]
        - field_z[1:-1, 1:-1, 2:]
        + field_z[1:-1, 1:-1, :-2]
    ) * prefactor
    # curl_z = df_y / dx - df_x / dy
    curl_z[1:-1, 1:-1, 1:-1] = (
        field_y[1:-1, 1:-1, 2:]
        - field_y[1:-1, 1:-1, :-2]
        - field_x[1:-1, 2:, 1:-1]
        + field_x[1:-1, :-2, 1:-1]
    ) * prefactor

    return (curl_x, curl_y, curl_z)


class CurlSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field_x = np.random.randn(n_samples, n_samples, n_samples).astype(
            real_t
        )
        self.ref_field_y = np.random.randn(n_samples, n_samples, n_samples).astype(
            real_t
        )
        self.ref_field_z = np.random.randn(n_samples, n_samples, n_samples).astype(
            real_t
        )
        self.prefactor = real_t(0.1)
        self.ref_curl_x, self.ref_curl_y, self.ref_curl_z = curl_reference(
            self.ref_field_x, self.ref_field_y, self.ref_field_z, self.prefactor
        )
        # 3D vector field for dimension independent kernel tests, ideally
        # this is the only that should be kept, but keeping the original code for
        # now, to prevent the existing tests from failing
        self.ref_field = np.zeros((3, n_samples, n_samples, n_samples), dtype=real_t)
        self.ref_field[0] = self.ref_field_x
        self.ref_field[1] = self.ref_field_y
        self.ref_field[2] = self.ref_field_z
        self.ref_curl = np.zeros((3, n_samples, n_samples, n_samples), dtype=real_t)
        self.ref_curl[0] = self.ref_curl_x
        self.ref_curl[1] = self.ref_curl_y
        self.ref_curl[2] = self.ref_curl_z

    def check_field_equals(self, curl):
        np.testing.assert_allclose(
            self.ref_curl,
            curl,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("reset_ghost_zone", [True, False])
def test_curl_3d(n_values, precision, reset_ghost_zone):
    real_t = get_real_t(precision)
    solution = CurlSolution(n_values, precision)
    curl = (
        np.ones_like(solution.ref_curl)
        if reset_ghost_zone
        else np.zeros_like(solution.ref_curl)
    )
    curl_pyst_kernel_3d = gen_curl_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        reset_ghost_zone=reset_ghost_zone,
    )
    curl_pyst_kernel_3d(
        curl=curl,
        field=solution.ref_field,
        prefactor=solution.prefactor,
    )
    solution.check_field_equals(curl)

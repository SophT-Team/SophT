import numpy as np

import psutil

import pytest

from scipy.fft import irfftn, rfftn

from sopht.numeric.eulerian_grid_ops import FFTPyFFTW3D
from sopht.numeric.eulerian_grid_ops import (
    fft_ifft_via_scipy_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def fft_ifft_reference(field):
    fourier_field = rfftn(field)
    inv_fourier_field = irfftn(fourier_field)
    return fourier_field, inv_fourier_field


class FFTSolution:
    def __init__(self, n_samples, precision="single"):
        self.real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = np.random.randn(n_samples, n_samples, n_samples).astype(
            self.real_t
        )
        self.ref_fourier_field, self.ref_inv_fourier_field = fft_ifft_reference(
            self.ref_field
        )

    @property
    def ref_rhs(self):
        return (self.ref_fourier_field, self.ref_inv_fourier_field)

    def check_equals(self, fourier_field, inv_fourier_field):
        np.testing.assert_allclose(
            self.ref_fourier_field, fourier_field, atol=self.test_tol
        )
        np.testing.assert_allclose(
            self.ref_inv_fourier_field, inv_fourier_field, atol=self.test_tol
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [8])
def test_scipy_fft_3d(n_values, precision):
    solution = FFTSolution(n_values, precision=precision)
    fourier_field = np.zeros_like(solution.ref_fourier_field)
    inv_fourier_field = np.zeros_like(solution.ref_inv_fourier_field)
    max_num_threads = psutil.cpu_count(logical=False)
    fft_ifft_via_scipy_kernel_3d(
        fourier_field,
        inv_fourier_field,
        solution.ref_field,
        num_threads=max_num_threads,
    )
    # assert correct
    solution.check_equals(fourier_field, inv_fourier_field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [8])
def test_pyfftw_fft_3d(n_values, precision):
    real_t = get_real_t(precision)
    solution = FFTSolution(n_values, precision=precision)
    max_num_threads = psutil.cpu_count(logical=False)
    pyfftw = FFTPyFFTW3D(
        n_values, n_values, n_values, num_threads=max_num_threads, real_t=real_t
    )
    fourier_field = np.zeros_like(solution.ref_fourier_field)
    inv_fourier_field = np.zeros_like(solution.ref_inv_fourier_field)
    pyfftw.fft_plan(input_array=solution.ref_field, output_array=fourier_field)
    # a copy is passed since complex to real transforms destroy input field
    pyfftw.ifft_plan(input_array=fourier_field.copy(), output_array=inv_fourier_field)

    solution.check_equals(fourier_field, inv_fourier_field)

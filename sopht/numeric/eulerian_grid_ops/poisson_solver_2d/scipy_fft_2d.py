"""Create reference FFT operations via scipy in 2D."""
import numpy as np
from scipy.fft import irfftn, rfftn


def fft_ifft_via_scipy_kernel_2d(
    fourier_field: np.ndarray,
    inv_fourier_field: np.ndarray,
    field: np.ndarray,
    num_threads: int = 1,
) -> None:
    """Perform reference FFT operations via scipy."""
    fourier_field[...] = rfftn(field, workers=num_threads)
    inv_fourier_field[...] = irfftn(fourier_field, workers=num_threads)

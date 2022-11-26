"""Create reference FFT operations via scipy in 3D."""
from scipy.fft import irfftn, rfftn


def fft_ifft_via_scipy_kernel_3d(
    fourier_field, inv_fourier_field, field, num_threads=1
):
    """Perform reference FFT operations via scipy."""
    fourier_field[...] = rfftn(field, workers=num_threads)
    inv_fourier_field[...] = irfftn(fourier_field, workers=num_threads)

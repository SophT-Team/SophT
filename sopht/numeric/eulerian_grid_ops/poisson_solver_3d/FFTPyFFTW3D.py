"""Class for performing FFT via PyFFTW in 3D."""
import numpy as np

import pyfftw


class FFTPyFFTW3D:
    """Class for performing FFT via PyFFTW in 3D."""

    def __init__(
        self,
        grid_size_z,
        grid_size_y,
        grid_size_x,
        num_threads=1,
        real_t=np.float64,
    ):
        """Class initialiser."""
        self.grid_size_z = grid_size_z
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.num_threads = num_threads
        self.real_t = real_t
        self.complex_dtype = np.complex64 if real_t == np.float32 else np.complex128
        self._create_fftw_plan()

    def _create_fftw_plan(self):
        """Create FFTW plan objects necessary for executing FFT later."""
        self.field_pyfftw_buffer = pyfftw.empty_aligned(
            (self.grid_size_z, self.grid_size_y, self.grid_size_x),
            dtype=self.real_t,
        )
        self.fourier_field_pyfftw_buffer = pyfftw.empty_aligned(
            (self.grid_size_z, self.grid_size_y, self.grid_size_x // 2 + 1),
            dtype=self.complex_dtype,
        )
        self.fft_plan = pyfftw.FFTW(
            self.field_pyfftw_buffer,
            self.fourier_field_pyfftw_buffer,
            axes=(0, 1, 2),
            direction="FFTW_FORWARD",
            flags=("FFTW_MEASURE",),
            threads=self.num_threads,
        )
        self.ifft_plan = pyfftw.FFTW(
            self.fourier_field_pyfftw_buffer,
            self.field_pyfftw_buffer,
            axes=(0, 1, 2),
            direction="FFTW_BACKWARD",
            flags=("FFTW_MEASURE",),
            threads=self.num_threads,
        )
        self.pyfftw_fftn = pyfftw.builders.fftn(
            self.field_pyfftw_buffer, threads=self.num_threads
        )
        self.pyfftw_ifftn = pyfftw.builders.ifftn(
            self.fourier_field_pyfftw_buffer, threads=self.num_threads
        )

    def fft_ifft_plan_kernel(self, fourier_field, inv_fourier_field, field):
        """Perform forward and backward transforms."""
        # Only used for benchmarking fft and ifft together
        self.fft_plan(input_array=field, output_array=fourier_field)
        self.ifft_plan(input_array=fourier_field, output_array=inv_fourier_field)

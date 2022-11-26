"""Unbounded Poisson solver kernels 2D via PyFFTW."""
import numpy as np

from sopht.numeric.eulerian_grid_ops.poisson_solver_2d.FFTPyFFTW2D import FFTPyFFTW2D
from sopht.numeric.eulerian_grid_ops.stencil_ops_2d.elementwise_ops_2d import (
    gen_elementwise_complex_product_pyst_kernel_2d,
    gen_elementwise_copy_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
)


class UnboundedPoissonSolverPYFFTW2D:
    """Class for solving unbounded Poisson in 2D via PyFFTW."""

    def __init__(
        self,
        grid_size_y,
        grid_size_x,
        x_range=1.0,
        num_threads=1,
        real_t=np.float64,
    ):
        """Class initialiser."""
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.x_range = x_range
        self.y_range = x_range * (grid_size_y / grid_size_x)
        self.dx = real_t(x_range / grid_size_x)
        self.num_threads = num_threads
        self.real_t = real_t
        pyfftw_construct = FFTPyFFTW2D(
            # 2 because FFTs taken on doubled domain
            grid_size_y=2 * grid_size_y,
            grid_size_x=2 * grid_size_x,
            num_threads=num_threads,
            real_t=real_t,
        )
        self.rfft = pyfftw_construct.fft_plan
        self.irfft = pyfftw_construct.ifft_plan
        self.domain_doubled_buffer = pyfftw_construct.field_pyfftw_buffer
        self.domain_doubled_fourier_buffer = (
            pyfftw_construct.fourier_field_pyfftw_buffer
        )
        # TODO avoid this allocation if possible, currently needed to do SIMD and
        #  parallel fourier space convolution
        self.convolution_buffer = np.zeros_like(self.domain_doubled_fourier_buffer)
        self.construct_fourier_greens_function_field()
        self.fourier_greens_function_times_dx_squared = (
            self.domain_doubled_fourier_buffer * (self.dx**2)
        )
        self.gen_elementwise_operation_kernels()

    def construct_fourier_greens_function_field(self):
        """Construct the unbounded Greens function."""
        x_double = np.linspace(
            0, 2 * self.x_range - self.dx, 2 * self.grid_size_x
        ).astype(self.real_t)
        y_double = np.linspace(
            0, 2 * self.y_range - self.dx, 2 * self.grid_size_y
        ).astype(self.real_t)
        # operations after this preserve dtype
        x_grid_double, y_grid_double = np.meshgrid(x_double, y_double)
        even_reflected_distance_field = np.sqrt(
            np.minimum(x_grid_double, 2 * self.x_range - x_grid_double) ** 2
            + np.minimum(y_grid_double, 2 * self.y_range - y_grid_double) ** 2
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            greens_function_field = -np.log(even_reflected_distance_field) / (2 * np.pi)
        # Regularization term
        greens_function_field[0, 0] = -(2 * np.log(self.dx / np.sqrt(np.pi)) - 1) / (
            4 * np.pi
        )
        self.rfft(
            input_array=greens_function_field,
            output_array=self.domain_doubled_fourier_buffer,
        )

    def gen_elementwise_operation_kernels(self):
        """Compile funcs for elementwise ops on buffers."""
        # both of these operate on domain doubled arrays
        self.set_fixed_val_kernel_2d = gen_set_fixed_val_pyst_kernel_2d(
            real_t=self.real_t,
            num_threads=self.num_threads,
            fixed_grid_size=(
                self.domain_doubled_buffer.shape[0],
                self.domain_doubled_buffer.shape[1],
            ),
        )
        # TODO add kernel strides info to enable fixed size version
        self.elementwise_copy_kernel_2d = gen_elementwise_copy_pyst_kernel_2d(
            real_t=self.real_t,
            num_threads=self.num_threads,
        )
        # this one operates on fourier buffer
        # TODO add kernel strides info to enable fixed size version
        self.elementwise_complex_product_kernel_2d = (
            gen_elementwise_complex_product_pyst_kernel_2d(
                real_t=self.real_t,
                num_threads=self.num_threads,
            )
        )

    def solve(self, solution_field, rhs_field):
        """Unbounded Poisson solver method.

        Solves Poisson equation in 2D: -del^2(solution_field) = rhs_field
        for unbounded domain using Greens function convolution and
        domain doubling trick (Hockney and Eastwood).
        """
        self.set_fixed_val_kernel_2d(field=self.domain_doubled_buffer, fixed_val=0)

        self.elementwise_copy_kernel_2d(
            field=self.domain_doubled_buffer[: self.grid_size_y, : self.grid_size_x],
            rhs_field=rhs_field,
        )

        self.rfft(
            input_array=self.domain_doubled_buffer,
            output_array=self.domain_doubled_fourier_buffer,
        )

        # Greens function convolution
        self.elementwise_complex_product_kernel_2d(
            product_field=self.convolution_buffer,
            field_1=self.domain_doubled_fourier_buffer,
            field_2=self.fourier_greens_function_times_dx_squared,
        )

        self.irfft(
            input_array=self.convolution_buffer,
            output_array=self.domain_doubled_buffer,
        )

        self.elementwise_copy_kernel_2d(
            field=solution_field,
            rhs_field=self.domain_doubled_buffer[
                : self.grid_size_y, : self.grid_size_x
            ],
        )

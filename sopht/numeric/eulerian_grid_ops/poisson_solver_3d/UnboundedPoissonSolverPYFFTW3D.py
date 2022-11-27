"""Unbounded Poisson solver kernels 3D via PyFFTW."""
import numpy as np

from sopht.numeric.eulerian_grid_ops.poisson_solver_3d.FFTPyFFTW3D import FFTPyFFTW3D
from sopht.numeric.eulerian_grid_ops.stencil_ops_3d.elementwise_ops_3d import (
    gen_elementwise_complex_product_pyst_kernel_3d,
    gen_elementwise_copy_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
)


class UnboundedPoissonSolverPYFFTW3D:
    """Class for solving unbounded Poisson in 3D via PyFFTW."""

    def __init__(
        self,
        grid_size_z,
        grid_size_y,
        grid_size_x,
        x_range=1.0,
        num_threads=1,
        real_t=np.float64,
    ):
        """Class initialiser."""
        self.grid_size_z = grid_size_z
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.x_range = x_range
        self.y_range = x_range * (grid_size_y / grid_size_x)
        self.z_range = x_range * (grid_size_z / grid_size_x)
        self.dx = real_t(x_range / grid_size_x)
        self.num_threads = num_threads
        self.real_t = real_t
        pyfftw_construct = FFTPyFFTW3D(
            # 2 because FFTs taken on doubled domain
            grid_size_z=2 * grid_size_z,
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
        self.fourier_greens_function_times_dx_cubed = (
            self.domain_doubled_fourier_buffer * (self.dx**3)
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
        z_double = np.linspace(
            0, 2 * self.z_range - self.dx, 2 * self.grid_size_z
        ).astype(self.real_t)
        # operations after this preserve dtype
        z_grid_double, y_grid_double, x_grid_double = np.meshgrid(
            z_double, y_double, x_double, indexing="ij"
        )
        even_reflected_distance_field = np.sqrt(
            np.minimum(x_grid_double, 2 * self.x_range - x_grid_double) ** 2
            + np.minimum(y_grid_double, 2 * self.y_range - y_grid_double) ** 2
            + np.minimum(z_grid_double, 2 * self.z_range - z_grid_double) ** 2
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            greens_function_field = (1 / even_reflected_distance_field) / (4 * np.pi)
        # Regularization term (straight from PPM)
        greens_function_field[0, 0, 0] = 1 / (4 * np.pi * self.dx)
        self.rfft(
            input_array=greens_function_field,
            output_array=self.domain_doubled_fourier_buffer,
        )

    def gen_elementwise_operation_kernels(self):
        """Compile funcs for elementwise ops on buffers."""
        # both of these operate on domain doubled arrays
        self.set_fixed_val_kernel_3d = gen_set_fixed_val_pyst_kernel_3d(
            real_t=self.real_t,
            num_threads=self.num_threads,
            fixed_grid_size=(
                self.domain_doubled_buffer.shape[0],
                self.domain_doubled_buffer.shape[1],
                self.domain_doubled_buffer.shape[2],
            ),
        )
        # TODO add kernel strides info to enable fixed size version
        self.elementwise_copy_kernel_3d = gen_elementwise_copy_pyst_kernel_3d(
            real_t=self.real_t,
            num_threads=self.num_threads,
        )
        # this one operates on fourier buffer
        # TODO add kernel strides info to enable fixed size version
        self.elementwise_complex_product_kernel_3d = (
            gen_elementwise_complex_product_pyst_kernel_3d(
                real_t=self.real_t,
                num_threads=self.num_threads,
            )
        )

    def solve(self, solution_field, rhs_field):
        """Unbounded Poisson solver method.

        Solves Poisson equation in 3D: -del^2(solution_field) = rhs_field
        for unbounded domain using Greens function convolution and
        domain doubling trick (Hockney and Eastwood).
        """
        self.set_fixed_val_kernel_3d(field=self.domain_doubled_buffer, fixed_val=0)

        self.elementwise_copy_kernel_3d(
            field=self.domain_doubled_buffer[
                : self.grid_size_z, : self.grid_size_y, : self.grid_size_x
            ],
            rhs_field=rhs_field,
        )

        self.rfft(
            input_array=self.domain_doubled_buffer,
            output_array=self.domain_doubled_fourier_buffer,
        )

        # Greens function convolution
        self.elementwise_complex_product_kernel_3d(
            product_field=self.convolution_buffer,
            field_1=self.domain_doubled_fourier_buffer,
            field_2=self.fourier_greens_function_times_dx_cubed,
        )

        self.irfft(
            input_array=self.convolution_buffer,
            output_array=self.domain_doubled_buffer,
        )

        self.elementwise_copy_kernel_3d(
            field=solution_field,
            rhs_field=self.domain_doubled_buffer[
                : self.grid_size_z, : self.grid_size_y, : self.grid_size_x
            ],
        )

    def vector_field_solve(self, solution_vector_field, rhs_vector_field):
        """Unbounded Poisson solver method (vector field solve).

        Solves 3 Poisson equations in 3D for each component:
        -del^2(solution_vector_field) = rhs_vector_field
        for unbounded domain using Greens function convolution and
        domain doubling trick (Hockney and Eastwood).
        """
        self.solve(
            solution_field=solution_vector_field[0], rhs_field=rhs_vector_field[0]
        )
        self.solve(
            solution_field=solution_vector_field[1], rhs_field=rhs_vector_field[1]
        )
        self.solve(
            solution_field=solution_vector_field[2], rhs_field=rhs_vector_field[2]
        )

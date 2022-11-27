import numpy as np

import psutil

import pytest

from scipy.fft import irfftn, rfftn

from sopht.numeric.eulerian_grid_ops import (
    FastDiagPoissonSolver3D,
)
from sopht.numeric.eulerian_grid_ops import (
    UnboundedPoissonSolverPYFFTW3D,
)
from sopht.utils.precision import get_real_t, get_test_tol


class UnboundedPoissonSolverSolution3D:
    def __init__(
        self, grid_size_z, grid_size_y, grid_size_x, x_range, precision="single"
    ):
        self.real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.grid_size_z = grid_size_z
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.x_range = x_range
        self.y_range = x_range * (grid_size_y / grid_size_x)
        self.z_range = x_range * (grid_size_z / grid_size_x)
        self.dx = self.real_t(x_range / grid_size_x)
        self.rhs_field = np.random.randn(
            self.grid_size_z, self.grid_size_y, self.grid_size_x
        ).astype(self.real_t)
        self.domain_doubled_rhs_field = np.zeros(
            (2 * self.grid_size_z, 2 * self.grid_size_y, 2 * self.grid_size_x),
            dtype=self.real_t,
        )
        complex_dtype = np.complex64 if self.real_t == np.float32 else np.complex128
        self.domain_doubled_fourier_buffer = np.zeros(
            (2 * self.grid_size_z, 2 * self.grid_size_y, self.grid_size_x + 1),
            dtype=complex_dtype,
        )
        self.fourier_greens_function_field = (
            self.construct_fourier_greens_function_field()
        )
        self.ref_solution_field = np.zeros_like(self.rhs_field)
        self.ref_solution_field[...] = self.poisson_solve_reference(
            rhs_field=self.rhs_field
        )

        self.rhs_vector_field = np.random.randn(
            3, self.grid_size_z, self.grid_size_y, self.grid_size_x
        ).astype(self.real_t)
        self.ref_solution_vector_field = np.zeros_like(self.rhs_vector_field)
        for i in range(3):
            self.ref_solution_vector_field[i] = self.poisson_solve_reference(
                rhs_field=self.rhs_vector_field[i]
            )

    def construct_fourier_greens_function_field(self):
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
        return rfftn(greens_function_field)

    def poisson_solve_reference(self, rhs_field):
        self.domain_doubled_rhs_field[...] = 0
        self.domain_doubled_rhs_field[
            : self.grid_size_z, : self.grid_size_y, : self.grid_size_x
        ] = rhs_field
        self.domain_doubled_fourier_buffer[...] = rfftn(self.domain_doubled_rhs_field)
        # Greens function convolution
        self.domain_doubled_fourier_buffer[
            ...
        ] *= self.fourier_greens_function_field * (self.dx**3)
        return irfftn(self.domain_doubled_fourier_buffer)[
            : self.grid_size_z, : self.grid_size_y, : self.grid_size_x
        ]

    def check_equals(self, solution_field):
        np.testing.assert_allclose(
            self.ref_solution_field, solution_field, atol=self.test_tol
        )

    def check_vector_field_equals(self, solution_vector_field):
        np.testing.assert_allclose(
            self.ref_solution_vector_field, solution_vector_field, atol=self.test_tol
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_unbounded_poisson_solve_pyfftw_3d(n_values, precision):
    real_t = get_real_t(precision)
    x_range = real_t(2.0)
    solution = UnboundedPoissonSolverSolution3D(
        grid_size_z=n_values,
        grid_size_y=n_values,
        grid_size_x=n_values,
        x_range=x_range,
        precision=precision,
    )
    unbounded_poisson_solver = UnboundedPoissonSolverPYFFTW3D(
        grid_size_z=n_values,
        grid_size_y=n_values,
        grid_size_x=n_values,
        x_range=x_range,
        real_t=real_t,
        num_threads=psutil.cpu_count(logical=False),
    )
    solution_field = np.zeros_like(solution.rhs_field)
    unbounded_poisson_solver_kernel = unbounded_poisson_solver.solve
    unbounded_poisson_solver_kernel(
        solution_field=solution_field, rhs_field=solution.rhs_field
    )
    # assert correct
    solution.check_equals(solution_field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_unbounded_vector_field_poisson_solve_pyfftw_3d(n_values, precision):
    real_t = get_real_t(precision)
    x_range = real_t(2.0)
    solution = UnboundedPoissonSolverSolution3D(
        grid_size_z=n_values,
        grid_size_y=n_values,
        grid_size_x=n_values,
        x_range=x_range,
        precision=precision,
    )
    unbounded_poisson_solver = UnboundedPoissonSolverPYFFTW3D(
        grid_size_z=n_values,
        grid_size_y=n_values,
        grid_size_x=n_values,
        x_range=x_range,
        real_t=real_t,
        num_threads=psutil.cpu_count(logical=False),
    )
    solution_vector_field = np.zeros_like(solution.rhs_vector_field)
    unbounded_poisson_solver_kernel = unbounded_poisson_solver.vector_field_solve
    unbounded_poisson_solver_kernel(
        solution_vector_field=solution_vector_field,
        rhs_vector_field=solution.rhs_vector_field,
    )
    # assert correct
    solution.check_vector_field_equals(solution_vector_field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_unbounded_poisson_solve_neumann_fast_diag_3d(n_values, precision):
    real_t = get_real_t(precision)
    dx = real_t(1.0 / n_values)
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, n_values).astype(real_t)
    z_grid, y_grid, x_grid = np.meshgrid(x, x, x, indexing="ij")
    wave_number = real_t(4)
    rhs_field = (
        np.cos(wave_number * np.pi * x_grid)
        * np.cos(wave_number * np.pi * y_grid)
        * np.cos(wave_number * np.pi * z_grid)
    )
    correct_solution_field = rhs_field / (3 * (wave_number**2) * np.pi**2)
    solution_field = np.zeros_like(rhs_field)

    unbounded_poisson_solver = FastDiagPoissonSolver3D(
        grid_size_z=n_values,
        grid_size_y=n_values,
        grid_size_x=n_values,
        dx=dx,
        real_t=real_t,
        bc_type="homogenous_neumann_along_xyz",
    )
    unbounded_poisson_solver_kernel = unbounded_poisson_solver.solve
    unbounded_poisson_solver_kernel(solution_field=solution_field, rhs_field=rhs_field)

    error_field = solution_field - correct_solution_field
    linf_norm_error = np.amax(np.fabs(error_field))
    # check error less than 0.01%
    error_tol = real_t(1e-4)
    assert linf_norm_error <= error_tol


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_unbounded_poisson_solve_vector_field_neumann_fast_diag_3d(n_values, precision):
    real_t = get_real_t(precision)
    dx = real_t(1.0 / n_values)
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, n_values).astype(real_t)
    z_grid, y_grid, x_grid = np.meshgrid(x, x, x, indexing="ij")
    dim = 3
    rhs_vector_field = np.zeros((dim, n_values, n_values, n_values), dtype=real_t)
    wave_number = real_t(4)
    for i in range(dim):
        rhs_vector_field[i] = (
            np.cos(wave_number * np.pi * x_grid)
            * np.cos(wave_number * np.pi * y_grid)
            * np.cos(wave_number * np.pi * z_grid)
        )
    correct_solution_vector_field = rhs_vector_field / (
        3 * (wave_number**2) * np.pi**2
    )
    solution_vector_field = np.zeros_like(rhs_vector_field)

    unbounded_poisson_solver = FastDiagPoissonSolver3D(
        grid_size_z=n_values,
        grid_size_y=n_values,
        grid_size_x=n_values,
        dx=dx,
        real_t=real_t,
        bc_type="homogenous_neumann_along_xyz",
    )
    unbounded_poisson_solver_kernel = unbounded_poisson_solver.vector_field_solve
    unbounded_poisson_solver_kernel(
        solution_vector_field=solution_vector_field, rhs_vector_field=rhs_vector_field
    )

    error_field = solution_vector_field - correct_solution_vector_field
    linf_norm_error = np.amax(np.fabs(error_field))
    # check error less than 0.01%
    error_tol = real_t(1e-4)
    assert linf_norm_error <= error_tol
